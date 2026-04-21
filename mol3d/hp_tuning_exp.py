import time
import json
import argparse
from typing import Literal
import optuna

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from data_loader.mol3d import Mol3d_KHopLifting, Mol3d_KNNLifting, Mol3d_KernelLifting, Mol3d_CycleLifting

DATASETS = {"KHop": Mol3d_KHopLifting, "KNN": Mol3d_KNNLifting, "Kernel": Mol3d_KernelLifting, "Cycle": Mol3d_CycleLifting}

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True, help="Path to save results JSON")
parser.add_argument("--lifting", required=True)
parser.add_argument("--model",required=True)
parser.add_argument("--datapath", required=True)
parser.add_argument("--datasize", required=True)
parser.add_argument("--ntrials",required=True)
args = parser.parse_args()

# ---Layers-------------------------- 

class CCConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False , normalize=False ,update_func: Literal["sigmoid", "relu", "leakyrelu", None] = None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_channels, in_channels, dtype=torch.float32))
        nn.init.xavier_uniform_(self.W)
        self.normalize = normalize
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm1d(out_channels) # Batchnorm
        self.update_func = update_func

    def _activate(self, x):
        match self.update_func:
            case "sigmoid":   return torch.sigmoid(x)
            case "relu":      return F.relu(x)
            case "leakyrelu": return F.leaky_relu(x)
            case _:           return x

    def forward(self, x, neighborhood):
        x_proj = x @ self.W.T
        agg = neighborhood @ x_proj
        if self.normalize:
            deg = neighborhood.sum(dim=1).to_dense().clamp(min=1).unsqueeze(1)
            agg = agg / deg
        if self.batch_norm:
            return self._activate(self.bn(agg))
        else:
            return self._activate(agg)
 
class CCAttLayer(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False, update_func: Literal["sigmoid", "relu", "leakyrelu", None] = None):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.update_func  = update_func

        self.Ws  = nn.Parameter(torch.empty(out_channels, in_channels,  dtype=torch.float32))
        self.Wt  = nn.Parameter(torch.empty(in_channels,  out_channels, dtype=torch.float32))
        self.a_s = nn.Parameter(torch.empty(out_channels, dtype=torch.float32))
        self.a_t = nn.Parameter(torch.empty(in_channels,  dtype=torch.float32))

        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm1d(out_channels)

        nn.init.xavier_uniform_(self.Ws)
        nn.init.xavier_uniform_(self.Wt)
        nn.init.xavier_uniform_(self.a_s.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_t.unsqueeze(0))

    def _activate(self, x):
        match self.update_func:
            case "sigmoid":   return torch.sigmoid(x)
            case "relu":      return F.relu(x)
            case "leakyrelu": return F.leaky_relu(x)
            case _:           return x

    def forward(self, x, neighborhood, x_target):
        neighborhood = neighborhood.coalesce()
        Z_s = x        @ self.Ws.T   # (N_src, out_ch)
        Z_t = x_target @ self.Wt.T   # (N_tgt, in_ch)

        rows, cols = neighborhood.indices()   # rows=tgt, cols=src
        e = (Z_s[cols] @ self.a_s) + (Z_t[rows] @ self.a_t)
        e = F.leaky_relu(e, negative_slope=0.2)

        e_max = torch.zeros(x_target.shape[0], device=x.device)
        e_max.scatter_reduce_(0, rows, e, reduce='amax', include_self=True)
        e_exp = torch.exp(e - e_max[rows])
        e_sum = torch.zeros(x_target.shape[0], device=x.device)
        e_sum.scatter_add_(0, rows, e_exp)
        att = e_exp / (e_sum[rows] + 1e-8)

        weighted = att.unsqueeze(1) * Z_s[cols]
        out = torch.zeros(x_target.shape[0], Z_s.shape[1], device=x.device)
        out.scatter_add_(0, rows.unsqueeze(1).expand_as(weighted), weighted)
        if self.batch_norm:
            return self._activate(self.bn(out))
        else:
            return self._activate(out)
        
class AdvConvLayer(nn.Module):
    def __init__(self,channels_rk0 ,channels_rk1, channels_rk2, batch_norm, normalize):
        super().__init__()
        self.conv_0_to_1 = CCConvLayer(channels_rk0, channels_rk1, update_func="relu", batch_norm=batch_norm, normalize=normalize)
        self.conv_1_to_2 = CCConvLayer(channels_rk1,  channels_rk2, update_func="relu", batch_norm=batch_norm, normalize=normalize)
        self.conv_0_to_2 = CCConvLayer(channels_rk0, channels_rk2, update_func="relu", batch_norm=batch_norm, normalize=normalize)

        self.conv_2_to_1 = CCConvLayer(channels_rk2, channels_rk1, update_func="relu", batch_norm=batch_norm, normalize=normalize)
        self.conv_2_to_0 = CCConvLayer(channels_rk2,  channels_rk0, update_func="relu", batch_norm=batch_norm, normalize=normalize)
        self.conv_1_to_0 = CCConvLayer(channels_rk1, channels_rk0, update_func="relu", batch_norm=batch_norm, normalize=normalize)

    def forward(self,x_0,icd01,icd02,icd12,x_1=None,x_2=None):
        if x_1 is not None:
            x_1_new = (self.conv_0_to_1(x_0, icd01.T) + x_1)/2
        else:
            x_1_new = self.conv_0_to_1(x_0, icd01.T)

        if x_2 is not None:
            x_2_new = (self.conv_0_to_2(x_0, icd02.T) + self.conv_1_to_2(x_1,icd12.T) + x_2)/3
        else:
            x_2_new = (self.conv_0_to_2(x_0, icd02.T) + self.conv_1_to_2(x_1_new,icd12.T))/2

        x_1_new_new = (self.conv_2_to_1(x_2_new, icd12) + x_1_new)/2
        x_0_new = (self.conv_2_to_0(x_2_new, icd02) + self.conv_1_to_0(x_1_new,icd01) + x_0)/3

        return x_0_new, x_1_new_new, x_2_new
    
class AdvAttLayer(nn.Module):
    def __init__(self,channels_rk0 ,channels_rk1, channels_rk2, batch_norm):
        super().__init__()
        self.att_0_to_1 = CCAttLayer(channels_rk0, channels_rk1, update_func="relu", batch_norm=batch_norm)
        self.att_1_to_2 = CCAttLayer(channels_rk1,  channels_rk2, update_func="relu", batch_norm=batch_norm)
        self.att_0_to_2 = CCAttLayer(channels_rk0, channels_rk2, update_func="relu", batch_norm=batch_norm)

        self.att_2_to_1 = CCAttLayer(channels_rk2, channels_rk1, update_func="relu", batch_norm=batch_norm)
        self.att_2_to_0 = CCAttLayer(channels_rk2,  channels_rk0, update_func="relu", batch_norm=batch_norm)
        self.att_1_to_0 = CCAttLayer(channels_rk1, channels_rk0, update_func="relu", batch_norm=batch_norm)

    def forward(self,x_0,x_1,x_2,icd01,icd02,icd12):
        x_1_new = (self.att_0_to_1(x_0, icd01.T,x_1) + x_1)/2
        x_2_new = (self.att_0_to_2(x_0, icd02.T,x_2) + self.att_1_to_2(x_1,icd12.T,x_2) + x_2)/3

        x_1_new_new = (self.att_2_to_1(x_2_new, icd12,x_1_new) + x_1_new)/2
        x_0_new = (self.att_2_to_0(x_2_new, icd02,x_0) + self.att_1_to_0(x_1_new,icd01,x_0) + x_0)/3

        return x_0_new, x_1_new_new, x_2_new

# ---Models-------------------------- 

class Simple_Conv_TNN(nn.Module):
    def __init__(self, in_channels, channels_rk0 ,channels_rk1, channels_rk2, channels_rk3,
                 size_hidden_layer1, size_hidden_layer2, output_channels, lr, batch_norm, gradient_clipping, normalize, epochs):
        super().__init__()

        self.lr = lr
        self.batch_norm = batch_norm
        self.normalize = normalize
        self.gradient_clipping = gradient_clipping
        self.epochs = epochs

        self.proj = nn.Linear(in_channels, channels_rk0)

        self.conv_0_to_1 = CCConvLayer(channels_rk0, channels_rk1, update_func="relu", batch_norm=batch_norm, normalize=normalize)
        self.conv_1_to_2 = CCConvLayer(channels_rk1,  channels_rk2, update_func="relu", batch_norm=batch_norm, normalize=normalize)
        self.conv_2_to_3 = CCConvLayer(channels_rk2, channels_rk3, update_func="relu", batch_norm=batch_norm, normalize=normalize)

        self.fc1 = nn.Linear(channels_rk3, size_hidden_layer1)
        self.fc2 = nn.Linear(size_hidden_layer1, size_hidden_layer2)
        self.fc3 = nn.Linear(size_hidden_layer2, output_channels)

    def forward(self, x, icd01, icd02, icd03, icd12, icd13, icd23):

        x = x.to(torch.float32)
        x_0 = self.proj(x)

        x_1 = self.conv_0_to_1(x_0, icd01.T)
        x_2 = self.conv_1_to_2(x_1, icd12.T)
        x_3 = self.conv_2_to_3(x_2, icd23.T)

        x_out = F.relu(self.fc1(x_3))
        x_out = F.relu(self.fc2(x_out))
        x_out = self.fc3(x_out)

        return x_out
    
    def fit(self, dataloader, device):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        self.to(device)
        self.train()
        epoch_losses = []
        epoch_times = []
        for epoch in range(self.epochs):
            t0 = time.time()
            total_loss = 0
            for x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap in dataloader:
                x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap = (
                x_0.to(device), icd_0_1.to(device), icd_0_2.to(device), icd_0_3.to(device),
                icd_1_2.to(device),icd_1_3.to(device), icd_2_3.to(device), hlgap.to(device))
                optimizer.zero_grad()
                output = self(x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3).squeeze(-1)
                loss = criterion(output, hlgap.squeeze(-1))
                loss.backward()
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            epoch_time = time.time() - t0
            epoch_losses.append(avg_loss)
            epoch_times.append(epoch_time)
            print(f"Epoch {epoch + 1:2d}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")
        return epoch_losses, epoch_times

    def test(self, dataloader, device):
        mae = 0.0
        self.eval()
        with torch.no_grad():
            for x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap in dataloader:
                x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap = (
                x_0.to(device), icd_0_1.to(device), icd_0_2.to(device), icd_0_3.to(device),
                icd_1_2.to(device),icd_1_3.to(device), icd_2_3.to(device), hlgap.to(device))
                output = self(x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3).squeeze(-1)
                mae += (output.squeeze(-1) - hlgap).abs().mean().item()
        mae /= len(dataloader)
        return mae

class Simple_Att_TNN(nn.Module):
    def __init__(self, in_channels, channels_rk0 ,channels_rk1, channels_rk2, channels_rk3,
                 size_hidden_layer1, size_hidden_layer2, output_channels, lr, batch_norm, gradient_clipping, normalize, epochs):
        super().__init__()

        self.lr = lr
        self.batch_norm = batch_norm
        self.normalize = normalize
        self.gradient_clipping = gradient_clipping
        self.epochs = epochs

        self.proj = nn.Linear(in_channels, channels_rk0)

        self.conv_0_to_1 = CCConvLayer(channels_rk0, channels_rk1, update_func="relu", batch_norm=batch_norm, normalize=normalize)
        self.conv_1_to_2 = CCConvLayer(channels_rk1,  channels_rk2, update_func="relu", batch_norm=batch_norm, normalize=normalize)
        self.conv_2_to_3 = CCConvLayer(channels_rk2, channels_rk3, update_func="relu", batch_norm=batch_norm, normalize=normalize)

        self.att_0_to_1 = CCAttLayer(channels_rk0, channels_rk1, update_func="relu", batch_norm=batch_norm)
        self.att_1_to_2 = CCAttLayer(channels_rk1,  channels_rk2, update_func="relu", batch_norm=batch_norm)
        self.att_2_to_3 = CCAttLayer(channels_rk2, channels_rk3, update_func="relu", batch_norm=batch_norm)

        self.fc1 = nn.Linear(channels_rk3, size_hidden_layer1)
        self.fc2 = nn.Linear(size_hidden_layer1, size_hidden_layer2)
        self.fc3 = nn.Linear(size_hidden_layer2, output_channels)

    def forward(self, x, icd01, icd02, icd03, icd12, icd13, icd23):

        x = x.to(torch.float32)
        x_0 = self.proj(x)

        x_1 = self.conv_0_to_1(x_0, icd01.T)
        x_2 = self.conv_1_to_2(x_1, icd12.T)
        x_3 = self.conv_2_to_3(x_2, icd23.T)

        x_1_new = self.att_0_to_1(x_0, icd01.T, x_1)
        x_2_new = self.att_1_to_2(x_1_new, icd12.T, x_2)
        x_3_new = self.att_2_to_3(x_2_new, icd23.T, x_3)

        x_out = F.relu(self.fc1(x_3_new))
        x_out = F.relu(self.fc2(x_out))
        x_out = self.fc3(x_out)

        return x_out
    
    def fit(self, dataloader, device):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        self.to(device)
        self.train()
        epoch_losses = []
        epoch_times = []
        for epoch in range(self.epochs):
            t0 = time.time()
            total_loss = 0
            for x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap in dataloader:
                x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap = (
                x_0.to(device), icd_0_1.to(device), icd_0_2.to(device), icd_0_3.to(device),
                icd_1_2.to(device),icd_1_3.to(device), icd_2_3.to(device), hlgap.to(device))
                optimizer.zero_grad()
                output = self(x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3).squeeze(-1)
                loss = criterion(output, hlgap.squeeze(-1))
                loss.backward()
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            epoch_time = time.time() - t0
            epoch_losses.append(avg_loss)
            epoch_times.append(epoch_time)
            print(f"Epoch {epoch + 1:2d}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")
        return epoch_losses, epoch_times

    def test(self, dataloader, device):
        mae = 0.0
        self.eval()
        with torch.no_grad():
            for x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap in dataloader:
                x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap = (
                x_0.to(device), icd_0_1.to(device), icd_0_2.to(device), icd_0_3.to(device),
                icd_1_2.to(device),icd_1_3.to(device), icd_2_3.to(device), hlgap.to(device))
                output = self(x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3).squeeze(-1)
                mae += (output.squeeze(-1) - hlgap).abs().mean().item()
        mae /= len(dataloader)
        return mae

class Adv_Conv_TNN(nn.Module):
    def __init__(self, in_channels, channels_rk0 ,channels_rk1, channels_rk2, channels_rk3,
                 size_hidden_layer1, size_hidden_layer2, output_channels, lr, batch_norm, gradient_clipping, normalize, epochs):
        super().__init__()

        self.lr = lr
        self.batch_norm = batch_norm
        self.normalize = normalize
        self.gradient_clipping = gradient_clipping
        self.epochs = epochs

        self.proj = nn.Linear(in_channels, channels_rk0)

        self.adv_conv_layer1 = AdvConvLayer(channels_rk0 ,channels_rk1, channels_rk2, batch_norm=batch_norm, normalize=normalize)
        self.adv_conv_layer2 = AdvConvLayer(channels_rk0 ,channels_rk1, channels_rk2, batch_norm=batch_norm, normalize=normalize)

        self.conv_0_to_3 = CCConvLayer(channels_rk0, channels_rk3, update_func="relu", batch_norm=batch_norm, normalize=normalize)
        self.conv_1_to_3 = CCConvLayer(channels_rk1, channels_rk3, update_func="relu", batch_norm=batch_norm, normalize=normalize)
        self.conv_2_to_3 = CCConvLayer(channels_rk2, channels_rk3, update_func="relu", batch_norm=batch_norm, normalize=normalize)

        self.fc1 = nn.Linear(channels_rk3, size_hidden_layer1)
        self.fc2 = nn.Linear(size_hidden_layer1, size_hidden_layer2)
        self.fc3 = nn.Linear(size_hidden_layer2, output_channels)

    def forward(self, x, icd01, icd02, icd03, icd12, icd13, icd23):

        x = x.to(torch.float32)

        x_0 = self.proj(x)

        x_0, x_1, x_2 = self.adv_conv_layer1(x_0,icd01,icd02,icd12)
        x_0, x_1, x_2 = self.adv_conv_layer2(x_0,icd01,icd02,icd12,x_1=x_1,x_2=x_2)

        x_3 = (self.conv_0_to_3(x_0, icd03.T) + self.conv_1_to_3(x_1, icd13.T) + self.conv_2_to_3(x_2, icd23.T))/3

        x_out = F.relu(self.fc1(x_3))
        x_out = F.relu(self.fc2(x_out))
        x_out = self.fc3(x_out)

        return x_out
    
    def fit(self, dataloader, device):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        self.to(device)
        self.train()
        epoch_losses = []
        epoch_times = []
        for epoch in range(self.epochs):
            t0 = time.time()
            total_loss = 0
            for x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap in dataloader:
                x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap = (
                x_0.to(device), icd_0_1.to(device), icd_0_2.to(device), icd_0_3.to(device),
                icd_1_2.to(device),icd_1_3.to(device), icd_2_3.to(device), hlgap.to(device))
                optimizer.zero_grad()
                output = self(x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3).squeeze(-1)
                loss = criterion(output, hlgap.squeeze(-1))
                loss.backward()
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            epoch_time = time.time() - t0
            epoch_losses.append(avg_loss)
            epoch_times.append(epoch_time)
            print(f"Epoch {epoch + 1:2d}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")
        return epoch_losses, epoch_times

    def test(self, dataloader, device):
        mae = 0.0
        self.eval()
        with torch.no_grad():
            for x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap in dataloader:
                x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap = (
                x_0.to(device), icd_0_1.to(device), icd_0_2.to(device), icd_0_3.to(device),
                icd_1_2.to(device),icd_1_3.to(device), icd_2_3.to(device), hlgap.to(device))
                output = self(x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3).squeeze(-1)
                mae += (output.squeeze(-1) - hlgap).abs().mean().item()
        mae /= len(dataloader)
        return mae

class Adv_Att_TNN(nn.Module):
    def __init__(self, in_channels, channels_rk0 ,channels_rk1, channels_rk2, channels_rk3,
                 size_hidden_layer1, size_hidden_layer2, output_channels, lr, batch_norm, gradient_clipping, normalize, epochs):
        super().__init__()

        self.lr = lr
        self.batch_norm = batch_norm
        self.normalize = normalize
        self.gradient_clipping = gradient_clipping
        self.epochs = epochs

        self.proj = nn.Linear(in_channels, channels_rk0)

        self.conv_0_to_1 = CCConvLayer(channels_rk0, channels_rk1, update_func="relu", batch_norm=batch_norm, normalize=normalize)
        self.conv_1_to_2 = CCConvLayer(channels_rk1,  channels_rk2, update_func="relu", batch_norm=batch_norm, normalize=normalize)
        self.conv_0_to_2 = CCConvLayer(channels_rk0, channels_rk2, update_func="relu", batch_norm=batch_norm, normalize=normalize)

        self.adv_att_layer1 = AdvAttLayer(channels_rk0 ,channels_rk1, channels_rk2, batch_norm=batch_norm)
        self.adv_att_layer2 = AdvAttLayer(channels_rk0 ,channels_rk1, channels_rk2, batch_norm=batch_norm)

        self.conv_0_to_3 = CCConvLayer(channels_rk0, channels_rk3, update_func="relu", batch_norm=batch_norm, normalize=normalize)
        self.conv_1_to_3 = CCConvLayer(channels_rk1, channels_rk3, update_func="relu", batch_norm=batch_norm, normalize=normalize)
        self.conv_2_to_3 = CCConvLayer(channels_rk2, channels_rk3, update_func="relu", batch_norm=batch_norm, normalize=normalize)

        self.fc1 = nn.Linear(channels_rk3, size_hidden_layer1)
        self.fc2 = nn.Linear(size_hidden_layer1, size_hidden_layer2)
        self.fc3 = nn.Linear(size_hidden_layer2, output_channels)

    def forward(self, x, icd01, icd02, icd03, icd12, icd13, icd23):

        x = x.to(torch.float32)

        x_0 = self.proj(x)
        x_1 = self.conv_0_to_1(x_0, icd01.T)
        x_2 = (self.conv_0_to_2(x_0, icd02.T) + self.conv_1_to_2(x_1,icd12.T))/2

        x_0,x_1,x_2 = self.adv_att_layer1(x_0,x_1,x_2, icd01,icd02,icd12)
        x_0,x_1,x_2 = self.adv_att_layer2(x_0,x_1,x_2, icd01,icd02,icd12)

        x_3 = (self.conv_0_to_3(x_0, icd03.T) + self.conv_1_to_3(x_1, icd13.T) + self.conv_2_to_3(x_2, icd23.T))/3

        x_out = F.relu(self.fc1(x_3))
        x_out = F.relu(self.fc2(x_out))
        x_out = self.fc3(x_out)

        return x_out
    
    def fit(self, dataloader, device):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        self.to(device)
        self.train()
        epoch_losses = []
        epoch_times = []
        for epoch in range(self.epochs):
            t0 = time.time()
            total_loss = 0
            for x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap in dataloader:
                x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap = (
                x_0.to(device), icd_0_1.to(device), icd_0_2.to(device), icd_0_3.to(device),
                icd_1_2.to(device),icd_1_3.to(device), icd_2_3.to(device), hlgap.to(device))
                optimizer.zero_grad()
                output = self(x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3).squeeze(-1)
                loss = criterion(output, hlgap.squeeze(-1))
                loss.backward()
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            epoch_time = time.time() - t0
            epoch_losses.append(avg_loss)
            epoch_times.append(epoch_time)
            print(f"Epoch {epoch + 1:2d}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")
        return epoch_losses, epoch_times

    def test(self, dataloader, device):
        mae = 0.0
        self.eval()
        with torch.no_grad():
            for x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap in dataloader:
                x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3, hlgap = (
                x_0.to(device), icd_0_1.to(device), icd_0_2.to(device), icd_0_3.to(device),
                icd_1_2.to(device),icd_1_3.to(device), icd_2_3.to(device), hlgap.to(device))
                output = self(x_0, icd_0_1, icd_0_2, icd_0_3, icd_1_2, icd_1_3,icd_2_3).squeeze(-1)
                mae += (output.squeeze(-1) - hlgap).abs().mean().item()
        mae /= len(dataloader)
        return mae

class Full_Att_TNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

MODELS = {"Simple_Conv":Simple_Conv_TNN,"Simple_Att":Simple_Att_TNN,"Adv_Conv":Adv_Conv_TNN,"Adv_Att":Adv_Att_TNN,}

# ---Collation helpers---------------

def sparse_block_diag(sparse_list):
    device = sparse_list[0].device
    dtype  = sparse_list[0].dtype
    rows, cols, vals = [], [], []
    row_offset = col_offset = 0
    for S in sparse_list:
        S = S.coalesce()
        i, v = S.indices(), S.values()
        n_rows, n_cols = S.shape
        rows.append(i[0] + row_offset)
        cols.append(i[1] + col_offset)
        vals.append(v)
        row_offset += n_rows
        col_offset += n_cols
    indices = torch.stack([torch.cat(rows), torch.cat(cols)])
    return torch.sparse_coo_tensor(indices, torch.cat(vals),
                                   size=(row_offset, col_offset),
                                   device=device, dtype=dtype)


def collate(batch):
    node_feat, icd01, icd02,icd03, icd12,icd13, icd23, homolumogap = zip(*batch)
    return (
        torch.cat(node_feat, dim=0),   # x_0: atomic numbers
        sparse_block_diag(icd01),
        sparse_block_diag(icd02),
        sparse_block_diag(icd03),
        sparse_block_diag(icd12),
        sparse_block_diag(icd13),
        sparse_block_diag(icd23),
        torch.cat(homolumogap, dim=0),
    )

# ---Prepare Data -------------------

datasize = int(args.datasize)

dataset = DATASETS[args.lifting](root=args.datapath, size= datasize)
print(len(dataset))

batch_size = 64
train_dataloader = DataLoader(Subset(dataset, range(int(0.6 *  datasize))),
                              batch_size=batch_size, shuffle=False, collate_fn=collate)
val_dataloader   = DataLoader(Subset(dataset, range(int(0.6 * datasize), int(0.8 * datasize))),
                              batch_size=batch_size, collate_fn=collate)
test_dataloader  = DataLoader(Subset(dataset, range(int(0.8 * datasize), len(dataset))),
                              batch_size=batch_size, collate_fn=collate)

# ---HP Tuning ----------------------

HYPERPARAMS = [("epochs", "int", (10,100)),("in_channels", "fixed", 4),("channels_rk0", "int", (16,128)),("channels_rk1", "int", (16,128)),
                ("channels_rk2", "int", (16,128)),("channels_rk3", "int", (16,128)),("size_hidden_layer1", "int", (16,128)),
                ("size_hidden_layer2", "int", (16,128)),("output_channels","fixed",1),("lr","float",(1e-4,1e-1)),
                ("batch_norm","categorical",[True, False]),("gradient_clipping","categorical",[True, False]),("normalize","categorical",[True, False])]

# in_channels, channels_rk0 ,channels_rk1, channels_rk2, channels_rk3,
#                 size_hidden_layer1, size_hidden_layer2, output_channels, lr, batch_norm, gradient_clipping, normalize, epochs

def hp_optimization(model, hps, n_trials,device="cpu"):
    fixed_params = {}
    for hp in hps:
        if hp[1] == "fixed":
            fixed_params[hp[0]] = hp[2]

    def objective(trial):
        hp_trial_dict = {}
        
        for hp in hps:
            if hp[1] == "categorical":
                hp_trial_dict[hp[0]] = trial.suggest_categorical(hp[0], hp[2])
            elif hp[1] == "int":
                hp_trial_dict[hp[0]] = trial.suggest_int(hp[0], hp[2][0], hp[2][1])
            elif hp[1] == "float":
                hp_trial_dict[hp[0]] = trial.suggest_float(hp[0], hp[2][0], hp[2][1])

        cur_model = model(**hp_trial_dict,**fixed_params)
        cur_model.fit(train_dataloader,device=device)
        mae = cur_model.test(val_dataloader,device=device)
        return mae
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_value, {**study.best_params, **fixed_params}

# ---Main ----------------------
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print(device)

model = MODELS[args.model]

best_val_mae, best_hps = hp_optimization(model, HYPERPARAMS, n_trials=int(args.ntrials), device=device)

cur_model = model(**best_hps)
epoch_losses, epoch_times = cur_model.fit(train_dataloader, device=device)
mae = cur_model.test(test_dataloader, device=device)

print(f"Test MAE: {mae}")

results = {
    "best_hps": best_hps,
    "best_val_mae": best_val_mae,
    "epoch_losses": epoch_losses,
    "epoch_times": epoch_times,
    "test_mae": mae,
}
with open(args.path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {args.path}")