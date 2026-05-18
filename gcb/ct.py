import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadPCA(nn.Module):
    def __init__(self, source_dim, target_dim, p, num_heads=1):
        super().__init__()

        self.num_heads = num_heads

        if source_dim % num_heads + target_dim % num_heads != 0:
            raise ValueError(f"source_dim ({source_dim}) and target_dim ({target_dim}) must both be divisible by num_heads ({num_heads})")
        
        self.source_dims = source_dim // num_heads
        self.target_dims = target_dim // num_heads

        self.att_layers = nn.ModuleList([PCAttention(self.source_dims, self.target_dims, p) for _ in range(num_heads)])

    def forward(self, x_source, x_target, neighborhood):
        outputs = torch.zeros_like(x_target)
        for i in range(self.num_heads):
            s = i * self.source_dims
            t = i * self.target_dims
            outputs[:, t:(t + self.target_dims)] = self.att_layers[i](
                x_source[:, s:(s + self.source_dims)], x_target[:, t:(t + self.target_dims)], neighborhood)
        return outputs


# SINGLE HEAD
class PCAttention(nn.Module):
    def __init__(self, source_dim, target_dim, p):
        super().__init__()

        self.source_dim = source_dim
        self.target_dim = target_dim
        self.p = p

        self.Q = nn.Parameter(torch.empty(target_dim, p, dtype=torch.float32))
        self.K = nn.Parameter(torch.empty(source_dim, p, dtype=torch.float32))
        self.V = nn.Parameter(torch.empty(source_dim, target_dim, dtype=torch.float32))

        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.V)

    def forward(self, x_source, x_target, neighborhood):
        query = x_target @ self.Q
        key = (x_source @ self.K).T
        value = x_source @ self.V
        out = torch.sparse.softmax((query @ key) * neighborhood, dim=-1) @ value
        return out
    
class PairwiseAttentionTransformer(nn.Module):
    def __init__(self, sources_dims, target_dim, num_heads, p, dropout):
        super().__init__()
        self.num_sources = len(sources_dims)
        self.sources_dims = sources_dims
        self.target_dim = target_dim
        self.num_heads = num_heads
        self.p = p
        self.dropout = dropout

        self.att_layers = nn.ModuleList([MultiheadPCA(sd, target_dim,p,num_heads) for sd in sources_dims])
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(sd) for sd in sources_dims])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(target_dim) for _ in sources_dims])
        self.target_layernorm = nn.LayerNorm(target_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.ffn1 = nn.ModuleList([nn.Linear(target_dim,target_dim) for _ in range(self.num_sources)])
        self.ffn2 = nn.ModuleList([nn.Linear(target_dim,target_dim) for _ in range(self.num_sources)])

    def forward(self, sources, x_target, neighborhoods):
        x_t1 = self.target_layernorm(x_target)
        x_s1_list = [ln(x_s) for ln,x_s in zip(self.layer_norms1, sources)]
        x_2_list = [att(x_s,x_t1,n) for att, x_s, n in zip(self.att_layers, x_s1_list, neighborhoods)]
        x_3_list = [self.dropout(x_2) for x_2 in x_2_list]
        x_4_list = [x_t1 + x_3 for x_3 in x_3_list]
        x_5_list = [ln(x_5) for ln,x_5 in zip(self.layer_norms2,x_4_list)]
        x_6_list = [self.dropout(ffn2(self.dropout(F.relu(ffn1(x_5))))) for x_5, ffn1, ffn2 in zip(x_5_list, self.ffn1, self.ffn2)]

        return torch.stack(x_6_list).sum(dim=0)


class CellularTransformer(nn.Module):
    def __init__(self, rk0_dim, rk1_dim, rk2_dim, output_dim ,num_layers, hidden_dim, num_heads,hidden_dim_per_head, att_dropout, emb_dropout, readout_dropout, num_readout_hidden_layers):
        super().__init__()
        self.rk0_dim = rk0_dim
        self.rk1_dim = rk1_dim
        self.rk2_dim = rk2_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.hidden_dim_per_head = hidden_dim_per_head
        self.att_dropout = att_dropout
        self.emb_dropout = emb_dropout
        self.readout_dropout = readout_dropout
        self.num_readout_hidden_layers = num_readout_hidden_layers


        #embedding layers
        self.emb_dropout_layer = nn.Dropout(p=emb_dropout)
        self.emb0 = nn.Linear(rk0_dim, hidden_dim)
        self.emb1 = nn.Linear(rk1_dim, hidden_dim)
        self.emb2 = nn.Linear(rk2_dim, hidden_dim)

        #pat-layers
        self.pat_layers_rk0 = nn.ModuleList([PairwiseAttentionTransformer([hidden_dim for _ in range(3)],hidden_dim,num_heads,p=hidden_dim_per_head,dropout=att_dropout) for _ in range(num_layers) ])
        self.pat_layers_rk1 = nn.ModuleList([PairwiseAttentionTransformer([hidden_dim for _ in range(3)],hidden_dim,num_heads,p=hidden_dim_per_head,dropout=att_dropout) for _ in range(num_layers) ])
        self.pat_layers_rk2 = nn.ModuleList([PairwiseAttentionTransformer([hidden_dim for _ in range(3)],hidden_dim,num_heads,p=hidden_dim_per_head,dropout=att_dropout) for _ in range(num_layers) ])

        #readout
        self.readout_dropout_layer = nn.Dropout(p=readout_dropout)
        if num_readout_hidden_layers == 0:
            self.readout_layers = nn.ModuleList([nn.Linear(hidden_dim, output_dim)])
        else:
            self.readout_layers = nn.ModuleList([nn.Linear(hidden_dim//2**(i),hidden_dim//(2**(i+1))) for i in range(num_readout_hidden_layers)])
            self.readout_layers.append(nn.Linear(hidden_dim//(2**num_readout_hidden_layers),output_dim))

    def _readout(self, x, node_counts):
        for layer in self.readout_layers[:-1]:
            x = self.readout_dropout_layer(F.relu(layer(x)))
        x = self.readout_layers[-1](x)
        graphs = x.split(node_counts.tolist(), dim=0)
        return torch.stack([g.sum(dim=0) for g in graphs])

    def forward(self, x_0, x_1, x_2, adj00, icd01, adj11, icd02, icd12, adj22, node_counts):
        #embedding
        x_0 = self.emb_dropout_layer(self.emb0(x_0))
        x_1 = self.emb_dropout_layer(self.emb1(x_1))
        x_2 = self.emb_dropout_layer(self.emb2(x_2))

        #pat layers
        for i in range(self.num_layers):
            x_0_new = self.pat_layers_rk0[i]([x_0,x_1,x_2],x_0,[adj00, icd01, icd02])
            x_1_new = self.pat_layers_rk1[i]([x_0,x_1,x_2],x_1,[icd01.T, adj11, icd12])
            x_2_new = self.pat_layers_rk2[i]([x_0,x_1,x_2],x_2,[icd02.T, icd12.T, adj22])
            x_0, x_1, x_2 = x_0_new, x_1_new, x_2_new
        
        #readout
        x_out = self._readout(x_0, node_counts)

        return x_out
        