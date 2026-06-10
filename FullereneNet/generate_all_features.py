import sys
import os
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "feature"))

from node_feature_generate import custom_sort, generate_node_feature_and_edge_index
from edge_feature_generate import generate_edge_feature
from ring_feature_generate_pentagon import generate_ring_feature_pentagon
from ring_feature_generate_hexagon import generate_ring_feature_hexagon

OPT_XYZ = os.path.join(os.path.dirname(__file__), "data", "optimized_xyz")
FEAT_DIR = os.path.join(os.path.dirname(__file__), "feature")


def get_xyz_files(size_dirs):
    files = []
    for d in size_dirs:
        files.extend(glob.glob(os.path.join(OPT_XYZ, d, "*.xyz")))
    return sorted(files, key=custom_sort)


def generate(name, size_dirs):
    print(f"\n=== Generating features for {name} ===")
    files = get_xyz_files(size_dirs)
    print(f"Found {len(files)} XYZ files")

    generate_node_feature_and_edge_index(
        files,
        edge_index_name=os.path.join(FEAT_DIR, f"edge_index_{name}.pt"),
        node_feature_name=os.path.join(FEAT_DIR, f"node_feature_{name}.pt"),
    )
    print("  node_feature and edge_index done")

    generate_edge_feature(
        files,
        edge_feature_name=os.path.join(FEAT_DIR, f"edge_feature_{name}.pt"),
    )
    print("  edge_feature done")

    generate_ring_feature_pentagon(
        files,
        ring_feature_name=os.path.join(FEAT_DIR, f"ring_feature_pentagon_{name}.pt"),
    )
    print("  ring_feature_pentagon done")

    generate_ring_feature_hexagon(
        files,
        ring_feature_hexagon_name=os.path.join(FEAT_DIR, f"ring_feature_hexagon_{name}.pt"),
    )
    print("  ring_feature_hexagon done")


C20_TO_C60 = [f"c{n}" for n in [20, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]]
C70 = ["c70"]
C72_TO_C100 = [f"c{n}" for n in [72, 74, 76, 78, 80, 82, 84, 86, 90, 92, 94, 96, 98, 100]]

generate("c60", C20_TO_C60)
generate("c70_non_IPR", C70)
generate("c72_100_IPR", C72_TO_C100)

print("\nAll features generated.")
