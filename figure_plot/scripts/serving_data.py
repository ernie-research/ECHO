"""Load the compact serving metrics used by Figures 3b and 6."""
import csv

from plot_paths import INPUT_DIR


ARMS = [
    ("echo_mtp", "ECHO", "tab:red", "o", "-"),
    ("eagle3_cold_final", "EAGLE3-cold", "tab:blue", "s", "-"),
    ("eagle3_wdsnospec_ep2", "EAGLE3-warm", "darkgoldenrod", "P", "-"),
    ("dspark_b7", "DSpark", "tab:green", "^", "-"),
    ("dflash_b8", "DFlash", "tab:brown", "*", "-"),
]


def load_serving_data():
    """Return entropy, group membership, accepted length, and decode speedup."""
    entropy, groups, accepted, speedup = {}, {}, {}, {}
    with (INPUT_DIR / "serving_metrics.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"]
            method = row["method"]
            entropy[dataset] = float(row["entropy"])
            groups[dataset] = row["group"]
            accepted[(method, dataset)] = float(row["accepted_length"])
            speedup[(method, dataset)] = float(row["decode_speedup"])
    return entropy, groups, accepted, speedup
