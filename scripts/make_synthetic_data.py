"""Create SYNTHETIC data with the UNSW-NB15 partitioned-file schema.

Only for smoke-testing the code and the Streamlit app when the real dataset is not
available. Numbers produced from this data are meaningless - never report them.

    python scripts/make_synthetic_data.py --out data/synthetic --n-train 6000 --n-test 3000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "sload", "dload",
    "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports",
]
CLASSES = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
           "Analysis", "Backdoor", "Shellcode", "Worms"]
WEIGHTS = np.array([0.32, 0.23, 0.19, 0.10, 0.07, 0.06, 0.012, 0.010, 0.006, 0.002])
PROTOS = ["tcp", "udp", "icmp", "arp", "ospf", "sctp"]
SERVICES = ["-", "http", "dns", "ftp", "smtp", "ssh", "pop3"]
STATES = ["FIN", "CON", "INT", "REQ", "RST"]


def generate(n, rng, shifts, proto_p):
    y = rng.choice(len(CLASSES), size=n, p=WEIGHTS / WEIGHTS.sum())
    base = rng.normal(2.0, 0.5, size=len(NUMERIC))
    X = np.exp(base + shifts[y] + rng.normal(0, 1.0, size=(n, len(NUMERIC))))
    df = pd.DataFrame(X, columns=NUMERIC)
    for c in ["is_ftp_login", "is_sm_ips_ports"]:
        df[c] = (df[c] > df[c].median() * 3).astype(int)
    df.insert(0, "id", np.arange(1, n + 1))
    df.insert(2, "proto", [rng.choice(PROTOS, p=proto_p[k]) for k in y])
    df.insert(3, "service", rng.choice(SERVICES, size=n))
    df.insert(4, "state", rng.choice(STATES, size=n))
    df["attack_cat"] = [CLASSES[k] for k in y]
    df["label"] = (y != 0).astype(int)
    # sprinkle a few dirty values to exercise the cleaning code
    df.loc[rng.choice(n, max(n // 500, 1), replace=False), "dur"] = np.inf
    df.loc[rng.choice(n, max(n // 500, 1), replace=False), "rate"] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    shifts = rng.normal(0, 0.9, size=(len(CLASSES), len(NUMERIC)))
    shifts[0] = 0
    proto_p = rng.dirichlet(np.ones(len(PROTOS)) * 1.5, size=len(CLASSES))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    generate(a.n_train, rng, shifts, proto_p).to_csv(out / "UNSW_NB15_training-set.csv", index=False)
    generate(a.n_test, rng, shifts, proto_p).to_csv(out / "UNSW_NB15_testing-set.csv", index=False)
    print(f"Synthetic files written to {out}/")


if __name__ == "__main__":
    main()
