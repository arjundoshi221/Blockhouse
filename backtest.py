#!/usr/bin/env python3
"""
Fast Cont‑Kukanov SOR back‑test  (≈ 1 min on i7‑10510U / 16 GB)
---------------------------------------------------------------
Models
    • STATIC                – plain allocator
    • UNCERTAINTY‑SIGMOID   – size haircut via logistic on z‑score × fill‑rate
    • UNCERTAINTY‑POWER     – size^α   (heavy–tail haircut)     × fill‑rate
Queue‑risk: θ_wait · E[T]   (wait‑time in seconds, scaled by 1/10)
Output
    • JSON (stdout)  + cumulative‑cost plot results_ALL.png
"""
# std‑lib only † numpy / pandas  (matplotlib imported lazily)
import argparse, json, time, warnings, multiprocessing as mp, itertools
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque, namedtuple
from functools import lru_cache
import matplotlib.pyplot as plt  # lazy import
import sys
import os

import numpy as np
import pandas as pd

out_dir = r"E:\PendriveBackup\Arjun2\CodingFiles-Arjun\ComputerScience\MSFM\Q1\interview_prep\BlockHouse\media"
CSV_PATH = r"E:\PendriveBackup\Arjun2\CodingFiles-Arjun\ComputerScience\MSFM\Q1\interview_prep\BlockHouse\data\l1_day.csv"


# ─────────────── Global constants ─────────────────────────────────── #
ORDER_SIZE = 5_000
ROLL = 100  # window for z‑score
ALPHA_TTR = 2 / 6  # ≈5‑s EWMA trade‑rate
QUEUE_SCALE = 10.0  # divide wait‑penalty by this
PLOT_DEFAULT = True

# parameter grids
GRID_COARSE = dict(
    lo=np.arange(0, 3.1, 1.0), lu=np.arange(0, 10.1, 2.0), th=np.arange(0, 10.1, 2.0)
)
GRID_FINE = dict(
    lo=np.arange(-0.5, 1.0, 0.5),
    lu=np.arange(-1.0, 1.5, 1.0),
    th=np.arange(-1.0, 1.5, 1.0),
)
N_PROC = max(mp.cpu_count() - 1, 1)

Venue = namedtuple("Venue", "id ask ask_sz fee rebate")


# ──────────────── helper classes  ─────────────────────────────────── #
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -10, 10)))


class FillRate:  # EWMA venue fill‑rate
    __slots__ = ("a", "v")

    def __init__(self, span=500):
        self.a = 2 / (span + 1)
        self.v = defaultdict(float)

    def update(self, vid, ex, posted):
        if posted:
            self.v[vid] = self.a * (ex / posted) + (1 - self.a) * self.v[vid]

    def get(self, vid):
        return max(self.v.get(vid, 0.7), 0.2)


class TradeRate:  # EWMA traded‑shares / sec
    __slots__ = ("v",)

    def __init__(self):
        self.v = defaultdict(float)

    def update(self, vid, sh, dt):
        if dt > 0:
            self.v[vid] = ALPHA_TTR * (sh / dt) + (1 - ALPHA_TTR) * self.v[vid]

    def get(self, vid):
        return max(self.v.get(vid, 1.0), 0.1)


# ─────────────── venue conversion  ────────────────────────────────── #
def row_to_venues(row, model, hist=None, fr=None):
    asks = row.filter(like="ask_px_00")
    sizes = row.filter(like="ask_sz_00")
    out = []
    for col, px, sz in zip(asks.index, asks, sizes):
        vid, sz = int(col.split("_")[-1]), int(sz)
        eff = sz
        if model.startswith("UNCERTAINTY"):
            dq = hist.setdefault(vid, deque(maxlen=ROLL))
            if model.endswith("SIGMOID"):
                mu, sd = (np.mean(dq) if dq else sz), (np.std(dq) if dq else 1)
                eff = int(sz * sigmoid((sz - mu) / sd if sd else 0))
            else:  # POWER α=0.6
                eff = int(sz**0.6)
            eff = int(eff * fr.get(vid))
            dq.append(sz)
        out.append(Venue(vid, float(px), eff, 0.002, 0.0002))
    return out


# ─────────────── fast greedy allocator  ───────────────────────────── #
@lru_cache(maxsize=None)
def alloc_greedy(target, asks, sizes):
    ven = [Venue(i, a, s, 0.002, 0.0002) for i, (a, s) in enumerate(zip(asks, sizes))]
    ix = np.argsort([v.ask + v.fee for v in ven])
    alloc, rem = [0] * len(ven), target
    for i in ix:
        q = min(rem, ven[i].ask_sz)
        alloc[i] = q
        rem -= q
        if not rem:
            break
    return tuple(alloc)


# exact allocator for final evaluation (100‑share granularity)
def alloc_exact(target, venues, waits, lam_o, lam_u, theta):
    CHUNK = 100
    ranges = [range(0, min(v.ask_sz, target) + CHUNK, CHUNK) for v in venues]
    best, best_c = None, float("inf")
    for alloc in itertools.product(*ranges):
        exe = sum(alloc)
        cash = sum(q * (v.ask + v.fee) for q, v in zip(alloc, venues))
        wait_pen = theta * sum(w * (q / ORDER_SIZE) for q, w in zip(alloc, waits))
        c = (
            cash
            + wait_pen
            + lam_u * max(target - exe, 0)
            + lam_o * max(exe - target, 0)
        )
        if c < best_c:
            best_c, best = c, alloc
    return best


# ─────────────── back‑test core  ──────────────────────────────────── #
def run(df, lo, lu, th, model):
    lam_o, lam_u, theta = lo / 1e4, lu / 1e4, th / 1e4 / QUEUE_SCALE
    rem, cash, path = ORDER_SIZE, 0.0, []
    hist, fr, tr = defaultdict(lambda: deque(maxlen=ROLL)), FillRate(), TradeRate()
    last_sz, last_ts = {}, None
    for ts, row in df.iterrows():  # ←  was:  for ts, row in df.itertuples():
        if rem == 0:
            break
        venues = (
            row_to_venues(row, model, hist, fr)
            if model != "STATIC"
            else row_to_venues(row, "STATIC")
        )
        waits = [v.ask_sz / tr.get(v.id) for v in venues]
        # greedy during run – enough accuracy
        split = alloc_greedy(
            rem, tuple(v.ask for v in venues), tuple(v.ask_sz for v in venues)
        )
        exe = spend = 0
        for q, v in zip(split, venues):
            f = min(q, v.ask_sz, rem - exe)
            exe += f
            spend += f * (v.ask + v.fee)
            fr.update(v.id, f, v.ask_sz)
        rem -= exe
        cash += spend
        path.append(cash)

        if last_ts is not None:
            dt = (ts - last_ts).total_seconds()
            for v in venues:
                shrink = max(last_sz.get(v.id, v.ask_sz) - v.ask_sz, 0)
                tr.update(v.id, shrink, dt)
        last_sz, last_ts = {v.id: v.ask_sz for v in venues}, ts

    if rem:
        cash += rem * (row.filter(like="ask_px_00").min() + 0.002)
        path.append(cash)
    return cash / ORDER_SIZE, cash, np.asarray(path)


# ─────────────── baselines  ───────────────────────────────────────── #
def baselines(df):
    asks = df.filter(like="ask_px_00").to_numpy(float, copy=False)
    sizes = df.filter(like="ask_sz_00").to_numpy(float, copy=False)
    fee = 0.002
    ba_avg = asks[0].min() + fee
    ba_tot = ba_avg * ORDER_SIZE
    row_min = asks.min(axis=1)
    bucket = 60
    full = (len(row_min) // bucket) * bucket
    twap = (
        row_min.mean()
        if full == 0
        else row_min[:full].reshape(-1, bucket).mean(axis=1).mean()
    )
    twap_tot = twap * ORDER_SIZE
    m = (sizes > 0) & (asks > 0)
    vwap = (asks[m] * sizes[m]).sum() / sizes[m].sum()
    return dict(
        best_ask=(ba_tot, ba_avg),
        twap=(twap * ORDER_SIZE, twap),
        vwap=(vwap * ORDER_SIZE, vwap),
    )


# ─────────────── grid search  ─────────────────────────────────────── #
def _worker(args):
    df, model, pars = args
    lo, lu, th = pars
    return run(df, lo, lu, th, model)[1], pars  # return total cash


def tune(df, model):
    coarse = list(
        itertools.product(GRID_COARSE["lo"], GRID_COARSE["lu"], GRID_COARSE["th"])
    )
    with mp.Pool(N_PROC) as p:
        best = min(
            p.imap_unordered(_worker, ((df, model, g) for g in coarse), 32),
            key=lambda x: x[0],
        )[
            1
        ]  # lo,lu,th
    # fine grid around winner
    lo_c, lu_c, th_c = best
    fine_lo = (lo_c + GRID_FINE["lo"]).clip(0)
    fine_lu = (lu_c + GRID_FINE["lu"]).clip(0)
    fine_th = (th_c + GRID_FINE["th"]).clip(0)
    fine = list(itertools.product(fine_lo, fine_lu, fine_th))
    with mp.Pool(N_PROC) as p:
        best_f = min(
            p.imap_unordered(_worker, ((df, model, g) for g in fine), 16),
            key=lambda x: x[0],
        )[1]
    return best_f  # lo,lu,th


# ─────────────── main  ────────────────────────────────────────────── #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        default=CSV_PATH,
        help="Path to level-1 CSV file (default set to provided data path)",
    )
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    global PLOT_DEFAULT
    PLOT_DEFAULT = args.plot

    raw = pd.read_csv(args.csv)
    raw["ts_event"] = pd.to_datetime(raw["ts_event"], utc=True)
    df = (
        raw.groupby(["ts_event", "publisher_id"])
        .first()
        .reset_index()
        .sort_values("ts_event")
        .set_index("ts_event")
    )

    base = baselines(df)
    models = ["STATIC", "UNCERTAINTY-SIGMOID", "UNCERTAINTY-POWER"]
    curves, results = {}, {}

    def run_model(m):
        lo, lu, th = tune(df, m)
        avg, tot, curve = run(df, lo, lu, th, m)
        # exact alloc once for final numbers
        venues = row_to_venues(
            df.iloc[0], m, defaultdict(lambda: deque(maxlen=ROLL)), FillRate()
        )
        waits = [v.ask_sz / 1.0 for v in venues]  # rough; small effect
        split = alloc_exact(
            ORDER_SIZE, venues, waits, lo / 1e4, lu / 1e4, th / 1e4 / QUEUE_SCALE
        )
        return m, curve, lo, lu, th, avg, tot

    with ThreadPoolExecutor(max_workers=3) as ex:
        for m, curve, lo, lu, th, avg, tot in ex.map(run_model, models):
            curves[m] = curve
            results[m] = dict(
                params=dict(lambda_over=lo, lambda_under0=lu, theta_wait=th),
                total_cash=tot,
                avg_price=avg,
                savings_bps=dict(
                    best_ask=1e4 * (base["best_ask"][1] - avg) / base["best_ask"][1],
                    twap=1e4 * (base["twap"][1] - avg) / base["twap"][1],
                    vwap=1e4 * (base["vwap"][1] - avg) / base["vwap"][1],
                ),
            )
    # ensure output folder exists and save plot there

    os.makedirs(out_dir, exist_ok=True)

    for m, c in curves.items():
        plt.plot(c, label=m, lw=1)
    plt.legend()
    plt.title("Cumulative cash")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "results_ALL.png"), dpi=150
    )  # ← replaces old savefig

    print(
        json.dumps(
            dict(
                baselines=dict(
                    best_ask=dict(
                        total_cash=base["best_ask"][0], avg_price=base["best_ask"][1]
                    ),
                    twap=dict(total_cash=base["twap"][0], avg_price=base["twap"][1]),
                    vwap=dict(total_cash=base["vwap"][0], avg_price=base["vwap"][1]),
                ),
                models=results,
            ),
            indent=2,
        )
    )


# -------------------------------------------------------------------- #
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    t = time.time()
    main()
    print(f"# runtime {time.time()-t:0.1f}s", file=sys.stderr)
