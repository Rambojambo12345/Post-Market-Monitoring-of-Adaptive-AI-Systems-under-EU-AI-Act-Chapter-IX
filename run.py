"""Run everything: python run.py [maze|tictactoe] [--seeds N] [--out DIR]. Writes results/."""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

from experiment import SETTINGS, run_seed, score

TARGET_NAMES = {
    "maze": {0: "C0 (no drift)", 1: "C1 (opp. corner)", 2: "C2 (near hazard)", 3: "C3 (adjacent)",
             4: "C4 (start)"},
    "tictactoe": {0: "R0 (no drift)", 1: "R1 (Win-Now)", 2: "R2 (Block)", 3: "R3 (Fork)",
                  4: "R4 (Anti-Fork)", 5: "R5 (Positional)"},
}
SIGNALS = [  # (signal, methods watching it) for the hardest-cases table
    ("Loss", ["CUSUM (loss)", "Page-Hinkley (loss)", "Geometric MA (loss)", "BOCD (loss)",
              "ADWIN (loss)", "KSWIN (loss)"]),
    ("Failure", ["DDM (failure)", "EDDM (failure)", "HDDM-A (failure)", "HDDM-W (failure)",
                 "RDDM (failure)", "ECDD-WT (failure)", "STEPD (failure)"]),
    ("Return distribution", ["KS (return)"]),
    ("States", ["KS (states)", "MMD (states)", "DAWIDD (states)"]),
    ("Actions", ["KS (actions)"]),
    ("Trajectory (states + actions)", ["Edit-operation"]),
]


def run_domain(domain, seeds, workers, out):
    results = {}
    for target in SETTINGS[domain]["targets"]:
        print(f"{domain} {TARGET_NAMES[domain][target]} ...", flush=True)
        with Pool(workers) as pool:
            runs = pool.starmap(run_seed, [(domain, target, s) for s in seeds])
        results[target] = runs

    with open(out / f"{domain}.json", "w") as f:
        json.dump(results, f)
    return results


def summarise(domain, runs):
    change = SETTINGS[domain]["change"]
    methods = runs[0]["alarms"]
    return {m: score([r["alarms"][m] for r in runs], change) for m in methods}


def false_alarms(domain, control):
    """Alarms per 100 episodes on the no-drift control, over the whole monitored period."""
    s = SETTINGS[domain]
    episodes = len(control) * (s["total"] - s["baseline"])
    return {m: 100 * sum(len(r["alarms"][m]) for r in control) / episodes
            for m in control[0]["alarms"]}


def fmt(x, digits=2):
    return "-" if x is None else f"{x:.{digits}f}"


def mean(runs, key):
    return sum(r[key] for r in runs) / len(runs)


def results_table(domain, results):
    targets = [t for t in results if t != 0]
    summaries = {t: summarise(domain, results[t]) for t in targets}
    fp = false_alarms(domain, results[0])
    lines = [
        "| Method | " + " | ".join(f"{TARGET_NAMES[domain][t]} DR | Del." for t in targets)
        + " | FP/100 (no drift) |",
        "|---" + "|---:|---:" * len(targets) + "|---:|",
    ]
    for method in summaries[targets[0]]:
        cells = []
        for t in targets:
            cells += [fmt(summaries[t][method]["dr"]), fmt(summaries[t][method]["delay"], 1)]
        lines.append(f"| {method} | " + " | ".join(cells) + f" | {fp[method]:.2f} |")

    lines += ["", "| Scenario | Return before | Return after | Failure rate before | Failure rate after |",
              "|---|---:|---:|---:|---:|"]
    for t in results:
        runs = results[t]
        lines.append(f"| {TARGET_NAMES[domain][t]} | {mean(runs, 'return_before'):.3f} | "
                     f"{mean(runs, 'return_after'):.3f} | {mean(runs, 'failed_before'):.3f} | "
                     f"{mean(runs, 'failed_after'):.3f} |")
    return lines


def hardest_cases_table(maze, tictactoe):
    c1, c3 = summarise("maze", maze[1]), summarise("maze", maze[3])
    r1 = summarise("tictactoe", tictactoe[1])
    fp_maze, fp_ttt = false_alarms("maze", maze[0]), false_alarms("tictactoe", tictactoe[0])
    lines = ["| Signal | Method | Maze C1 | Maze C3 | TTT R1 | FP/100 maze | FP/100 TTT |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for signal, methods in SIGNALS:
        for m in methods:
            lines.append(f"| {signal} | {m} | {c1[m]['dr']:.2f} | {c3[m]['dr']:.2f} | "
                         f"{r1[m]['dr']:.2f} | {fp_maze[m]:.2f} | {fp_ttt[m]:.2f} |")
    return lines


def write_tables(out):
    loaded = {}
    for domain in SETTINGS:
        path = out / f"{domain}.json"
        if path.exists():
            loaded[domain] = {int(t): runs for t, runs in json.loads(path.read_text()).items()}

    lines = ["DR = detection rate within the grace window, Del. = mean delay in episodes, "
             "FP/100 = alarms per 100 episodes on the no-drift control (C0 / R0).", ""]
    if "maze" in loaded:
        lines += ["## Maze (environmental drift)", ""] + results_table("maze", loaded["maze"]) + [""]
    if "tictactoe" in loaded:
        lines += ["## Tic-tac-toe (adversarial drift)", ""] + results_table("tictactoe", loaded["tictactoe"]) + [""]
    if len(loaded) == 2:
        lines += ["## Hardest cases", ""] + hardest_cases_table(loaded["maze"], loaded["tictactoe"]) + [""]
    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", nargs="?", choices=list(SETTINGS) + ["all"], default="all")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.out.mkdir(exist_ok=True)
    seeds = list(range(args.seeds))
    for domain in (SETTINGS if args.domain == "all" else [args.domain]):
        run_domain(domain, seeds, args.workers, args.out)
    write_tables(args.out)
    print(f"tables written to {args.out / 'tables.md'}")


if __name__ == "__main__":
    main()
