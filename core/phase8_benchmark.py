"""
Phase 8 — Metrics & Baselines Benchmark Suite
================================================
Runs rigorous evaluation across >= 10 random seeds per configuration.
Produces a final results table (Markdown + CSV) comparing:
  1. all-shortest-path (Dijkstra)
  2. k-shortest+greedy (pick least congested candidate)
  3. QPSO-hybrid (our approach: QPSO + candidate-swap local search)

Metrics collected per run:
  - Total travel time (BPR-congested)
  - Total distance
  - Congestion cost (Σ load_e^2)
  - Constraint violations
  - Runtime (s)
"""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Adjust path to import core modules
sys.path.insert(0, str(Path(__file__).parent))

from phase1_world_state import (
    apply_loads_to_graph,
    bpr_travel_time,
    download_graph,
    enrich_graph,
    generate_synthetic_congestion,
)
from phase2_candidate_generation import build_candidate_pool
from phase3_qubo import MU_SWITCH_PENALTY_S, Vehicle, build_qubo, compute_lambda
from phase4_qpso import run_qpso
from phase5_local_search import run_local_search_and_guard

log = logging.getLogger("phase8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SEEDS = 10
N_VEHICLES = 200
K_CANDIDATES = 5
SCENARIO = "rush_hour"

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_metrics(
    assignment: List[int],
    vehicles: List[Vehicle],
    candidates: Dict[int, Any],
    G: Any,
    congestion_state: Dict,
    algorithm: str,
    runtime_s: float,
    seed: int,
) -> Dict[str, Any]:
    """Computes all required metrics for a single algorithm run."""
    total_time = 0.0
    total_dist = 0.0
    n_violations = 0
    edge_counts: Dict[Any, int] = {}

    for v_idx, vehicle in enumerate(vehicles):
        k = assignment[v_idx] if v_idx < len(assignment) else 0
        cands = candidates.get(v_idx, [])
        
        # If no candidates or invalid choice, mark violation
        if not cands or k >= len(cands):
            n_violations += 1
            continue

        path, _ = cands[k]
        
        for u, w in zip(path[:-1], path[1:]):
            if not G.has_edge(u, w):
                continue
            edge_data = G[u][w]
            best_ek = min(
                edge_data,
                key=lambda ek: edge_data[ek].get("free_flow_time", float("inf"))
            )
            d = edge_data[best_ek]
            load = max(
                (congestion_state.get((u, w, ek), 0.0) for ek in range(5)),
                default=0.0
            )
            
            total_time += bpr_travel_time(
                d["free_flow_time"], load,
                d["capacity"], d["bpr_alpha"], d["bpr_beta"]
            )
            total_dist += d.get("length", 0.0)
            edge_counts[(u, w)] = edge_counts.get((u, w), 0) + 1

    congestion_cost = float(sum(c ** 2 for c in edge_counts.values()))

    return {
        "algorithm": algorithm,
        "seed": seed,
        "total_travel_time_s": total_time,
        "total_distance_m": total_dist,
        "congestion_cost": congestion_cost,
        "n_violations": n_violations,
        "runtime_s": runtime_s,
    }


def format_stats(values: List[float], decimals: int = 2) -> str:
    """Returns 'mean ± std [min, max]'."""
    mean_val = np.mean(values)
    std_val = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    fmt = f"{{:.{decimals}f}}"
    s_mean = fmt.format(mean_val)
    s_std = fmt.format(std_val)
    s_min = fmt.format(min_val)
    s_max = fmt.format(max_val)
    
    return f"{s_mean} ± {s_std} [{s_min}, {s_max}]"


# ---------------------------------------------------------------------------
# Main Benchmark Loop
# ---------------------------------------------------------------------------
def run_benchmark_suite():
    print(f"\n{'=' * 80}")
    print(f"PHASE 8 - METRICS & BASELINES BENCHMARK SUITE")
    print(f"{'=' * 80}")
    print(f"  Configuration:")
    print(f"    Seeds         : {N_SEEDS}")
    print(f"    Vehicles      : {N_VEHICLES}")
    print(f"    Candidates (K): {K_CANDIDATES}")
    print(f"    Scenario      : {SCENARIO}")
    print(f"    Location      : Navrangpura, Ahmedabad")
    print(f"{'-' * 80}\n")

    # 1. Load/Prepare Graph (cache it so we don't re-download)
    graph_path = DATA_DIR / "phase1_graph.pkl"
    if graph_path.exists():
        log.info("Loading cached graph...")
        import pickle
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
    else:
        log.info("Downloading graph for Navrangpura...")
        G = download_graph(center=(23.0285, 72.5546), radius_m=1000.0)
        G = enrich_graph(G)
        import pickle
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
            
    nodes = list(G.nodes())
    results_all = []

    # 2. Run over N_SEEDS
    for seed in range(N_SEEDS):
        log.info(f"--- Running Seed {seed+1}/{N_SEEDS} (seed={seed}) ---")
        
        # O-D Sampling
        rng = random.Random(seed)
        import networkx as nx
        
        od_pairs = []
        vehicles = []
        attempts = 0
        while len(od_pairs) < N_VEHICLES and attempts < N_VEHICLES * 20:
            o, d = rng.sample(nodes, 2)
            if nx.has_path(G, o, d):
                od_pairs.append((int(o), int(d)))
                vehicles.append(Vehicle(vid=len(od_pairs)-1, origin=int(o), destination=int(d)))
            attempts += 1
            
        # Candidates
        t0 = time.perf_counter()
        raw_pool = build_candidate_pool(G, od_pairs, K=K_CANDIDATES)
        candidates = {v_idx: raw_pool.get((o, d), []) for v_idx, (o, d) in enumerate(od_pairs)}
        
        # Congestion
        cong_state = generate_synthetic_congestion(G, num_vehicles=N_VEHICLES, seed=seed, scenario=SCENARIO)
        apply_loads_to_graph(G, cong_state)
        
        # --- Baseline 1: Dijkstra all-shortest ---
        t0 = time.perf_counter()
        dij_assign = [0] * len(vehicles)
        dij_rt = time.perf_counter() - t0
        res_dij = _compute_metrics(dij_assign, vehicles, candidates, G, cong_state, "dijkstra_all_shortest", dij_rt, seed)
        results_all.append(res_dij)
        
        # --- Baseline 2: k-shortest greedy ---
        t0 = time.perf_counter()
        greedy_assign = []
        for v_idx in range(len(vehicles)):
            cands = candidates.get(v_idx, [])
            if not cands:
                greedy_assign.append(0)
                continue
            best_k, best_c = 0, float("inf")
            for k, (path, ff_cost) in enumerate(cands):
                cong_add = sum(
                    max((cong_state.get((u, w, ek), 0.0) for ek in range(5)), default=0.0)
                    for u, w in zip(path[:-1], path[1:])
                )
                if ff_cost + cong_add < best_c:
                    best_c = ff_cost + cong_add
                    best_k = k
            greedy_assign.append(best_k)
        greedy_rt = time.perf_counter() - t0
        res_greedy = _compute_metrics(greedy_assign, vehicles, candidates, G, cong_state, "k_shortest_greedy", greedy_rt, seed)
        results_all.append(res_greedy)
        
        # --- QPSO-Hybrid ---
        lam = compute_lambda(vehicles, candidates, cong_state)
        Q, offset, meta = build_qubo(vehicles, candidates, cong_state, lam=lam, mu=MU_SWITCH_PENALTY_S)
        V_q, K_q = meta["V"], meta["K_max"]
        
        t0 = time.perf_counter()
        qpso_result = run_qpso(Q, offset, V_q, K_q, seed=seed)
        ls_result = run_local_search_and_guard(qpso_result.best_assignment, None, Q, offset, meta, candidates)
        qpso_rt = time.perf_counter() - t0
        
        res_qpso = _compute_metrics(ls_result.stable_assignment, vehicles, candidates, G, cong_state, "qpso_hybrid", qpso_rt, seed)
        results_all.append(res_qpso)
        
        log.info(f"Seed {seed} complete. QPSO Energy: {qpso_result.best_energy:.2f} -> {ls_result.refined_energy:.2f}")

    # 3. Aggregate Statistics
    stats = defaultdict(lambda: defaultdict(list))
    for r in results_all:
        algo = r["algorithm"]
        stats[algo]["total_travel_time_s"].append(r["total_travel_time_s"])
        stats[algo]["total_distance_m"].append(r["total_distance_m"])
        stats[algo]["congestion_cost"].append(r["congestion_cost"])
        stats[algo]["n_violations"].append(r["n_violations"])
        stats[algo]["runtime_s"].append(r["runtime_s"])

    # 4. Print Markdown Table
    print(f"\n\n{'=' * 120}")
    print(f"BENCHMARK RESULTS (Aggregated over {N_SEEDS} seeds)")
    print(f"{'=' * 120}")
    
    headers = ["Algorithm", "Total Travel Time (s)", "Congestion Cost (Sum load^2)", "Runtime (s)", "Violations"]
    print(f"| {headers[0]:<25} | {headers[1]:<35} | {headers[2]:<35} | {headers[3]:<30} | {headers[4]:<10} |")
    print(f"|{'-'*27}|{'-'*37}|{'-'*37}|{'-'*32}|{'-'*12}|")
    
    algos = ["dijkstra_all_shortest", "k_shortest_greedy", "qpso_hybrid"]
    for algo in algos:
        tt = format_stats(stats[algo]["total_travel_time_s"], 1)
        cc = format_stats(stats[algo]["congestion_cost"], 1)
        rt = format_stats(stats[algo]["runtime_s"], 3)
        vi = f"{np.mean(stats[algo]['n_violations']):.0f}"
        
        print(f"| {algo:<25} | {tt:<35} | {cc:<35} | {rt:<30} | {vi:<10} |")
        
    print(f"{'=' * 120}\n")

    # 5. Save to CSV
    csv_path = DATA_DIR / "phase8_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_all[0].keys())
        writer.writeheader()
        writer.writerows(results_all)
    log.info(f"Raw results saved to {csv_path}")

    # Save summary to JSON
    summary_path = DATA_DIR / "phase8_summary.json"
    summary_data = {}
    for algo in algos:
        summary_data[algo] = {
            metric: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals))
            } for metric, vals in stats[algo].items()
        }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    log.info(f"Aggregated summary saved to {summary_path}")


if __name__ == "__main__":
    run_benchmark_suite()
