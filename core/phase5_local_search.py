"""
Phase 5 — Local Search + Stability Guard
=========================================
Two post-QPSO refinement layers, applied in sequence:

  1. Candidate-swap local search (NOT 2-Opt, NOT path-level operator)
     After QPSO converges, for each vehicle try swapping its chosen route
     with each of its other K-1 candidates. Accept if the swap lowers the
     global QUBO objective. Repeat until no improving swap is found OR
     a max-pass limit is reached.

  2. Hysteresis stability guard (8% margin, AQH-Route mitigation)
     A vehicle's route is ONLY changed on re-optimisation if the new
     route's individual travel cost is MORE THAN 8% cheaper than the
     current route's cost. This is a post-filter that prevents flip-
     flopping between routes whose cost difference is within noise.

Key design decisions
---------------------
- Local search is over the CANDIDATE MENU (K routes per vehicle), not
  over individual road edges. 2-Opt is a TSP-tour operator that is
  invalid on simple paths — we explicitly do NOT use it.
- The stability guard is NOT part of the QUBO objective. It is applied
  AFTER Phase 4+5 as a post-filter on the final assignment delta.
- Both components are fully deterministic (no RNG).
- Max passes for local search = 10 (prevents infinite loops on flat
  objective landscapes while still allowing thorough exploration).

Architecture note
-----------------
Phase 5 consumes:
  - Q, offset, meta        — from Phase 3
  - best_assignment         — from Phase 4 QPSO
  - vehicles, candidates   — from Phase 2 / Phase 3

Phase 5 produces:
  - refined_assignment     — after local search
  - stable_assignment      — after hysteresis guard
  - LocalSearchResult      — full diagnostics
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("phase5")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_LOCAL_SEARCH_PASSES: int   = 10    # safety cap on improvement passes
HYSTERESIS_MARGIN:       float = 0.08  # 8% improvement required to accept change


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LocalSearchResult:
    """
    Full diagnostics from Phase 5.

    Attributes
    ----------
    initial_assignment  : assignment from Phase 4 QPSO (input).
    refined_assignment  : assignment after candidate-swap local search.
    stable_assignment   : assignment after hysteresis guard.
    initial_energy      : QUBO energy before local search.
    refined_energy      : QUBO energy after local search.
    n_passes            : number of improvement passes completed.
    n_swaps             : total number of improving swaps accepted.
    swaps_per_pass      : list of swaps accepted in each pass.
    improvements        : list of (vehicle_idx, old_k, new_k, delta_energy).
    n_hysteresis_blocked: number of changes blocked by the 8% guard.
    runtime_s           : wall-clock seconds.
    """
    initial_assignment:   List[int]
    refined_assignment:   List[int]
    stable_assignment:    List[int]
    initial_energy:       float
    refined_energy:       float
    n_passes:             int
    n_swaps:              int
    swaps_per_pass:       List[int]
    improvements:         List[Tuple[int, int, int, float]]
    n_hysteresis_blocked: int
    runtime_s:            float


# ---------------------------------------------------------------------------
# QUBO energy helpers (duplicated here for self-containment — no circular import)
# ---------------------------------------------------------------------------

def _onehot(assignment: List[int], V: int, K: int) -> np.ndarray:
    x = np.zeros(V * K, dtype=np.float64)
    for v, k in enumerate(assignment):
        if k < K:
            x[v * K + k] = 1.0
    return x


def _qubo_energy(
    assignment: List[int],
    Q: np.ndarray,
    offset: float,
    V: int,
    K: int,
) -> float:
    """Evaluate QUBO energy for a decoded assignment (upper-triangular form)."""
    x = _onehot(assignment, V, K)
    e = float(np.dot(x, np.diag(Q)))        # linear diagonal terms
    e += float(x @ np.triu(Q, k=1) @ x)    # quadratic upper-tri terms
    return e + offset


# ---------------------------------------------------------------------------
# 1. Candidate-swap local search
# ---------------------------------------------------------------------------

def candidate_swap_local_search(
    assignment: List[int],
    Q: np.ndarray,
    offset: float,
    V: int,
    K: int,
    max_passes: int = MAX_LOCAL_SEARCH_PASSES,
) -> Tuple[List[int], float, int, int, List[int], List[Tuple[int, int, int, float]]]:
    """
    Greedy candidate-swap local search over the K-route menu.

    Algorithm
    ---------
    For each pass:
      For each vehicle v (in order):
        current_k = assignment[v]
        For each alternative route k ≠ current_k:
          Create trial assignment with vehicle v using route k.
          Compute global QUBO energy of trial assignment.
          If trial energy < current best energy:
            Accept swap (update assignment, update current energy).
            Record improvement.
    Repeat until no improvement in a full pass, or max_passes reached.

    This is NOT 2-Opt. It operates entirely over the candidate menu
    (one discrete choice per vehicle), never over road-level path nodes.

    Parameters
    ----------
    assignment : initial route indices, one per vehicle (from QPSO).
    Q, offset  : QUBO matrix and offset.
    V, K       : number of vehicles and candidates.
    max_passes : maximum number of improvement passes.

    Returns
    -------
    (refined_assignment, refined_energy, n_passes, n_swaps,
     swaps_per_pass, improvements)
    """
    current_assignment = list(assignment)
    current_energy     = _qubo_energy(current_assignment, Q, offset, V, K)

    n_passes     = 0
    n_swaps      = 0
    swaps_per_pass: List[int] = []
    improvements: List[Tuple[int, int, int, float]] = []

    for pass_idx in range(max_passes):
        swaps_this_pass = 0

        for v in range(V):
            current_k = current_assignment[v]

            for k in range(K):
                if k == current_k:
                    continue

                # Trial: vehicle v switches to route k
                trial = list(current_assignment)
                trial[v] = k
                trial_energy = _qubo_energy(trial, Q, offset, V, K)

                if trial_energy < current_energy - 1e-9:
                    # Improving swap found — accept immediately (first-improvement)
                    delta = trial_energy - current_energy
                    improvements.append((v, current_k, k, delta))
                    log.debug(
                        "Pass %d: vehicle %d route %d→%d  ΔE=%.6f",
                        pass_idx, v, current_k, k, delta,
                    )
                    current_assignment = trial
                    current_energy     = trial_energy
                    current_k          = k
                    swaps_this_pass   += 1
                    n_swaps           += 1

        swaps_per_pass.append(swaps_this_pass)
        n_passes += 1

        if swaps_this_pass == 0:
            log.debug("Local search converged after %d passes.", pass_idx + 1)
            break

    return (
        current_assignment,
        current_energy,
        n_passes,
        n_swaps,
        swaps_per_pass,
        improvements,
    )


# ---------------------------------------------------------------------------
# 2. Hysteresis stability guard
# ---------------------------------------------------------------------------

def hysteresis_stability_guard(
    new_assignment: List[int],
    prev_assignment: Optional[List[int]],
    candidates: Dict[int, List[Tuple[List[int], float]]],
    margin: float = HYSTERESIS_MARGIN,
) -> Tuple[List[int], int]:
    """
    Post-filter: only accept a vehicle's new route if its individual cost
    is more than `margin` (8%) cheaper than its current route's cost.

    This prevents oscillation where a vehicle flip-flops between two routes
    with nearly identical costs, wasting re-optimisation cycles.

    Parameters
    ----------
    new_assignment  : proposed assignment from Phase 4+5 local search.
    prev_assignment : previous round's committed assignment. If None
                      (first round), all changes are accepted unconditionally.
    candidates      : {v_idx: [(path, cost), ...]}
    margin          : fractional improvement required (default 0.08 = 8%).

    Returns
    -------
    (stable_assignment, n_blocked)
    stable_assignment : new assignment with hysteresis-blocked vehicles
                        reverted to their previous route.
    n_blocked         : number of vehicles whose change was blocked.

    Note
    ----
    This uses per-vehicle individual travel cost (the candidate cost from
    Yen's algorithm), NOT the global QUBO energy. It is a per-vehicle
    anti-oscillation filter, applied AFTER the global optimisation.
    """
    if prev_assignment is None:
        # First round: no previous state — accept everything.
        return list(new_assignment), 0

    stable    = list(new_assignment)
    n_blocked = 0

    for v_idx, (new_k, prev_k) in enumerate(zip(new_assignment, prev_assignment)):
        if new_k == prev_k:
            continue  # No change — nothing to guard

        v_candidates = candidates.get(v_idx, [])
        if new_k >= len(v_candidates) or prev_k >= len(v_candidates):
            continue  # Out-of-range — leave as-is

        new_cost  = v_candidates[new_k][1]
        prev_cost = v_candidates[prev_k][1]

        if prev_cost <= 0:
            continue  # Avoid division by zero

        # Fractional improvement required: new_cost < prev_cost * (1 - margin)
        improvement_fraction = (prev_cost - new_cost) / prev_cost

        if improvement_fraction <= margin:
            # Improvement too small — revert to previous route
            stable[v_idx] = prev_k
            n_blocked     += 1
            log.debug(
                "Hysteresis guard: vehicle %d blocked (new_cost=%.2f, "
                "prev_cost=%.2f, improvement=%.1f%% < %.1f%%)",
                v_idx, new_cost, prev_cost,
                improvement_fraction * 100, margin * 100,
            )

    return stable, n_blocked


# ---------------------------------------------------------------------------
# Phase 5 main pipeline
# ---------------------------------------------------------------------------

def run_local_search_and_guard(
    initial_assignment: List[int],
    prev_assignment: Optional[List[int]],
    Q: np.ndarray,
    offset: float,
    meta: Dict,
    candidates: Dict[int, List[Tuple[List[int], float]]],
    max_passes: int = MAX_LOCAL_SEARCH_PASSES,
    hysteresis_margin: float = HYSTERESIS_MARGIN,
) -> LocalSearchResult:
    """
    Full Phase 5 pipeline:
      1. Candidate-swap local search.
      2. Hysteresis stability guard.

    Parameters
    ----------
    initial_assignment : route indices from Phase 4 QPSO.
    prev_assignment    : committed assignment from the previous optimisation
                         round (for hysteresis). None on first round.
    Q, offset, meta    : QUBO from Phase 3.
    candidates         : Phase 2 candidate pool (for hysteresis cost lookup).
    max_passes         : local search pass cap (default 10).
    hysteresis_margin  : fractional improvement required (default 8%).

    Returns
    -------
    LocalSearchResult
    """
    t0 = time.perf_counter()
    V  = meta["V"]
    K  = meta["K_max"]

    # Step 1: Initial energy
    initial_energy = _qubo_energy(initial_assignment, Q, offset, V, K)

    # Step 2: Candidate-swap local search
    (
        refined_assignment,
        refined_energy,
        n_passes,
        n_swaps,
        swaps_per_pass,
        improvements,
    ) = candidate_swap_local_search(
        initial_assignment, Q, offset, V, K, max_passes=max_passes
    )

    # Step 3: Hysteresis stability guard
    stable_assignment, n_blocked = hysteresis_stability_guard(
        refined_assignment, prev_assignment, candidates,
        margin=hysteresis_margin,
    )

    runtime = time.perf_counter() - t0

    return LocalSearchResult(
        initial_assignment   = list(initial_assignment),
        refined_assignment   = refined_assignment,
        stable_assignment    = stable_assignment,
        initial_energy       = initial_energy,
        refined_energy       = refined_energy,
        n_passes             = n_passes,
        n_swaps              = n_swaps,
        swaps_per_pass       = swaps_per_pass,
        improvements         = improvements,
        n_hysteresis_blocked = n_blocked,
        runtime_s            = runtime,
    )


# ---------------------------------------------------------------------------
# Acceptance test
# ---------------------------------------------------------------------------

def run_phase5(
    Q: np.ndarray,
    offset: float,
    meta: Dict,
    vehicles,
    candidates: Dict[int, List[Tuple[List[int], float]]],
    initial_assignment: List[int],
    prev_assignment: Optional[List[int]] = None,
) -> LocalSearchResult:
    """
    Phase 5 acceptance test.

    PASS conditions:
      1. refined_energy <= initial_energy  (local search never worsens)
      2. stable_assignment is a valid assignment (one route per vehicle)
      3. Hysteresis guard correctly blocks small improvements

    Runs two scenarios:
      A. No previous assignment (first round) — guard accepts all.
      B. With previous assignment — guard blocks changes < 8%.
    """
    V = meta["V"]
    K = meta["K_max"]

    print(f"\n{'=' * 60}")
    print("PHASE 5 - LOCAL SEARCH + STABILITY GUARD")
    print(f"{'=' * 60}")
    print(f"  Vehicles         : {V}")
    print(f"  K routes/vehicle : {K}")
    print(f"  Initial energy   : {_qubo_energy(initial_assignment, Q, offset, V, K):.4f}")
    print(f"  Max passes       : {MAX_LOCAL_SEARCH_PASSES}")
    print(f"  Hysteresis margin: {HYSTERESIS_MARGIN*100:.0f}%")

    # -----------------------------------------------------------------------
    # Scenario A: First round (no previous assignment)
    # -----------------------------------------------------------------------
    print(f"\n  Scenario A: First optimisation round (no prev assignment)")
    result_a = run_local_search_and_guard(
        initial_assignment=initial_assignment,
        prev_assignment=None,
        Q=Q, offset=offset, meta=meta, candidates=candidates,
    )
    print(f"    Initial energy   : {result_a.initial_energy:.4f}")
    print(f"    Refined energy   : {result_a.refined_energy:.4f}")
    print(f"    Energy delta     : {result_a.refined_energy - result_a.initial_energy:+.6f}")
    print(f"    Passes run       : {result_a.n_passes}")
    print(f"    Improving swaps  : {result_a.n_swaps}")
    print(f"    Swaps per pass   : {result_a.swaps_per_pass}")
    print(f"    Hysteresis blocked: {result_a.n_hysteresis_blocked} (should be 0, no prev)")

    ok_a1 = result_a.refined_energy <= result_a.initial_energy + 1e-9
    ok_a2 = result_a.n_hysteresis_blocked == 0
    ok_a3 = len(result_a.stable_assignment) == V
    print(f"    [{'PASS' if ok_a1 else 'FAIL'}] refined_energy <= initial_energy")
    print(f"    [{'PASS' if ok_a2 else 'FAIL'}] No hysteresis blocks on first round")
    print(f"    [{'PASS' if ok_a3 else 'FAIL'}] Valid assignment length")

    if result_a.improvements:
        print(f"\n    Improvements found:")
        for v_idx, old_k, new_k, delta in result_a.improvements:
            cands = candidates.get(v_idx, [])
            old_cost = cands[old_k][1] if old_k < len(cands) else float('nan')
            new_cost = cands[new_k][1] if new_k < len(cands) else float('nan')
            print(f"      Vehicle {v_idx}: route {old_k}({old_cost:.1f}s) "
                  f"-> route {new_k}({new_cost:.1f}s)  ΔE={delta:+.4f}")
    else:
        print(f"\n    No improving swaps found (QPSO already at local optimum).")

    # -----------------------------------------------------------------------
    # Scenario B: Re-optimisation round with previous assignment
    #             Artificially create a scenario where hysteresis should block
    # -----------------------------------------------------------------------
    print(f"\n  Scenario B: Re-optimisation with previous assignment")

    # Create a "previous assignment" that is the refined assignment from A
    # Then create a "new proposal" that differs slightly (within 8% margin)
    # by using a slightly different assignment
    prev_assign_b = list(result_a.refined_assignment)

    # Create a modified assignment where some vehicles are proposed to change
    # to a route that is only marginally better (< 8%) or worse
    proposed_b = list(prev_assign_b)

    # Find a vehicle where route 0 and route 1 costs differ by < 8%
    hysteresis_test_vehicle = None
    hysteresis_expected_block = False
    for v_idx in range(V):
        cands = candidates.get(v_idx, [])
        if len(cands) >= 2:
            cost_0 = cands[prev_assign_b[v_idx]][1]
            for alt_k in range(len(cands)):
                if alt_k == prev_assign_b[v_idx]:
                    continue
                cost_alt = cands[alt_k][1]
                if cost_alt > 0 and cost_0 > 0:
                    improvement = (cost_0 - cost_alt) / cost_0
                    if -0.05 < improvement < HYSTERESIS_MARGIN:
                        # Small or negative improvement — guard should block
                        proposed_b[v_idx] = alt_k
                        hysteresis_test_vehicle = v_idx
                        hysteresis_expected_block = True
                        print(f"    Forcing vehicle {v_idx}: {prev_assign_b[v_idx]} "
                              f"({cost_0:.1f}s) -> {alt_k} ({cost_alt:.1f}s), "
                              f"improvement={improvement*100:.1f}% (expect block)")
                        break
        if hysteresis_test_vehicle is not None:
            break

    if hysteresis_test_vehicle is None:
        print(f"    (All routes differ by > 8% — all changes will be accepted)")

    _, n_blocked_b = hysteresis_stability_guard(
        proposed_b, prev_assign_b, candidates, margin=HYSTERESIS_MARGIN
    )

    ok_b1 = (n_blocked_b >= 1) if hysteresis_expected_block else True
    print(f"    Changes proposed : {sum(1 for a, b in zip(proposed_b, prev_assign_b) if a != b)}")
    print(f"    Hysteresis blocked: {n_blocked_b}")
    print(f"    [{'PASS' if ok_b1 else 'FAIL'}] Hysteresis guard blocks sub-threshold changes")

    # -----------------------------------------------------------------------
    # Scenario C: Verify guard accepts large improvements (> 8%)
    # -----------------------------------------------------------------------
    print(f"\n  Scenario C: Large improvement passes hysteresis guard")
    proposed_c = list(prev_assign_b)

    large_improvement_vehicle = None
    for v_idx in range(V):
        cands = candidates.get(v_idx, [])
        if len(cands) >= 2:
            cost_curr = cands[prev_assign_b[v_idx]][1]
            # Simulate a big improvement by proposing a very cheap route
            # We artificially test: route 0 vs a very expensive one
            best_k = min(range(len(cands)), key=lambda k: cands[k][1])
            worst_k = max(range(len(cands)), key=lambda k: cands[k][1])
            if best_k != prev_assign_b[v_idx]:
                cost_best = cands[best_k][1]
                if cost_curr > 0:
                    improvement = (cost_curr - cost_best) / cost_curr
                    if improvement > HYSTERESIS_MARGIN:
                        proposed_c[v_idx] = best_k
                        large_improvement_vehicle = v_idx
                        print(f"    Vehicle {v_idx}: {prev_assign_b[v_idx]} "
                              f"({cost_curr:.1f}s) -> {best_k} ({cost_best:.1f}s), "
                              f"improvement={improvement*100:.1f}% (expect accept)")
                        break

    stable_c, n_blocked_c = hysteresis_stability_guard(
        proposed_c, prev_assign_b, candidates, margin=HYSTERESIS_MARGIN
    )

    if large_improvement_vehicle is not None:
        accepted = stable_c[large_improvement_vehicle] == proposed_c[large_improvement_vehicle]
        ok_c1 = accepted
        print(f"    Hysteresis blocked: {n_blocked_c}")
        print(f"    Large improvement accepted: {'YES' if accepted else 'NO'}")
        print(f"    [{'PASS' if ok_c1 else 'FAIL'}] Guard accepts >8% improvements")
    else:
        ok_c1 = True
        print(f"    (No large improvement found in this candidate pool — scenario skipped)")

    # -----------------------------------------------------------------------
    # Overall verdict
    # -----------------------------------------------------------------------
    all_pass = ok_a1 and ok_a2 and ok_a3 and ok_b1 and ok_c1
    verdict  = "[PASS]" if all_pass else "[FAIL]"
    print(f"\n  Phase 5 runtime: {result_a.runtime_s*1000:.2f} ms")
    print(f"  {verdict} Phase 5 acceptance test.")
    print(f"{'=' * 60}\n")

    return result_a


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pickle
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from phase3_qubo import Vehicle      # noqa: F401
    from phase4_qpso import QPSOResult   # noqa: F401

    DATA_DIR        = Path(__file__).parent.parent / "data"
    qpso_result_path = DATA_DIR / "phase4_qpso_result.pkl"

    if not qpso_result_path.exists():
        raise FileNotFoundError(
            f"Phase 4 result not found at {qpso_result_path}. "
            "Run core/phase4_qpso.py first."
        )

    log.info("Loading Phase 4 result from %s ...", qpso_result_path)
    with open(qpso_result_path, "rb") as f:
        data = pickle.load(f)

    Q                = data["Q"]
    offset           = data["offset"]
    meta             = data["meta"]
    vehicles         = data["vehicles"]
    candidates       = data["candidates"]
    qpso_result      = data["result"]

    log.info(
        "QPSO best assignment: %s  energy=%.4f",
        qpso_result.best_assignment,
        qpso_result.best_energy,
    )

    # Run Phase 5 acceptance test
    ls_result = run_phase5(
        Q=Q,
        offset=offset,
        meta=meta,
        vehicles=vehicles,
        candidates=candidates,
        initial_assignment=qpso_result.best_assignment,
        prev_assignment=None,   # First round — no previous assignment
    )

    # Save Phase 5 result for Phase 6
    result_path = DATA_DIR / "phase5_result.pkl"
    with open(result_path, "wb") as f:
        pickle.dump({
            "ls_result":          ls_result,
            "qpso_result":        qpso_result,
            "vehicles":           vehicles,
            "candidates":         candidates,
            "Q":                  Q,
            "offset":             offset,
            "meta":               meta,
        }, f)
    print(f"Phase 5 result saved --> {result_path}")
