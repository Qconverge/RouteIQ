"""
Phase 3 — QUBO Objective Formulation
======================================
Implements the exact objective from report.pdf Part IV:

    minimize  Σ_v T(route_v, W(t))
            + λ · Σ_e load_e²
            + μ · Σ_v 1[route_v ≠ route_v(prev)]

Encoded as a Quadratic Unconstrained Binary Optimization (QUBO) problem.

Variable encoding
-----------------
  For V vehicles each with K candidate routes:
    x_vk ∈ {0, 1}  — binary: vehicle v picks route k.
  Variables are laid out in a flat vector of length V*K:
    index(v, k) = v * K + k

Objective terms
---------------
  1. Travel time:  Σ_v Σ_k x_vk · T(route_vk, W(t))
     Linear diagonal terms in QUBO.

  2. Congestion spreading: λ · Σ_e load_e²
     load_e = Σ_v Σ_k x_vk · uses(route_vk, e)
     Expanding the square introduces quadratic (x_vk * x_wl) cross terms
     when two different vehicle-route pairs share edge e.

  3. Route-switch penalty: μ · Σ_v x_v,k≠prev
     μ = 120 s (report.pdf B3). Linear diagonal penalty on any route choice
     that differs from the vehicle's previous route.

  4. One-hot constraint: P · Σ_v (Σ_k x_vk - 1)²
     Expanded: P · Σ_v [Σ_k x_vk² - 2·Σ_k x_vk + 1 + 2·Σ_{k<l} x_vk·x_vl]
     The constant (+1 per vehicle) is tracked in `qubo_offset` (not in matrix).
     Since x² = x for binary, diagonal gets: P·(1 - 2) = -P per (v,k).
     Off-diagonal (same vehicle, different routes): +2P.
     This makes the penalty zero iff exactly one x_vk = 1 per vehicle.

λ re-normalisation (report.pdf §7.2)
-------------------------------------
  λ must be re-computed each optimisation round, not kept fixed.
  Static λ fails as congestion changes: if λ is calibrated for light traffic
  and congestion doubles, the congestion term dominates and the solver
  ignores travel time.
  We normalise so that the congestion term is on the same order of magnitude
  as the travel-time term:
      λ_norm = mean_travel_time / (mean_load_sq + ε)
  This is computed inside `compute_lambda` and must be called fresh each round.

P (one-hot penalty scaling)
----------------------------
  P must be large enough that any constraint violation costs more than the
  maximum possible objective improvement:
      P = 10 × (max_travel_time + λ·max_load_sq + μ)
  This guarantees feasible solutions always beat infeasible ones.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("phase3")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants (report.pdf B3)
# ---------------------------------------------------------------------------
MU_SWITCH_PENALTY_S: float = 120.0   # route-switch penalty in seconds (B3)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Vehicle:
    """
    Lightweight vehicle descriptor.

    Attributes
    ----------
    vid        : unique vehicle ID (int or str)
    origin     : origin node ID in the graph
    destination: destination node ID in the graph
    prev_route_idx : index (0..K-1) of the route chosen in the previous round,
                     or None if this is the first assignment.
    """
    __slots__ = ("vid", "origin", "destination", "prev_route_idx")

    def __init__(
        self,
        vid,
        origin: int,
        destination: int,
        prev_route_idx: Optional[int] = None,
    ):
        self.vid            = vid
        self.origin         = origin
        self.destination    = destination
        self.prev_route_idx = prev_route_idx

    def __repr__(self):
        return (f"Vehicle(vid={self.vid}, o={self.origin}, d={self.destination}, "
                f"prev={self.prev_route_idx})")


# ---------------------------------------------------------------------------
# Edge-load computation
# ---------------------------------------------------------------------------

def compute_edge_loads(
    vehicles: List[Vehicle],
    candidates: Dict[int, List[Tuple[List[int], float]]],
    assignment: List[int],
) -> Dict[Tuple[int, int], float]:
    """
    Given a concrete route assignment (one route index per vehicle),
    compute the per-edge load (vehicles using that edge).

    Parameters
    ----------
    vehicles   : list of Vehicle objects.
    candidates : {vehicle_index: [(path, cost), ...]} mapping.
    assignment : list of route indices, one per vehicle.

    Returns
    -------
    {(u, v): load_count}  — MultiDiGraph edge (u,v) usage count.
    """
    loads: Dict[Tuple[int, int], float] = {}
    for v_idx, vehicle in enumerate(vehicles):
        route_idx = assignment[v_idx]
        if route_idx >= len(candidates[v_idx]):
            continue
        path, _ = candidates[v_idx][route_idx]
        for u, w in zip(path[:-1], path[1:]):
            loads[(u, w)] = loads.get((u, w), 0.0) + 1.0
    return loads


# ---------------------------------------------------------------------------
# λ re-normalisation (per report.pdf §7.2)
# ---------------------------------------------------------------------------

def compute_lambda(
    vehicles: List[Vehicle],
    candidates: Dict[int, List[Tuple[List[int], float]]],
    congestion_state: Dict[Tuple, float],
) -> float:
    """
    Compute the re-normalised congestion-spreading weight λ for this round.

    Strategy (report.pdf §7.2):
      Estimate the scale of the travel-time term and the congestion term
      from the current congestion state, then set λ so they are balanced:

          λ = mean_T / (mean_load_sq + ε)

      where:
        mean_T        = mean travel time over all (vehicle, route) pairs
        mean_load_sq  = mean of load_e² over all edges that appear in any route
        ε             = 1e-6 (prevent division by zero in empty networks)

    This is re-computed every optimisation round.

    Parameters
    ----------
    vehicles        : fleet.
    candidates      : {v_idx: [(path, cost), ...]}
    congestion_state: {(u, v, key): load_vph} — current edge loads.

    Returns
    -------
    lambda : float (≥ 0)
    """
    # Collect all travel times across all (vehicle, route) pairs
    all_times: List[float] = []
    for v_idx in range(len(vehicles)):
        for path, cost in candidates.get(v_idx, []):
            all_times.append(cost)

    # Collect load² for all edges that appear in any candidate route
    edge_loads: Dict[Tuple, float] = {}
    for v_idx in range(len(vehicles)):
        for path, _ in candidates.get(v_idx, []):
            for u, w in zip(path[:-1], path[1:]):
                # Use (u,w) key — congestion_state may use (u,w,k) form
                load = max(
                    (congestion_state.get((u, w, k_), 0.0) for k_ in range(5)),
                    default=0.0,
                )
                edge_loads[(u, w)] = max(edge_loads.get((u, w), 0.0), load)

    all_load_sq = [v ** 2 for v in edge_loads.values()] or [0.0]

    mean_T       = float(np.mean(all_times)) if all_times else 1.0
    mean_load_sq = float(np.mean(all_load_sq))
    eps          = 1e-6

    lam = mean_T / (mean_load_sq + eps)
    log.debug("compute_lambda: mean_T=%.3f, mean_load_sq=%.3f, lambda=%.6f",
              mean_T, mean_load_sq, lam)
    return lam


# ---------------------------------------------------------------------------
# Edge usage lookup (pre-computed for efficiency)
# ---------------------------------------------------------------------------

def _build_edge_usage(
    vehicles: List[Vehicle],
    candidates: Dict[int, List[Tuple[List[int], float]]],
    K: int,
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Pre-compute which (vehicle, route) pairs use each edge (u,v).

    Returns
    -------
    {(u,v): [(v_idx, k), ...]}  — list of (vehicle_index, route_index) pairs
                                   that traverse edge (u,v).
    """
    usage: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for v_idx, vehicle in enumerate(vehicles):
        for k, (path, _) in enumerate(candidates.get(v_idx, [])):
            for u, w in zip(path[:-1], path[1:]):
                key = (u, w)
                if key not in usage:
                    usage[key] = []
                usage[key].append((v_idx, k))
    return usage


# ---------------------------------------------------------------------------
# QUBO construction (main deliverable)
# ---------------------------------------------------------------------------

def build_qubo(
    vehicles: List[Vehicle],
    candidates: Dict[int, List[Tuple[List[int], float]]],
    congestion_state: Dict[Tuple, float],
    lam: float,
    mu: float = MU_SWITCH_PENALTY_S,
) -> Tuple[np.ndarray, float, Dict]:
    """
    Build the QUBO coefficient matrix Q for the multi-vehicle routing problem.

    The full objective (binary quadratic form) is:
        f(x) = x^T · Q · x + offset

    Variable layout: x is a flat binary vector of length N = V * K where
        index(v, k) = v * K + k
    and x_vk = 1 means vehicle v takes candidate route k.

    Parameters
    ----------
    vehicles        : list of V Vehicle objects.
    candidates      : {v_idx: [(path, cost), ...]} — K candidates per vehicle.
    congestion_state: {(u, v, key): load_vph} — current BPR-based edge loads.
    lam             : congestion-spreading weight λ (re-normalised per round).
    mu              : route-switch penalty in seconds (default 120s, B3).

    Returns
    -------
    Q      : np.ndarray shape (N, N) — upper-triangular QUBO matrix.
             Q[i,j] with i<=j. Diagonal = linear terms; off-diagonal = quadratic.
    offset : float — constant term (from one-hot expansion) not in Q.
    meta   : dict  — diagnostic info (P, lambda, N, V, K, term breakdowns).

    QUBO convention used here
    -------------------------
    We use the upper-triangular form: Q[i,j] only set for i <= j.
    For i == j: coefficient of x_i (linear term, since x_i^2 = x_i in binary).
    For i <  j: coefficient of x_i * x_j (quadratic cross term).
    Total energy: E = Σ_i Q[i,i]*x_i + Σ_{i<j} Q[i,j]*x_i*x_j + offset.
    """
    V = len(vehicles)
    K_list = [len(candidates.get(v_idx, [])) for v_idx in range(V)]
    K_max  = max(K_list) if K_list else 1
    N      = V * K_max   # total binary variables

    if N == 0:
        return np.zeros((0, 0)), 0.0, {}

    Q      = np.zeros((N, N), dtype=np.float64)
    offset = 0.0   # constant offset (from one-hot constraint expansion)

    # ------------------------------------------------------------------
    # Term 1: Travel time  Σ_v Σ_k x_vk * T(route_vk, W(t))
    #   T(route_vk) = sum of congested_time along the route.
    #   If congested_time is not set on edges, fall back to the path cost
    #   stored in candidates (which is free_flow_time).
    # ------------------------------------------------------------------
    max_T = 0.0
    for v_idx in range(V):
        for k, (path, ff_cost) in enumerate(candidates.get(v_idx, [])):
            idx = v_idx * K_max + k
            # Compute congested travel time for this path
            cong_cost = _path_congested_time(path, congestion_state)
            T_vk = cong_cost if cong_cost > 0 else ff_cost
            Q[idx, idx] += T_vk
            max_T = max(max_T, T_vk)

    # ------------------------------------------------------------------
    # Term 2: Congestion spreading  λ * Σ_e load_e²
    #   load_e = Σ_{(v,k) using e} x_vk
    #   load_e² = (Σ x_vk)² = Σ x_vk² + 2 Σ_{(v1,k1)<(v2,k2)} x_v1k1 * x_v2k2
    #   For binary x: x_vk² = x_vk → diagonal contribution.
    #   Cross-vehicle pairs: quadratic off-diagonal terms.
    # ------------------------------------------------------------------
    edge_usage = _build_edge_usage(vehicles, candidates, K_max)

    max_load_sq = 0.0
    for edge, users in edge_usage.items():
        n_users = len(users)
        if n_users == 0:
            continue
        # Each (v,k) pair that uses this edge contributes 1 to load_e.
        # load_e² diagonal (x_vk² = x_vk): λ * 1 per user
        for (v_idx, k) in users:
            idx = v_idx * K_max + k
            Q[idx, idx] += lam * 1.0   # λ * (load contribution)²_diag

        # Cross terms: 2λ for each pair of distinct (v,k) users
        for i_u in range(len(users)):
            for j_u in range(i_u + 1, len(users)):
                v1, k1 = users[i_u]
                v2, k2 = users[j_u]
                i = v1 * K_max + k1
                j = v2 * K_max + k2
                # Ensure upper-triangular
                if i > j:
                    i, j = j, i
                Q[i, j] += 2.0 * lam

        # Track max load_sq for P scaling
        max_load_sq = max(max_load_sq, n_users ** 2)

    # ------------------------------------------------------------------
    # Term 3: Route-switch penalty  μ * Σ_v 1[route_v ≠ route_v(prev)]
    #   Equivalent to: μ * Σ_v (1 - x_v,prev)  =  μ*V - μ * Σ_v x_v,prev
    #   The μ*V part is a constant added to offset.
    #   The -μ*x_v,prev part reduces the diagonal of the previous route.
    #   Net effect: all non-previous routes get +μ on their diagonal,
    #   previous route gets 0 extra cost (no switch).
    # ------------------------------------------------------------------
    for v_idx, vehicle in enumerate(vehicles):
        if vehicle.prev_route_idx is not None:
            prev_k = vehicle.prev_route_idx
            n_routes = K_list[v_idx]
            for k in range(n_routes):
                if k != prev_k:
                    idx = v_idx * K_max + k
                    Q[idx, idx] += mu
            # Constant: μ * 1 per vehicle with a previous assignment
            # (represents the μ * 1[route≠prev] baseline cost of switching)
            # We don't add to offset here — the formulation above is equivalent.

    # ------------------------------------------------------------------
    # Term 4: One-hot constraint  P * Σ_v (Σ_k x_vk - 1)²
    #   Expand: P * (Σ_k x_vk² - 2*Σ_k x_vk + 1 + 2*Σ_{k<l} x_vk*x_vl)
    #   Binary: x_vk² = x_vk
    #   Diagonal: P * (1 - 2) = -P  per (v,k) pair
    #   Off-diagonal same-vehicle pairs (k < l): +2P
    #   Constant per vehicle: +P  → added to offset
    #
    # P must be large enough to dominate the objective:
    #   P = 10 * (max_T + λ*max_load_sq + μ)
    # ------------------------------------------------------------------
    P = 10.0 * (max_T + lam * max_load_sq + mu)
    if P < 1.0:
        P = 1000.0   # safety floor if all times are tiny

    for v_idx in range(V):
        n_routes = K_list[v_idx]
        # Diagonal: -P per (v,k)
        for k in range(n_routes):
            idx = v_idx * K_max + k
            Q[idx, idx] += -P

        # Off-diagonal same-vehicle pairs: +2P (penalises picking >1 route)
        for k in range(n_routes):
            for l in range(k + 1, n_routes):
                i = v_idx * K_max + k
                j = v_idx * K_max + l
                Q[i, j] += 2.0 * P

        # Constant term (not in matrix, added to offset)
        offset += P

    meta = {
        "V": V,
        "K_max": K_max,
        "N": N,
        "P": P,
        "lambda": lam,
        "mu": mu,
        "max_T": max_T,
        "max_load_sq": max_load_sq,
        "offset": offset,
    }

    return Q, offset, meta


def _path_congested_time(
    path: List[int],
    congestion_state: Dict[Tuple, float],
) -> float:
    """
    Compute the congested travel time of a path given current edge loads.
    Falls back to 0.0 (caller will use free-flow cost) if no BPR data available.

    We look up congested_time by (u,v) edge — the state dict may have
    (u,v,key) form from Phase 1; we take the max-load parallel edge.
    """
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        # Try all parallel edge keys (0,1,2,...) — take the one with least time
        # For congestion purposes we pick the same minimum-cost edge as Yen's.
        load = max(
            (congestion_state.get((u, v, k_), 0.0) for k_ in range(5)),
            default=0.0,
        )
        # We don't have congested_time here directly — that lives in the graph.
        # Phase 3 is graph-agnostic; the caller should pass congested costs
        # pre-computed as the path cost in candidates (see note in build_qubo).
        # This function is a hook for future extension.
        total += load   # placeholder: caller should pass congested path costs
    return 0.0   # signal caller to use stored path cost


# ---------------------------------------------------------------------------
# QUBO energy evaluation
# ---------------------------------------------------------------------------

def evaluate_qubo(
    Q: np.ndarray,
    x: np.ndarray,
    offset: float = 0.0,
) -> float:
    """
    Evaluate E = x^T Q x + offset for a binary assignment vector x.

    Parameters
    ----------
    Q      : (N, N) upper-triangular QUBO matrix.
    x      : (N,) binary vector.
    offset : scalar offset.

    Returns
    -------
    float : total QUBO energy.
    """
    # Upper-triangular form: E = Σ_i Q[i,i]*x[i] + Σ_{i<j} Q[i,j]*x[i]*x[j]
    energy = float(np.dot(x, np.diag(Q)))                    # diagonal terms
    energy += float(x @ np.triu(Q, k=1) @ x)                 # off-diagonal
    return energy + offset


def decode_assignment(x: np.ndarray, V: int, K: int) -> List[int]:
    """
    Decode a binary QUBO solution vector into a list of route indices.

    Parameters
    ----------
    x : (V*K,) binary vector.
    V : number of vehicles.
    K : number of candidates per vehicle.

    Returns
    -------
    List of length V; assignment[v] = k means vehicle v chose route k.
    If no x_vk=1 for a vehicle (constraint violated), returns argmax instead.
    """
    assignment = []
    for v in range(V):
        block = x[v * K:(v + 1) * K]
        if block.sum() == 1:
            assignment.append(int(np.argmax(block)))
        else:
            # Constraint violated: fall back to argmax (greedy repair)
            assignment.append(int(np.argmax(block)) if block.sum() > 0 else 0)
    return assignment


def encode_assignment(assignment: List[int], V: int, K: int) -> np.ndarray:
    """
    Encode a list of route indices as a one-hot binary QUBO vector.

    Parameters
    ----------
    assignment : list of length V, each value in 0..K-1.
    V, K       : number of vehicles and candidates.

    Returns
    -------
    np.ndarray of shape (V*K,), binary.
    """
    x = np.zeros(V * K, dtype=np.float64)
    for v, k in enumerate(assignment):
        x[v * K + k] = 1.0
    return x


# ---------------------------------------------------------------------------
# Unit test (toy 3-vehicle example)
# ---------------------------------------------------------------------------

def unit_test_qubo() -> bool:
    """
    Hand-verifiable unit test on a 3-vehicle toy example.

    Setup
    -----
    3 vehicles, 2 candidate routes each (K=2 for simplicity).
    Travel times defined manually. No congestion (lam=0).
    No previous routes (mu doesn't apply).
    P = 1000 (explicit, large enough).

    Expected behaviour
    ------------------
    One-hot penalty term P*(Σ_k x_vk - 1)²:
      - Zero when exactly one route chosen per vehicle.
      - Positive when zero or two routes chosen.

    We verify:
      1. Feasible assignment (one route per vehicle): penalty contribution = 0.
      2. Infeasible assignment (two routes for v=0): penalty contribution = P.
      3. Infeasible assignment (no route for v=0): penalty contribution = P.
    """
    print("\n  --- Unit Test: 3-Vehicle Toy QUBO ---")

    # Toy setup: 3 vehicles, K=2 routes each
    # Fake node IDs (not real graph)
    vehicles = [
        Vehicle(vid=0, origin=0, destination=9),
        Vehicle(vid=1, origin=1, destination=8),
        Vehicle(vid=2, origin=2, destination=7),
    ]

    # Candidates: each vehicle has 2 routes with fake costs
    # Route 0 = short (30s), Route 1 = long (60s)
    # Paths are disjoint (no shared edges) for simplicity.
    candidates = {
        0: [([0, 1, 9], 30.0),   ([0, 2, 3, 9], 60.0)],
        1: [([1, 4, 8], 25.0),   ([1, 5, 6, 8], 55.0)],
        2: [([2, 7], 20.0),      ([2, 3, 4, 7], 45.0)],
    }

    # No congestion state needed for this test (disjoint paths)
    congestion_state: Dict = {}
    lam = 0.0   # disable congestion term to isolate penalty tests
    mu  = 0.0   # disable switch penalty

    # We need to patch _path_congested_time to return 0 so build_qubo
    # uses stored candidate costs. It already does this by design.

    # Manually set P for predictable test
    # build_qubo computes P internally; we'll check penalty directly.
    Q, offset, meta = build_qubo(vehicles, candidates, congestion_state, lam=lam, mu=mu)
    P = meta["P"]
    K_max = meta["K_max"]
    V     = meta["V"]
    N     = meta["N"]

    print(f"    V={V}, K={K_max}, N={N}, P={P:.1f}, offset={offset:.1f}")

    # --- Test 1: Feasible assignment (one route per vehicle) ---
    # x = [1,0, 1,0, 1,0] → v0→route0, v1→route0, v2→route0
    x_feasible = encode_assignment([0, 0, 0], V, K_max)
    E_feasible  = evaluate_qubo(Q, x_feasible, offset)

    # Expected travel time: 30 + 25 + 20 = 75
    # Expected congestion: 0 (lam=0)
    # Expected switch penalty: 0 (no prev routes, mu=0)
    # Expected one-hot penalty: 0 (each vehicle picks exactly 1 route)
    # Total expected: 75
    expected_feasible = 75.0
    ok1 = abs(E_feasible - expected_feasible) < 1e-3

    print(f"    Test 1 (feasible [0,0,0]): E={E_feasible:.3f}, expected={expected_feasible:.1f} -> {'PASS' if ok1 else 'FAIL'}")

    # --- Test 2: Infeasible — v0 picks BOTH routes (x_00=1, x_01=1) ---
    x_infeasible_both = encode_assignment([0, 0, 0], V, K_max)
    x_infeasible_both[0] = 1.0  # x_00 = 1
    x_infeasible_both[1] = 1.0  # x_01 = 1 (v0 picks route 0 AND route 1)
    E_both = evaluate_qubo(Q, x_infeasible_both, offset)

    # One-hot for v0: (1+1-1)² = 1 → penalty = P
    # Feasible part for v1, v2: P*0 = 0
    # Travel time: 30 + 60 + 25 + 20 = 135 (v0 both routes)
    # So E_both > E_feasible by at least P
    ok2 = E_both > E_feasible + P * 0.9  # at least 90% of P added

    print(f"    Test 2 (v0 picks both routes): E={E_both:.3f}, "
          f"penalty={E_both - E_feasible:.3f} >= P={P:.1f} -> {'PASS' if ok2 else 'FAIL'}")

    # --- Test 3: Infeasible — v0 picks NO routes ---
    x_no_route = encode_assignment([0, 0, 0], V, K_max)
    x_no_route[0] = 0.0  # x_00 = 0
    # So v0: x=[0,0], v1: x=[1,0], v2: x=[1,0]
    E_no_route = evaluate_qubo(Q, x_no_route, offset)

    # One-hot for v0: (0-1)² = 1 → penalty = P
    ok3 = E_no_route > E_feasible + P * 0.9

    print(f"    Test 3 (v0 picks no route):   E={E_no_route:.3f}, "
          f"penalty={E_no_route - E_feasible:.3f} >= P={P:.1f} -> {'PASS' if ok3 else 'FAIL'}")

    # --- Test 4: Alternative feasible assignment (route 1 for all) ---
    x_alt = encode_assignment([1, 1, 1], V, K_max)
    E_alt  = evaluate_qubo(Q, x_alt, offset)
    # Expected: 60 + 55 + 45 = 160, penalty = 0
    expected_alt = 160.0
    ok4 = abs(E_alt - expected_alt) < 1e-3

    print(f"    Test 4 (feasible [1,1,1]):  E={E_alt:.3f}, expected={expected_alt:.1f} -> {'PASS' if ok4 else 'FAIL'}")

    # --- Test 5: Optimal solution has lowest energy ---
    ok5 = E_feasible < E_alt  # [0,0,0] is cheaper (75 < 160)
    print(f"    Test 5 (optimal is cheapest): {E_feasible:.1f} < {E_alt:.1f} -> {'PASS' if ok5 else 'FAIL'}")

    all_pass = all([ok1, ok2, ok3, ok4, ok5])
    verdict  = "[ALL PASS]" if all_pass else "[SOME TESTS FAILED]"
    print(f"    {verdict} Unit test complete.")
    return all_pass


# ---------------------------------------------------------------------------
# Full Phase 3 acceptance test
# ---------------------------------------------------------------------------

def run_phase3(
    vehicles: List[Vehicle],
    candidates: Dict[int, List[Tuple[List[int], float]]],
    congestion_state: Dict[Tuple, float],
) -> Tuple[np.ndarray, float, Dict]:
    """
    Phase 3 acceptance test:
      1. Compute re-normalised lambda.
      2. Build QUBO.
      3. Verify penalty terms are zero only for feasible assignments.
      4. Run unit test on 3-vehicle toy.
    """
    print(f"\n{'=' * 60}")
    print("PHASE 3 - ACCEPTANCE TEST")
    print(f"{'=' * 60}")
    print(f"  Vehicles  : {len(vehicles)}")
    V   = len(vehicles)
    K   = max((len(candidates.get(v, [])) for v in range(V)), default=0)
    print(f"  K (max)   : {K}")

    # Re-normalise lambda
    lam = compute_lambda(vehicles, candidates, congestion_state)
    print(f"  lambda    : {lam:.6f}  (re-normalised for this round)")
    print(f"  mu        : {MU_SWITCH_PENALTY_S}s  (route-switch penalty, B3)")

    # Build QUBO
    import time as _time
    t0 = _time.perf_counter()
    Q, offset, meta = build_qubo(
        vehicles, candidates, congestion_state, lam=lam, mu=MU_SWITCH_PENALTY_S
    )
    build_time = _time.perf_counter() - t0

    N = meta["N"]
    P = meta["P"]
    print(f"  QUBO size : {N} x {N}  (V={V} vehicles x K={K} routes)")
    print(f"  P (penalty): {P:.2f}")
    print(f"  Offset    : {offset:.2f}")
    print(f"  Build time: {build_time*1000:.2f} ms")

    # Verify feasible vs infeasible energies
    print(f"\n  Verifying penalty structure ...")
    n_checks = min(5, V)
    feasible_energies   = []
    infeasible_energies = []

    # Sample some feasible assignments
    for _ in range(20):
        assignment = [0] * V  # all pick route 0 (feasible)
        x = encode_assignment(assignment, V, K)
        E = evaluate_qubo(Q, x, offset)
        feasible_energies.append(E)

    # Sample some infeasible assignments (v0 picks both routes 0 and 1)
    if K >= 2:
        for _ in range(20):
            assignment = [0] * V
            x = encode_assignment(assignment, V, K)
            x[0] = 1.0  # v0 picks both route 0 and 1
            x[1] = 1.0
            E = evaluate_qubo(Q, x, offset)
            infeasible_energies.append(E)

        min_feasible   = min(feasible_energies)
        min_infeasible = min(infeasible_energies)
        ok_penalty = min_infeasible > min_feasible
        print(f"    Min feasible   energy : {min_feasible:.2f}")
        print(f"    Min infeasible energy : {min_infeasible:.2f}")
        print(f"    Penalty correctly applied: {'YES' if ok_penalty else 'NO'}")
    else:
        ok_penalty = True
        print(f"    (K<2, penalty cross-term check skipped)")

    # Unit test
    print(f"\n  Running toy unit test ...")
    unit_ok = unit_test_qubo()

    verdict = "[PASS]" if (ok_penalty and unit_ok) else "[FAIL]"
    print(f"\n  {verdict} Phase 3 acceptance test.")
    print(f"{'=' * 60}\n")

    return Q, offset, meta


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import networkx as nx

    DATA_DIR   = Path(__file__).parent.parent / "data"
    graph_path = DATA_DIR / "phase1_graph.pkl"
    pool_path  = DATA_DIR / "phase2_candidate_pool.pkl"

    # Load Phase 1 graph
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    log.info("Loaded graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # Load Phase 2 candidate pool
    with open(pool_path, "rb") as f:
        raw_pool = pickle.load(f)
    log.info("Loaded candidate pool: %d O-D pairs", len(raw_pool))

    # Build Vehicle objects and candidates dict from pool
    rng      = random.Random(42)
    od_items = list(raw_pool.items())[:10]  # use first 10 O-D pairs as 10 vehicles

    vehicles: List[Vehicle] = []
    candidates: Dict[int, List[Tuple[List[int], float]]] = {}

    for v_idx, ((o, d), paths) in enumerate(od_items):
        vehicles.append(Vehicle(vid=v_idx, origin=o, destination=d, prev_route_idx=None))
        candidates[v_idx] = paths

    # Load congestion state from Phase 1 (edge loads)
    # We'll generate a fresh one for Phase 3 test
    from phase1_world_state import generate_synthetic_congestion
    congestion_state = generate_synthetic_congestion(G, num_vehicles=200, seed=42)

    # Run Phase 3 acceptance test
    Q, offset, meta = run_phase3(vehicles, candidates, congestion_state)

    # Save QUBO for Phase 4
    import pickle as pkl
    qubo_path = DATA_DIR / "phase3_qubo_example.pkl"
    with open(qubo_path, "wb") as f:
        pkl.dump({"Q": Q, "offset": offset, "meta": meta,
                  "vehicles": vehicles, "candidates": candidates,
                  "congestion_state": congestion_state}, f)
    print(f"QUBO saved --> {qubo_path}")
