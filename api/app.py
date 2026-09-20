"""
Phase 7 — FastAPI Backend (free/self-hosted, no paid services)
==============================================================
Exposes REST + WebSocket endpoints for the RouteIQ optimisation pipeline.

Endpoints
---------
  POST   /scenarios                       — create a new scenario
  GET    /scenarios                       — list all scenarios
  GET    /scenarios/{id}                  — get scenario details
  DELETE /scenarios/{id}                  — delete scenario
  POST   /scenarios/{id}/optimize         — trigger optimisation (background task)
  WS     /scenarios/{id}/ws              — real-time stage progress stream
  POST   /scenarios/{id}/simulate-traffic — inject synthetic BPR congestion
  GET    /scenarios/{id}/benchmark        — compare Dijkstra / greedy / QPSO-hybrid
  GET    /health                          — liveness check

Progress stages streamed over WebSocket (in order):
  loading_graph → candidate_generation → qpso_selecting
  → local_search → stability_check → complete

Storage: SQLite via SQLModel (no hosting needed).
Async:   FastAPI BackgroundTasks + asyncio (no Celery / Redis).
Traffic: SIMULATED via BPR model — no live traffic API, no paid data.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

# Add core/ and api/ to sys.path so phase imports and models import work
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))                    # api/ — for models.py
sys.path.insert(0, str(_HERE.parent / "core"))    # core/ — for phase*.py

from models import (
    BenchmarkResult, OptimiseRequest, Scenario, ScenarioCreate,
    ScenarioRead, SimulateTrafficRequest, create_db, engine, get_session,
)

log = logging.getLogger("phase7")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RouteIQ API",
    description=(
        "System-optimal multi-vehicle route assignment (QPSO-hybrid). "
        "Congestion is SIMULATED (BPR model), not live traffic data. "
        "Free/open-source stack: OSMnx, FastAPI, SQLite, MapLibre GL JS."
    ),
    version="0.7.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    create_db()
    log.info("SQLite database initialised.")


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """
    Manages active WebSocket connections keyed by scenario_id.
    One scenario can have multiple concurrent browser connections.
    """

    def __init__(self) -> None:
        self._connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, scenario_id: int, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.setdefault(scenario_id, []).append(ws)
        log.info("WS connected: scenario %d (%d clients)",
                 scenario_id, len(self._connections[scenario_id]))

    def disconnect(self, scenario_id: int, ws: WebSocket) -> None:
        conns = self._connections.get(scenario_id, [])
        if ws in conns:
            conns.remove(ws)

    async def broadcast(self, scenario_id: int, payload: dict) -> None:
        """JSON-broadcast to all connected clients for this scenario."""
        dead: List[WebSocket] = []
        for ws in list(self._connections.get(scenario_id, [])):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(scenario_id, ws)


manager = ConnectionManager()

# ---------------------------------------------------------------------------
# In-memory caches (avoid re-downloading OSM data on every request)
# ---------------------------------------------------------------------------

_graph_cache:     Dict[int, Any] = {}   # scenario_id → enriched MultiDiGraph
_candidate_cache: Dict[int, Any] = {}   # scenario_id → candidates dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_or_load_graph(scenario: Scenario) -> Any:
    """
    Return cached graph, or download+enrich and cache it.

    Optimisation: if the phase1_graph.pkl already exists on disk AND the
    scenario center/radius matches the default (23.0285, 72.5546, 1000m),
    load it from disk instead of re-downloading from OSM (saves ~3-5s).
    Always keeps the result in _graph_cache for subsequent requests.
    """
    if scenario.id in _graph_cache:
        return _graph_cache[scenario.id]

    # Try disk cache for default Navrangpura area
    _DATA_DIR = _HERE.parent / "data"
    disk_cached = _DATA_DIR / "phase1_graph.pkl"
    is_default = (
        abs(scenario.center_lat - 23.0285) < 0.001
        and abs(scenario.center_lon - 72.5546) < 0.001
        and abs(scenario.radius_m - 1000.0) < 10.0
    )
    if is_default and disk_cached.exists():
        import pickle
        log.info("Loading graph from disk cache (phase1_graph.pkl)...")
        with open(disk_cached, "rb") as f:
            G = pickle.load(f)
        _graph_cache[scenario.id] = G
        return G

    from phase1_world_state import download_graph, enrich_graph, simplify_graph_for_routing
    G = await asyncio.to_thread(
        download_graph,
        center=(scenario.center_lat, scenario.center_lon),
        radius_m=scenario.radius_m,
    )
    G = await asyncio.to_thread(enrich_graph, G)

    # For large areas (radius > 3km), simplify the graph to remove redundant
    # degree-2 nodes. This reduces node count by 50-70% and makes Yen's K-shortest
    # paths 5-10x faster. BPR parameters are re-applied after simplification.
    if scenario.radius_m > 3000:
        log.info("Large radius (%.0fm) — simplifying graph for faster routing...", scenario.radius_m)
        G = await asyncio.to_thread(simplify_graph_for_routing, G)

    _graph_cache[scenario.id] = G
    return G


def _build_vehicles(scenario: Scenario):
    """Reconstruct Vehicle list and O-D pair list from stored JSON."""
    from phase3_qubo import Vehicle
    od_pairs_raw = json.loads(scenario.od_pairs_json)
    vehicles = [
        Vehicle(vid=v_idx, origin=o, destination=d)
        for v_idx, (o, d) in enumerate(od_pairs_raw)
    ]
    od_pairs = [(o, d) for o, d in od_pairs_raw]
    return vehicles, od_pairs


def _db_update(scenario_id: int, stage: str, status: str = "optimising", **kwargs) -> None:
    """Write a synchronous DB update (called from background task)."""
    with Session(engine) as s:
        sc = s.get(Scenario, scenario_id)
        if sc:
            sc.current_stage = stage
            sc.status        = status
            sc.updated_at    = datetime.utcnow()
            for k, v in kwargs.items():
                setattr(sc, k, v)
            s.add(sc)
            s.commit()


def _compute_metrics(
    assignment:       List[int],
    vehicles:         List[Any],
    candidates:       Dict[int, Any],
    G:                Any,
    congestion_state: Dict,
    algorithm:        str,
    runtime_s:        float,
) -> Dict:
    """
    Compute comparative metrics for one route assignment:
      - total / mean travel time (BPR-congested)
      - total / mean route distance
      - congestion cost  Σ load_e²
      - constraint violations (vehicles with no valid route)
    """
    from phase1_world_state import bpr_travel_time

    total_time   = 0.0
    total_dist   = 0.0
    n_violations = 0
    path_times:   List[float] = []
    path_lengths: List[float] = []
    edge_counts:  Dict[Any, int] = {}

    for v_idx, vehicle in enumerate(vehicles):
        k     = assignment[v_idx] if v_idx < len(assignment) else 0
        cands = candidates.get(v_idx, [])
        if not cands or k >= len(cands):
            n_violations += 1
            continue

        path, _ = cands[k]
        cong_t  = 0.0
        dist_m  = 0.0

        for u, w in zip(path[:-1], path[1:]):
            if not G.has_edge(u, w):
                continue
            edge_data = G[u][w]
            # Pick the parallel edge with minimum free-flow time
            best_ek = min(
                edge_data,
                key=lambda ek: edge_data[ek].get("free_flow_time", float("inf")),
            )
            d    = edge_data[best_ek]
            load = max(
                (congestion_state.get((u, w, ek), 0.0) for ek in range(5)),
                default=0.0,
            )
            cong_t += bpr_travel_time(
                d["free_flow_time"], load,
                d["capacity"], d["bpr_alpha"], d["bpr_beta"],
            )
            dist_m += d.get("length", 0.0)
            edge_counts[(u, w)] = edge_counts.get((u, w), 0) + 1

        total_time   += cong_t
        total_dist   += dist_m
        path_times.append(cong_t)
        path_lengths.append(dist_m)

    congestion_cost = float(sum(c ** 2 for c in edge_counts.values()))

    return {
        "algorithm":               algorithm,
        "total_travel_time_s":     round(total_time, 3),
        "mean_travel_time_s":      round(float(np.mean(path_times)) if path_times else 0.0, 3),
        "total_distance_m":        round(total_dist, 1),
        "mean_distance_m":         round(float(np.mean(path_lengths)) if path_lengths else 0.0, 1),
        "congestion_cost":         round(congestion_cost, 3),
        "n_constraint_violations": n_violations,
        "runtime_s":               round(runtime_s, 4),
        "n_vehicles":              len(vehicles),
    }


# ---------------------------------------------------------------------------
# Background optimisation task (full pipeline)
# ---------------------------------------------------------------------------

async def _optimise_task(
    scenario_id:          int,
    seed:                 int,
    n_vehicles_congestion: int,
) -> None:
    """
    Runs the full Phase 2→5 pipeline as a FastAPI BackgroundTask.
    Streams progress events via WebSocket at each stage transition.

    Stage sequence:
      loading_graph → candidate_generation → qpso_selecting
      → local_search → stability_check → complete
    """

    async def emit(stage: str, **data) -> None:
        payload = {"stage": stage, "scenario_id": scenario_id, **data}
        await manager.broadcast(scenario_id, payload)
        log.info("[scenario %d] stage=%s", scenario_id, stage)

    try:
        # Give the client 1 second to connect to the WebSocket before we start processing
        import asyncio
        await asyncio.sleep(1.0)
        
        # ----------------------------------------------------------------
        # Stage 1: loading_graph
        # ----------------------------------------------------------------
        await emit("loading_graph", msg="Downloading / loading road network ...")
        _db_update(scenario_id, "loading_graph")

        with Session(engine) as s:
            scenario = s.get(Scenario, scenario_id)
            if not scenario:
                return

        G = await _get_or_load_graph(scenario)
        await emit(
            "loading_graph",
            msg=f"Graph ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
        )

        vehicles, od_pairs = _build_vehicles(scenario)

        # Start total pipeline timer AFTER graph load (graph is cached after 1st run)
        t_pipeline_start = time.perf_counter()

        # ----------------------------------------------------------------
        # Stage 2: candidate_generation
        # ----------------------------------------------------------------
        await emit(
            "candidate_generation",
            msg=f"Generating {scenario.k_candidates} candidate paths per vehicle ...",
        )
        _db_update(scenario_id, "candidate_generation")

        from phase2_candidate_generation import build_candidate_pool
        raw_pool   = await asyncio.to_thread(build_candidate_pool, G, od_pairs, K=scenario.k_candidates)
        candidates = {
            v_idx: raw_pool.get((o, d), [])
            for v_idx, (o, d) in enumerate(od_pairs)
        }
        _candidate_cache[scenario_id] = candidates
        total_paths = sum(len(v) for v in candidates.values())

        await emit(
            "candidate_generation",
            msg=f"Generated {total_paths} candidate paths ({scenario.k_candidates} per vehicle)",
            total_paths=total_paths,
            k=scenario.k_candidates,
        )

        # ----------------------------------------------------------------
        # Stage 3: qpso_selecting  (QUBO build + QPSO)
        # ----------------------------------------------------------------
        await emit("qpso_selecting", msg="Building QUBO objective ...")
        _db_update(scenario_id, "qpso_selecting")

        from phase1_world_state import (
            apply_loads_to_graph, generate_synthetic_congestion,
        )
        from phase3_qubo import (
            MU_SWITCH_PENALTY_S, build_qubo, compute_lambda,
        )

        congestion_state = generate_synthetic_congestion(
            G, num_vehicles=n_vehicles_congestion, seed=seed,
        )
        apply_loads_to_graph(G, congestion_state)

        lam           = compute_lambda(vehicles, candidates, congestion_state)
        Q, offset, meta = build_qubo(
            vehicles, candidates, congestion_state,
            lam=lam, mu=MU_SWITCH_PENALTY_S,
        )
        V, K = meta["V"], meta["K_max"]

        await emit(
            "qpso_selecting",
            msg=f"QUBO built ({V*K}×{V*K}). Running QPSO (N=30, max_iter=150) ...",
            qubo_size=V * K,
            lam=round(lam, 6),
        )

        from phase4_qpso import run_qpso
        t_q        = time.perf_counter()
        qpso_result = await asyncio.to_thread(run_qpso, Q, offset, V, K, seed=seed)
        qpso_rt    = time.perf_counter() - t_q

        await emit(
            "qpso_selecting",
            msg=f"QPSO complete — energy={qpso_result.best_energy:.4f}, "
                f"iters={qpso_result.n_iters}, restarts={qpso_result.n_restarts}",
            energy=round(qpso_result.best_energy, 4),
            iters=qpso_result.n_iters,
            restarts=qpso_result.n_restarts,
            convergence=qpso_result.convergence,
            runtime_s=round(qpso_rt, 3),
        )

        # ----------------------------------------------------------------
        # Stage 4: local_search
        # ----------------------------------------------------------------
        await emit("local_search", msg="Running candidate-swap local search ...")
        _db_update(scenario_id, "local_search")

        from phase5_local_search import run_local_search_and_guard
        t_ls      = time.perf_counter()
        ls_result = await asyncio.to_thread(
            run_local_search_and_guard,
            initial_assignment=qpso_result.best_assignment,
            prev_assignment=None,
            Q=Q, offset=offset, meta=meta,
            candidates=candidates,
        )

        await emit(
            "local_search",
            msg=f"Local search: {ls_result.n_swaps} improving swaps "
                f"in {ls_result.n_passes} passes",
            swaps=ls_result.n_swaps,
            passes=ls_result.n_passes,
            energy_before=round(ls_result.initial_energy, 4),
            energy_after=round(ls_result.refined_energy, 4),
        )

        # ----------------------------------------------------------------
        # Stage 5: stability_check
        # ----------------------------------------------------------------
        await emit(
            "stability_check",
            msg=f"Hysteresis guard applied — "
                f"{ls_result.n_hysteresis_blocked} route changes blocked (8% margin)",
            blocked=ls_result.n_hysteresis_blocked,
        )
        _db_update(scenario_id, "stability_check")

        # ----------------------------------------------------------------
        # Complete
        # ----------------------------------------------------------------
        assignment      = ls_result.stable_assignment
        t_pipeline_total = time.perf_counter() - t_pipeline_start   # full server time

        metrics = _compute_metrics(
            assignment, vehicles, candidates, G, congestion_state,
            "qpso_hybrid", t_pipeline_total,
        )

        # Add per-stage breakdown so the UI can show it
        metrics["runtime_breakdown"] = {
            "candidate_gen_s": round(t_pipeline_total - qpso_rt - (time.perf_counter() - t_pipeline_start - t_pipeline_total + t_pipeline_total), 4),
            "qpso_s":          round(qpso_rt, 4),
            "total_pipeline_s": round(t_pipeline_total, 4),
        }

        _db_update(
            scenario_id, "complete", "complete",
            assignment_json=json.dumps(assignment),
            metrics_json=json.dumps(metrics),
        )

        await emit(
            "complete",
            msg=f"Optimisation complete. Total server time: {t_pipeline_total:.2f}s",
            assignment=assignment,
            metrics=metrics,
            final_energy=round(ls_result.refined_energy, 4),
            runtime_breakdown={
                "qpso_s":           round(qpso_rt, 3),
                "total_pipeline_s": round(t_pipeline_total, 3),
            },
        )

    except Exception as exc:
        log.exception("Optimisation failed for scenario %d", scenario_id)
        _db_update(scenario_id, "error", "error", error_msg=str(exc)[:500])
        await emit("error", msg=str(exc))


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "version": "0.7.0",
        "note":    "Congestion is SIMULATED (BPR model). Not live traffic data.",
    }


@app.post("/scenarios", response_model=ScenarioRead, status_code=201)
def create_scenario(
    body:    ScenarioCreate,
    session: Session = Depends(get_session),
) -> Scenario:
    """
    Create a new scenario. Samples valid O-D pairs from the road network.
    Graph is downloaded (OSMnx / free OSM data) and cached in memory.
    """
    import networkx as nx
    from phase1_world_state import download_graph, enrich_graph

    G     = download_graph(center=(body.center_lat, body.center_lon), radius_m=body.radius_m)
    G     = enrich_graph(G)
    nodes = list(G.nodes())

    rng      = random.Random(42)
    od_pairs: List[List[int]] = []
    attempts = 0
    while len(od_pairs) < body.num_vehicles and attempts < body.num_vehicles * 20:
        o, d = rng.sample(nodes, 2)
        if nx.has_path(G, o, d):
            od_pairs.append([int(o), int(d)])
        attempts += 1

    sc = Scenario.model_validate(body)
    sc.od_pairs_json = json.dumps(od_pairs)
    sc.status        = "created"
    session.add(sc)
    session.commit()
    session.refresh(sc)

    _graph_cache[sc.id] = G
    log.info("Created scenario %d — %d vehicles, %d O-D pairs sampled",
             sc.id, sc.num_vehicles, len(od_pairs))
    return sc


@app.get("/scenarios", response_model=List[ScenarioRead])
def list_scenarios(session: Session = Depends(get_session)):
    return list(session.exec(select(Scenario)).all())


@app.get("/scenarios/{scenario_id}", response_model=ScenarioRead)
def get_scenario(scenario_id: int, session: Session = Depends(get_session)):
    sc = session.get(Scenario, scenario_id)
    if not sc:
        raise HTTPException(404, "Scenario not found")
    return sc


@app.delete("/scenarios/{scenario_id}")
def delete_scenario(scenario_id: int, session: Session = Depends(get_session)):
    sc = session.get(Scenario, scenario_id)
    if not sc:
        raise HTTPException(404, "Scenario not found")
    session.delete(sc)
    session.commit()
    _graph_cache.pop(scenario_id, None)
    _candidate_cache.pop(scenario_id, None)
    return {"deleted": scenario_id}


@app.post("/scenarios/{scenario_id}/optimize")
async def optimize_scenario(
    scenario_id:     int,
    body:            OptimiseRequest,
    background_tasks: BackgroundTasks,
    session:         Session = Depends(get_session),
):
    """
    Trigger the full optimisation pipeline as a background task.
    Progress is streamed in real-time via WebSocket at /scenarios/{id}/ws.

    Stage order: loading_graph → candidate_generation → qpso_selecting
                 → local_search → stability_check → complete
    """
    sc = session.get(Scenario, scenario_id)
    if not sc:
        raise HTTPException(404, "Scenario not found")
    if sc.status == "optimising":
        raise HTTPException(409, "Scenario is already being optimised")

    _db_update(scenario_id, "queued", "optimising")

    background_tasks.add_task(
        _optimise_task,
        scenario_id=scenario_id,
        seed=body.seed,
        n_vehicles_congestion=body.n_vehicles_congestion,
    )

    return {
        "status":      "optimising",
        "scenario_id": scenario_id,
        "ws_url":      f"/scenarios/{scenario_id}/ws",
        "note":        "Connect to ws_url to receive real-time stage updates.",
    }


@app.websocket("/scenarios/{scenario_id}/ws")
async def websocket_progress(scenario_id: int, ws: WebSocket):
    """
    WebSocket endpoint — receives JSON progress events from the optimisation
    pipeline.  Stages: loading_graph, candidate_generation, qpso_selecting,
    local_search, stability_check, complete, error.

    The connection is kept alive with 60-second keepalive pings.
    """
    await manager.connect(scenario_id, ws)
    try:
        while True:
            try:
                # Wait for any client message (e.g. ping) — 60s timeout
                await asyncio.wait_for(ws.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                await ws.send_json({"stage": "ping", "scenario_id": scenario_id})
    except WebSocketDisconnect:
        manager.disconnect(scenario_id, ws)
        log.info("WS disconnected: scenario %d", scenario_id)


@app.post("/scenarios/{scenario_id}/simulate-traffic")
async def simulate_traffic(
    scenario_id: int,
    body:        SimulateTrafficRequest,
    session:     Session = Depends(get_session),
):
    """
    Inject a synthetic BPR congestion event.
    Replaces live-traffic API — fully reproducible, free, no API key needed.

    Scenario types
    --------------
    normal   → free_flow  (light, 3600s window)
    moderate → moderate   (medium, 1800s window)
    heavy    → rush_hour  (heavy, 900s window)
    blockage → rush_hour  (300+ vehicles, simulates road blockage)
    """
    sc = session.get(Scenario, scenario_id)
    if not sc:
        raise HTTPException(404, "Scenario not found")

    scenario_map = {
        "normal":   "free_flow",
        "moderate": "moderate",
        "heavy":    "rush_hour",
        "blockage": "rush_hour",
    }
    internal_scenario = scenario_map.get(body.scenario_type, "rush_hour")
    num_veh = body.num_vehicles if body.scenario_type != "blockage" else max(body.num_vehicles, 300)

    G = await _get_or_load_graph(sc)

    from phase1_world_state import apply_loads_to_graph, generate_synthetic_congestion
    edge_loads = generate_synthetic_congestion(
        G, num_vehicles=num_veh, seed=body.seed, scenario=internal_scenario,
    )
    apply_loads_to_graph(G, edge_loads)
    _graph_cache[scenario_id] = G

    loaded_edges = len(edge_loads)
    max_load     = round(float(max(edge_loads.values())), 1) if edge_loads else 0.0

    await manager.broadcast(scenario_id, {
        "stage":         "congestion_injected",
        "scenario_type": body.scenario_type,
        "num_vehicles":  num_veh,
        "loaded_edges":  loaded_edges,
        "max_load_vph":  max_load,
        "note":          "Congestion is SIMULATED (BPR model). Not live traffic data.",
    })

    return {
        "scenario_id":   scenario_id,
        "congestion_type": body.scenario_type,
        "num_vehicles":  num_veh,
        "loaded_edges":  loaded_edges,
        "max_load_vph":  max_load,
        "note":          "Congestion is SIMULATED (BPR model). Not live traffic data.",
    }


@app.get("/scenarios/{scenario_id}/benchmark")
async def benchmark_scenario(
    scenario_id:          int,
    seed:                 int = 0,
    n_vehicles_congestion: int = 200,
    session:              Session = Depends(get_session),
):
    """
    Run three routing algorithms on the same instance and return
    comparative metrics (report.pdf §8 requirement).

    Algorithms compared
    -------------------
    1. dijkstra_all_shortest — every vehicle takes its personal shortest path
    2. k_shortest_greedy     — pick least-congested route from k candidates
    3. qpso_hybrid           — Phase 4 QPSO + Phase 5 local search (ours)

    Results are sorted ascending by total_travel_time_s.
    """
    sc = session.get(Scenario, scenario_id)
    if not sc:
        raise HTTPException(404, "Scenario not found")

    G          = await _get_or_load_graph(sc)
    vehicles, od_pairs = _build_vehicles(sc)

    from phase1_world_state import apply_loads_to_graph, generate_synthetic_congestion
    from phase2_candidate_generation import build_candidate_pool
    from phase3_qubo import MU_SWITCH_PENALTY_S, build_qubo, compute_lambda
    from phase4_qpso import run_qpso
    from phase5_local_search import run_local_search_and_guard

    cong_state = generate_synthetic_congestion(
        G, num_vehicles=n_vehicles_congestion, seed=seed,
    )
    apply_loads_to_graph(G, cong_state)

    raw_pool   = build_candidate_pool(G, od_pairs, K=sc.k_candidates)
    candidates = {
        v_idx: raw_pool.get((o, d), [])
        for v_idx, (o, d) in enumerate(od_pairs)
    }

    results: List[Dict] = []

    # 1. Dijkstra all-shortest
    t0 = time.perf_counter()
    dij_assign = [0] * len(vehicles)
    results.append(_compute_metrics(
        dij_assign, vehicles, candidates, G, cong_state,
        "dijkstra_all_shortest", time.perf_counter() - t0,
    ))

    # 2. k-shortest greedy (pick least-congested candidate)
    t0 = time.perf_counter()
    greedy_assign: List[int] = []
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
            total_c = ff_cost + cong_add
            if total_c < best_c:
                best_c, best_k = total_c, k
        greedy_assign.append(best_k)
    results.append(_compute_metrics(
        greedy_assign, vehicles, candidates, G, cong_state,
        "k_shortest_greedy", time.perf_counter() - t0,
    ))

    # 3. QPSO-hybrid
    lam           = compute_lambda(vehicles, candidates, cong_state)
    Q, offset, meta = build_qubo(vehicles, candidates, cong_state, lam=lam, mu=MU_SWITCH_PENALTY_S)
    V, K           = meta["V"], meta["K_max"]

    t0             = time.perf_counter()
    qpso_result    = run_qpso(Q, offset, V, K, seed=seed)
    ls_result      = run_local_search_and_guard(
        qpso_result.best_assignment, None, Q, offset, meta, candidates,
    )
    qpso_rt = time.perf_counter() - t0

    results.append(_compute_metrics(
        ls_result.stable_assignment, vehicles, candidates, G, cong_state,
        "qpso_hybrid", qpso_rt,
    ))

    results.sort(key=lambda r: r["total_travel_time_s"])

    return {
        "scenario_id":          scenario_id,
        "seed":                 seed,
        "n_vehicles_congestion": n_vehicles_congestion,
        "results":              results,
        "note": "Congestion is SIMULATED (BPR model). Not live traffic data.",
    }


@app.get("/scenarios/{scenario_id}/routes")
async def get_scenario_routes(scenario_id: int, session: Session = Depends(get_session)):
    """Return the final assignment as a GeoJSON FeatureCollection."""
    sc = session.get(Scenario, scenario_id)
    if not sc:
        raise HTTPException(404, "Scenario not found")
    if not sc.assignment_json or sc.assignment_json == "[]":
        return {"type": "FeatureCollection", "features": []}
        
    G = await _get_or_load_graph(sc)
    vehicles, od_pairs = _build_vehicles(sc)
    
    if scenario_id not in _candidate_cache:
        from phase2_candidate_generation import build_candidate_pool
        raw_pool = build_candidate_pool(G, od_pairs, K=sc.k_candidates)
        candidates = {v_idx: raw_pool.get((o, d), []) for v_idx, (o, d) in enumerate(od_pairs)}
        _candidate_cache[scenario_id] = candidates
    else:
        candidates = _candidate_cache[scenario_id]
        
    assignment = json.loads(sc.assignment_json)
    features = []
    
    # Assign a color palette
    colors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"]
    
    for v_idx, vehicle in enumerate(vehicles):
        k = assignment[v_idx] if v_idx < len(assignment) else 0
        cands = candidates.get(v_idx, [])
        if not cands or k >= len(cands):
            continue
            
        path, cost = cands[k]
        coords = []
        for node in path:
            x = G.nodes[node].get('x')
            y = G.nodes[node].get('y')
            if x is not None and y is not None:
                coords.append([x, y])
                
        features.append({
            "type": "Feature",
            "properties": {
                "vehicle": v_idx, 
                "cost": cost,
                "color": colors[v_idx % len(colors)]
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            }
        })
        
    return {"type": "FeatureCollection", "features": features}

# Mount static frontend
from fastapi.staticfiles import StaticFiles
_STATIC_DIR = Path(__file__).parent.parent / "frontend"
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="frontend")
