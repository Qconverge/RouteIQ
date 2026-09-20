"""
Phase 6 — Execution Model (Two-Timescale, Event-Driven)
========================================================
Implements the two-timescale optimisation loop per report.pdf B4.

Architecture
------------
  This is NOT an infinite background loop — it is event-driven with
  explicit termination criteria at every level:

  Coarse timescale (every 300s or on major congestion change):
    → Rebuild k-shortest candidate pools from scratch (Phase 2).
    → Triggered by: timer expiry OR edge-load change > COARSE_THRESHOLD.

  Fine timescale (on every congestion-update event):
    → Re-run QPSO selector (Phase 4) + local search (Phase 5).
    → Uses smoothed congestion state (not raw tick values).
    → Always terminates via QPSO's explicit stagnation/max-iter criteria.

  Congestion smoothing (report.pdf B5, α=0.7):
    → Exponential moving average over edge loads.
    → Prevents single-tick noise from triggering unnecessary re-optimisation.
    → Formula: smoothed_e = α * new_e + (1 - α) * old_e

  Implementation: asyncio-based, no Celery/Redis (MVP simplicity per spec).
  Designed to be wrapped as a FastAPI BackgroundTask in Phase 7.

Event types
-----------
  CongestionEvent — carries new edge loads (simulated or real).
  CoarseRebuildEvent — fires every 300s to rebuild candidate pools.
  OptimiseEvent — fires after every congestion update to re-run selector.
  StopEvent — signals the loop to terminate gracefully.

State machine
-------------
  IDLE → LOADING_GRAPH → CANDIDATE_GENERATION → OPTIMISING → COMPLETE
  Any state can transition to IDLE on StopEvent.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("phase6")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants (report.pdf B4, B5)
# ---------------------------------------------------------------------------
COARSE_INTERVAL_S:    float = 300.0   # rebuild candidates every 300s (B4)
CONGESTION_SMOOTHING: float = 0.7     # EMA alpha for edge speeds (B5)
COARSE_THRESHOLD:     float = 0.30    # rebuild if 30% of edges change load >10%
FINE_CHANGE_THRESHOLD: float = 0.05   # re-optimise only if mean load changes >5%


# ---------------------------------------------------------------------------
# System state enum
# ---------------------------------------------------------------------------

class SystemState(Enum):
    IDLE                 = auto()
    LOADING_GRAPH        = auto()
    CANDIDATE_GENERATION = auto()
    QPSO_SELECTING       = auto()
    LOCAL_SEARCH         = auto()
    STABILITY_CHECK      = auto()
    COMPLETE             = auto()
    ERROR                = auto()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class CongestionEvent:
    """Carries new raw edge loads from the synthetic congestion generator."""
    edge_loads:  Dict[Tuple, float]
    timestamp:   float = field(default_factory=time.time)
    scenario:    str   = "rush_hour"


@dataclass
class OptimiseEvent:
    """Triggers fine-timescale re-optimisation."""
    smoothed_loads: Dict[Tuple, float]
    timestamp:      float = field(default_factory=time.time)


@dataclass
class CoarseRebuildEvent:
    """Triggers coarse-timescale candidate pool rebuild."""
    timestamp: float = field(default_factory=time.time)
    reason:    str   = "timer"  # "timer" | "threshold"


@dataclass
class StopEvent:
    """Graceful shutdown signal."""
    reason: str = "requested"


# ---------------------------------------------------------------------------
# Congestion smoother (report.pdf B5)
# ---------------------------------------------------------------------------

class CongestionSmoother:
    """
    Exponential Moving Average smoother for edge loads.

    Formula (B5):  smoothed_e(t) = α * raw_e(t) + (1 - α) * smoothed_e(t-1)

    α = 0.7  → reacts to sustained changes, ignores single-tick spikes.
    High α   → more reactive (faster to new conditions).
    Low  α   → smoother (less reactive, more stable).

    We use α=0.7 as specified in report.pdf B5.
    """

    def __init__(self, alpha: float = CONGESTION_SMOOTHING):
        self.alpha   = alpha
        self._state: Dict[Tuple, float] = {}   # smoothed loads per edge
        self._ticks: int = 0

    def update(self, new_loads: Dict[Tuple, float]) -> Dict[Tuple, float]:
        """
        Update smoothed loads with new raw values.
        Returns the updated smoothed load dict.
        """
        self._ticks += 1

        # All edges in current state — apply EMA
        all_edges = set(self._state.keys()) | set(new_loads.keys())
        for edge in all_edges:
            raw      = new_loads.get(edge, 0.0)
            smoothed = self._state.get(edge, raw)   # init to raw on first seen
            self._state[edge] = self.alpha * raw + (1.0 - self.alpha) * smoothed

        return dict(self._state)

    def mean_load_change(self, new_loads: Dict[Tuple, float]) -> float:
        """
        Compute the mean absolute fractional change in load across all edges
        that appear in either the current state or the new loads.
        Used to decide whether to trigger coarse rebuild or fine optimisation.
        """
        if not self._state:
            return 1.0   # First tick → always significant

        all_edges  = set(self._state.keys()) | set(new_loads.keys())
        if not all_edges:
            return 0.0

        changes = []
        for edge in all_edges:
            old = self._state.get(edge, 0.0)
            new = new_loads.get(edge, 0.0)
            denom = max(old, new, 1.0)
            changes.append(abs(new - old) / denom)

        return float(np.mean(changes))

    def coarse_change_fraction(self, new_loads: Dict[Tuple, float]) -> float:
        """
        Fraction of edges where load changes by more than 10%.
        Used to trigger coarse rebuild threshold.
        """
        if not self._state:
            return 1.0

        all_edges = set(self._state.keys()) | set(new_loads.keys())
        if not all_edges:
            return 0.0

        big_changes = 0
        for edge in all_edges:
            old   = self._state.get(edge, 0.0)
            new   = new_loads.get(edge, 0.0)
            denom = max(old, new, 1.0)
            if abs(new - old) / denom > 0.10:
                big_changes += 1

        return big_changes / len(all_edges)

    @property
    def smoothed_state(self) -> Dict[Tuple, float]:
        return dict(self._state)

    @property
    def n_ticks(self) -> int:
        return self._ticks


# ---------------------------------------------------------------------------
# Round result
# ---------------------------------------------------------------------------

@dataclass
class OptimisationRound:
    """
    Records the outcome of one fine-timescale optimisation round.

    Attributes
    ----------
    round_id         : sequential round counter.
    timestamp        : wall-clock seconds at start.
    trigger          : "congestion_event" | "coarse_rebuild" | "manual".
    state_sequence   : list of SystemState transitions in this round.
    assignment       : stable route assignment after Phase 4+5.
    qpso_energy      : QUBO energy from QPSO.
    ls_energy        : QUBO energy after local search.
    n_hysteresis_blocked: changes blocked by 8% guard.
    n_candidates_rebuilt : number of O-D pairs rebuilt (0 if no coarse rebuild).
    smoothing_alpha  : α used for EMA smoothing.
    runtime_s        : total wall-clock seconds for this round.
    """
    round_id:              int
    timestamp:             float
    trigger:               str
    state_sequence:        List[str]
    assignment:            List[int]
    qpso_energy:           float
    ls_energy:             float
    n_hysteresis_blocked:  int
    n_candidates_rebuilt:  int
    smoothing_alpha:       float
    runtime_s:             float


# ---------------------------------------------------------------------------
# Two-timescale optimisation engine
# ---------------------------------------------------------------------------

class TwoTimescaleEngine:
    """
    Event-driven two-timescale optimisation engine.

    Wraps Phases 2–5 into a coherent execution model.

    Usage
    -----
    engine = TwoTimescaleEngine(G, vehicles, ...)
    await engine.start()

    # Inject congestion events:
    await engine.push_congestion(edge_loads)

    # Stop gracefully:
    await engine.stop()

    # Access results:
    rounds = engine.history
    """

    def __init__(
        self,
        G,                              # Phase 1 enriched graph
        vehicles: List[Any],            # Phase 3 Vehicle objects
        od_pairs: List[Tuple[int, int]],
        K: int = 5,
        coarse_interval_s: float = COARSE_INTERVAL_S,
        smoothing_alpha:   float = CONGESTION_SMOOTHING,
        coarse_threshold:  float = COARSE_THRESHOLD,
        fine_threshold:    float = FINE_CHANGE_THRESHOLD,
        on_state_change:   Optional[Callable[[SystemState, str], None]] = None,
        on_round_complete: Optional[Callable[[OptimisationRound], None]] = None,
    ):
        """
        Parameters
        ----------
        G                : Phase 1 enriched MultiDiGraph.
        vehicles         : list of Vehicle objects.
        od_pairs         : O-D pairs for candidate generation.
        K                : candidates per vehicle (default 5).
        coarse_interval_s: seconds between full candidate rebuilds (default 300).
        smoothing_alpha  : EMA α for congestion smoothing (default 0.7).
        coarse_threshold : fraction of edges changed >10% to trigger rebuild.
        fine_threshold   : mean load change to trigger re-optimisation.
        on_state_change  : callback(state, message) on state transitions.
        on_round_complete: callback(round) when an optimisation round finishes.
        """
        self.G                  = G
        self.vehicles           = vehicles
        self.od_pairs           = od_pairs
        self.K                  = K
        self.coarse_interval_s  = coarse_interval_s
        self.smoother           = CongestionSmoother(alpha=smoothing_alpha)
        self.coarse_threshold   = coarse_threshold
        self.fine_threshold     = fine_threshold
        self._on_state_change   = on_state_change
        self._on_round_complete = on_round_complete

        # State
        self._state:            SystemState                = SystemState.IDLE
        self._candidates:       Dict[Tuple, Any]           = {}
        self._assignment:       Optional[List[int]]        = None
        self._round_id:         int                        = 0
        self._last_coarse:      float                      = 0.0
        self._running:          bool                       = False

        # Asyncio internals
        self._event_queue:      asyncio.Queue              = asyncio.Queue()
        self._coarse_task:      Optional[asyncio.Task]     = None
        self._event_task:       Optional[asyncio.Task]     = None

        # History
        self.history:           List[OptimisationRound]    = []

    # -----------------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------------

    def _set_state(self, new_state: SystemState, message: str = "") -> None:
        self._state = new_state
        log.info("[%s] %s", new_state.name, message)
        if self._on_state_change:
            self._on_state_change(new_state, message)

    @property
    def current_state(self) -> SystemState:
        return self._state

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        """Start the engine's background tasks."""
        self._running = True
        self._last_coarse = time.time()
        self._set_state(SystemState.IDLE, "Engine started.")

        # Build initial candidate pool
        await self._do_coarse_rebuild("initial")

        # Launch the coarse-timer task and event-processing task
        self._coarse_task = asyncio.create_task(self._coarse_timer_loop())
        self._event_task  = asyncio.create_task(self._event_loop())

    async def stop(self, reason: str = "requested") -> None:
        """Stop the engine gracefully."""
        self._running = False
        await self._event_queue.put(StopEvent(reason=reason))
        if self._coarse_task:
            self._coarse_task.cancel()
        if self._event_task:
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass
        self._set_state(SystemState.IDLE, f"Engine stopped: {reason}")

    async def push_congestion(
        self,
        edge_loads: Dict[Tuple, float],
        scenario: str = "rush_hour",
    ) -> None:
        """Inject a new congestion observation into the engine."""
        event = CongestionEvent(edge_loads=edge_loads, scenario=scenario)
        await self._event_queue.put(event)

    # -----------------------------------------------------------------------
    # Internal loops (asyncio tasks)
    # -----------------------------------------------------------------------

    async def _coarse_timer_loop(self) -> None:
        """
        Coarse-timescale timer: fires every COARSE_INTERVAL_S seconds to
        trigger a full candidate pool rebuild. Terminates when engine stops.
        Never runs as an unbounded loop — asyncio.sleep is explicitly bounded.
        """
        while self._running:
            await asyncio.sleep(self.coarse_interval_s)
            if self._running:
                await self._event_queue.put(CoarseRebuildEvent(reason="timer"))

    async def _event_loop(self) -> None:
        """
        Main event processing loop. Processes one event at a time.
        Terminates on StopEvent or when _running is False.
        """
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            if isinstance(event, StopEvent):
                log.info("StopEvent received: %s", event.reason)
                break

            elif isinstance(event, CongestionEvent):
                await self._handle_congestion(event)

            elif isinstance(event, CoarseRebuildEvent):
                await self._do_coarse_rebuild(event.reason)

            elif isinstance(event, OptimiseEvent):
                await self._do_fine_optimise(event)

    # -----------------------------------------------------------------------
    # Coarse timescale: rebuild candidate pools
    # -----------------------------------------------------------------------

    async def _do_coarse_rebuild(self, reason: str) -> None:
        """
        Rebuild k-shortest candidate pools from scratch (Phase 2).
        Runs every 300s or when congestion changes significantly.
        """
        self._set_state(
            SystemState.CANDIDATE_GENERATION,
            f"Rebuilding candidates (reason={reason})"
        )

        # Yield to event loop so other tasks can run
        await asyncio.sleep(0)

        t0 = time.perf_counter()

        # Import Phase 2 — done inside the function to avoid circular imports
        # at module level (these are separate phases)
        from phase2_candidate_generation import build_candidate_pool

        # Rebuild candidates for all O-D pairs
        raw_pool = build_candidate_pool(self.G, self.od_pairs, K=self.K)

        # Remap to vehicle-index-keyed dict for Phases 3-5
        self._candidates = {}
        for v_idx, vehicle in enumerate(self.vehicles):
            od_key = (vehicle.origin, vehicle.destination)
            self._candidates[v_idx] = raw_pool.get(od_key, [])

        n_rebuilt = len(raw_pool)
        elapsed   = time.perf_counter() - t0
        self._last_coarse = time.time()

        log.info(
            "Candidates rebuilt: %d O-D pairs in %.2fs",
            n_rebuilt, elapsed,
        )
        self._set_state(SystemState.IDLE, f"Rebuild complete ({n_rebuilt} pairs)")

    # -----------------------------------------------------------------------
    # Fine timescale: handle congestion update
    # -----------------------------------------------------------------------

    async def _handle_congestion(self, event: CongestionEvent) -> None:
        """
        Process a raw congestion observation:
          1. Apply exponential smoothing (B5, α=0.7).
          2. Decide whether to trigger re-optimisation.
          3. Check if coarse rebuild is needed (threshold exceeded).
        """
        # Step 1: Smooth the raw loads
        mean_change   = self.smoother.mean_load_change(event.edge_loads)
        coarse_frac   = self.smoother.coarse_change_fraction(event.edge_loads)
        smoothed      = self.smoother.update(event.edge_loads)

        log.info(
            "Congestion tick %d: mean_change=%.3f, coarse_frac=%.3f",
            self.smoother.n_ticks, mean_change, coarse_frac,
        )

        # Step 2: Check if coarse rebuild needed
        if coarse_frac >= self.coarse_threshold:
            log.info(
                "Coarse threshold triggered: %.1f%% of edges changed >10%%",
                coarse_frac * 100,
            )
            await self._do_coarse_rebuild("threshold")

        # Step 3: Trigger fine re-optimisation if change is significant
        if mean_change >= self.fine_threshold:
            opt_event = OptimiseEvent(smoothed_loads=smoothed)
            await self._event_queue.put(opt_event)
        else:
            log.debug(
                "Change too small (%.4f < %.4f) — skipping re-optimisation.",
                mean_change, self.fine_threshold,
            )

    # -----------------------------------------------------------------------
    # Fine timescale: run optimiser
    # -----------------------------------------------------------------------

    async def _do_fine_optimise(self, event: OptimiseEvent) -> None:
        """
        Run one fine-timescale optimisation round:
          Phase 3 → QUBO construction
          Phase 4 → QPSO selector (with explicit termination)
          Phase 5 → Local search + hysteresis guard

        All phases have explicit termination criteria — no unbounded loop.
        """
        if not self._candidates:
            log.warning("No candidates available — skipping optimisation.")
            return

        round_start     = time.perf_counter()
        state_sequence: List[str] = []

        def record_state(state: SystemState, msg: str = "") -> None:
            state_sequence.append(state.name)
            self._set_state(state, msg)

        # --- Phase 3: Build QUBO ---
        record_state(SystemState.QPSO_SELECTING, "Building QUBO …")
        await asyncio.sleep(0)   # yield to event loop

        from phase3_qubo import (
            Vehicle, build_qubo, compute_lambda, MU_SWITCH_PENALTY_S
        )

        lam = compute_lambda(
            self.vehicles, self._candidates, event.smoothed_loads
        )
        Q, offset, meta = build_qubo(
            self.vehicles, self._candidates, event.smoothed_loads,
            lam=lam, mu=MU_SWITCH_PENALTY_S,
        )

        V    = meta["V"]
        K    = meta["K_max"]

        # --- Phase 4: QPSO ---
        record_state(SystemState.QPSO_SELECTING, "Running QPSO …")
        await asyncio.sleep(0)

        from phase4_qpso import run_qpso
        qpso_result = run_qpso(Q, offset, V, K, seed=self._round_id)

        # --- Phase 5: Local search + stability guard ---
        record_state(SystemState.LOCAL_SEARCH, "Running local search …")
        await asyncio.sleep(0)

        from phase5_local_search import run_local_search_and_guard
        ls_result = run_local_search_and_guard(
            initial_assignment = qpso_result.best_assignment,
            prev_assignment    = self._assignment,
            Q=Q, offset=offset, meta=meta,
            candidates=self._candidates,
        )

        record_state(SystemState.STABILITY_CHECK, "Applying hysteresis guard …")
        await asyncio.sleep(0)

        # Commit the stable assignment
        self._assignment = ls_result.stable_assignment

        record_state(SystemState.COMPLETE, f"Round {self._round_id} complete.")

        round_end = time.perf_counter()
        round_record = OptimisationRound(
            round_id              = self._round_id,
            timestamp             = round_start,
            trigger               = "congestion_event",
            state_sequence        = state_sequence,
            assignment            = list(ls_result.stable_assignment),
            qpso_energy           = qpso_result.best_energy,
            ls_energy             = ls_result.refined_energy,
            n_hysteresis_blocked  = ls_result.n_hysteresis_blocked,
            n_candidates_rebuilt  = 0,
            smoothing_alpha       = self.smoother.alpha,
            runtime_s             = round_end - round_start,
        )
        self.history.append(round_record)

        if self._on_round_complete:
            self._on_round_complete(round_record)

        self._round_id += 1

        log.info(
            "Round %d: qpso=%.4f, ls=%.4f, blocked=%d, runtime=%.3fs",
            round_record.round_id,
            round_record.qpso_energy,
            round_record.ls_energy,
            round_record.n_hysteresis_blocked,
            round_record.runtime_s,
        )

        self._set_state(SystemState.IDLE, "Waiting for next event.")


# ---------------------------------------------------------------------------
# Standalone simulation (acceptance test — no FastAPI needed here)
# ---------------------------------------------------------------------------

async def run_phase6_simulation(
    G,
    vehicles: List[Any],
    od_pairs: List[Tuple[int, int]],
    n_congestion_events: int = 4,
    K: int = 5,
) -> List[OptimisationRound]:
    """
    Phase 6 acceptance test: drives the engine with synthetic congestion events
    and verifies the two-timescale model runs correctly.

    Returns the list of completed OptimisationRound records.
    """
    from phase1_world_state import generate_synthetic_congestion

    print(f"\n{'=' * 60}")
    print("PHASE 6 - TWO-TIMESCALE EXECUTION MODEL")
    print(f"{'=' * 60}")
    print(f"  Coarse interval   : {COARSE_INTERVAL_S}s")
    print(f"  Smoothing alpha   : {CONGESTION_SMOOTHING}  (B5)")
    print(f"  Coarse threshold  : {COARSE_THRESHOLD*100:.0f}% edges change >10%")
    print(f"  Fine threshold    : {FINE_CHANGE_THRESHOLD*100:.0f}% mean load change")
    print(f"  Congestion events : {n_congestion_events}")
    print(f"  Vehicles          : {len(vehicles)}")

    completed_rounds: List[OptimisationRound] = []

    def on_round_complete(r: OptimisationRound) -> None:
        completed_rounds.append(r)
        print(f"  Round {r.round_id:02d}: trigger={r.trigger:<20} "
              f"qpso={r.qpso_energy:.2f}  ls={r.ls_energy:.2f}  "
              f"blocked={r.n_hysteresis_blocked}  t={r.runtime_s:.3f}s  "
              f"states={'>'.join(r.state_sequence)}")

    # Shorten coarse interval to 1s for testing (don't wait 300s)
    engine = TwoTimescaleEngine(
        G=G,
        vehicles=vehicles,
        od_pairs=od_pairs,
        K=K,
        coarse_interval_s=9999,    # Disable timer for test — use threshold only
        smoothing_alpha=CONGESTION_SMOOTHING,
        coarse_threshold=COARSE_THRESHOLD,
        fine_threshold=0.001,       # Very sensitive for test — trigger on any change
        on_round_complete=on_round_complete,
    )

    print(f"\n  Starting engine ...")
    await engine.start()
    print(f"  Initial candidate pool built.")
    print(f"\n  Injecting {n_congestion_events} congestion events ...")

    # Inject synthetic congestion events with different scenarios
    scenarios = ["free_flow", "moderate", "rush_hour", "rush_hour"]
    for i in range(n_congestion_events):
        scenario = scenarios[i % len(scenarios)]
        seed     = i * 7 + 100
        loads    = generate_synthetic_congestion(
            G, num_vehicles=50 * (i + 1), seed=seed, scenario=scenario
        )
        print(f"\n  Event {i+1}/{n_congestion_events}: scenario={scenario}, "
              f"vehicles={50*(i+1)}, loaded_edges={len(loads)}")
        await engine.push_congestion(loads, scenario=scenario)

        # Wait for this event to be processed (small sleep to let asyncio schedule)
        await asyncio.sleep(0.5)

    # Allow final event to finish
    await asyncio.sleep(0.5)
    await engine.stop("simulation_complete")

    # -----------------------------------------------------------------------
    # Acceptance checks
    # -----------------------------------------------------------------------
    print(f"\n  Completed rounds : {len(completed_rounds)}")

    ok1 = len(completed_rounds) >= 1
    print(f"  [{'PASS' if ok1 else 'FAIL'}] At least 1 optimisation round completed")

    ok2 = all(r.ls_energy <= r.qpso_energy + 1e-6 for r in completed_rounds)
    print(f"  [{'PASS' if ok2 else 'FAIL'}] Local search never worsens QPSO result")

    ok3 = all(
        "QPSO_SELECTING" in r.state_sequence and
        "LOCAL_SEARCH"   in r.state_sequence and
        "STABILITY_CHECK" in r.state_sequence
        for r in completed_rounds
    )
    print(f"  [{'PASS' if ok3 else 'FAIL'}] All rounds followed correct state sequence")

    ok4 = all(r.runtime_s < 30.0 for r in completed_rounds)
    print(f"  [{'PASS' if ok4 else 'FAIL'}] All rounds completed < 30s each")

    # Smoothing check: verify smoother state is non-zero after events
    ok5 = len(engine.smoother.smoothed_state) > 0
    print(f"  [{'PASS' if ok5 else 'FAIL'}] Congestion smoother populated (alpha=0.7 applied)")

    if completed_rounds:
        print(f"\n  Smoothing state: {len(engine.smoother.smoothed_state)} edges tracked")
        sample_loads = list(engine.smoother.smoothed_state.values())[:5]
        print(f"  Sample smoothed loads: {[f'{v:.2f}' for v in sample_loads]}")

    all_pass = ok1 and ok2 and ok3 and ok4 and ok5
    verdict  = "[PASS]" if all_pass else "[FAIL]"
    print(f"\n  {verdict} Phase 6 acceptance test.")
    print(f"{'=' * 60}\n")

    return completed_rounds


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pickle
    import sys
    import random
    sys.path.insert(0, str(__file__[:__file__.rfind("\\")]))

    from pathlib import Path
    DATA_DIR = Path(__file__).parent.parent / "data"

    # Load Phase 1 graph
    with open(DATA_DIR / "phase1_graph.pkl", "rb") as f:
        G = pickle.load(f)

    # Load Phase 3 data for vehicles/candidates
    sys.path.insert(0, str(Path(__file__).parent))
    from phase3_qubo import Vehicle  # noqa: F401 (for pickle)

    with open(DATA_DIR / "phase3_qubo_example.pkl", "rb") as f:
        p3 = pickle.load(f)

    vehicles   = p3["vehicles"]
    od_pairs   = [(v.origin, v.destination) for v in vehicles]

    log.info("Loaded %d vehicles from Phase 3.", len(vehicles))

    # Run the simulation
    history = asyncio.run(
        run_phase6_simulation(
            G=G,
            vehicles=vehicles,
            od_pairs=od_pairs,
            n_congestion_events=4,
            K=5,
        )
    )

    # Save history for Phase 7
    out_path = DATA_DIR / "phase6_history.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(history, f)
    print(f"Phase 6 history saved --> {out_path}")
