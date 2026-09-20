# RouteIQ: Quantum-Inspired System-Optimal Routing

RouteIQ is a system-optimal multi-vehicle routing prototype built for SIH 2026. It applies a Quantum-behaved Particle Swarm Optimization (QPSO) hybrid algorithm to assign routes to vehicles in a way that minimizes total network congestion.

## System Limitations (Honesty & Positioning Pass)

Before evaluating this prototype, please note the following explicit constraints and limitations:

1. **Simulated Congestion (BPR)**: Congestion is simulated using the Bureau of Public Roads (BPR) function, calibrated to OSM road classes. It does **not** use a live traffic feed. The system's effectiveness is pending real-world sensor validation.
2. **No Global Optimality Guarantee**: The QPSO-hybrid selector is a metaheuristic. While it provides measured, empirical improvements over greedy baselines, it does not provide mathematical guarantees of finding the absolute global optimum.
3. **"Quantum-Inspired"**: The term refers strictly to a classical algorithm inspired by quantum-mechanical formalism (wave functions). It runs on standard classical hardware and provides **no quantum hardware speedup**. We do not use phrases like "quantum-level" or imply quantum computing capabilities.

## Architecture

The project is broken into a strict, reproducible pipeline:
- **Phase 1 (World State):** OSMnx downloads and caches free map data. Synthetic BPR congestion is generated.
- **Phase 2 (Candidates):** Yen's K-Shortest Paths (K=5) pre-computes valid routes.
- **Phase 3 (QUBO):** Converts travel times and congestion penalties into a Quadratic Unconstrained Binary Optimization matrix.
- **Phase 4 (QPSO):** Solves the QUBO using a Quantum-behaved Particle Swarm Optimizer.
- **Phase 5 (Local Search):** Refines QPSO output via candidate-swapping and an 8% hysteresis guard to prevent oscillation.
- **Phase 6 (Execution Engine):** Two-timescale asyncio event loop handling coarse (candidate) and fine (optimisation) updates.
- **Phase 7 (API):** FastAPI server exposing REST and WebSocket endpoints.
- **Phase 8 (Benchmarks):** Rigorous statistical evaluation across multiple random seeds.
- **Phase 9 (Dashboard):** Plain HTML/JS + MapLibre GL frontend.

## Usage

Start the backend and frontend server:
```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```
Then navigate to `http://127.0.0.1:8000/` in your browser.

Run the Phase 8 benchmark suite:
```bash
python core/phase8_benchmark.py
```
