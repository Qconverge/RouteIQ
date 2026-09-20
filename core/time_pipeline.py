import sys, time, pickle, random
sys.path.insert(0, 'core')
import networkx as nx

with open('data/phase1_graph.pkl', 'rb') as f:
    G = pickle.load(f)

rng = random.Random(42)
nodes = list(G.nodes())
od_pairs = []
while len(od_pairs) < 10:
    o, d = rng.sample(nodes, 2)
    if nx.has_path(G, o, d):
        od_pairs.append((o, d))

# Stage 1: Candidate Generation
t0 = time.perf_counter()
from phase2_candidate_generation import build_candidate_pool
pool = build_candidate_pool(G, od_pairs, K=5, max_workers=6)
t_candidates = time.perf_counter() - t0

# Stage 2: QUBO
from phase3_qubo import build_qubo, compute_lambda, Vehicle
from phase1_world_state import generate_synthetic_congestion, apply_loads_to_graph
vehicles = [Vehicle(vid=i, origin=o, destination=d) for i,(o,d) in enumerate(od_pairs)]
candidates = {i: pool.get((o,d),[]) for i,(o,d) in enumerate(od_pairs)}
cong = generate_synthetic_congestion(G, num_vehicles=10, seed=0)
apply_loads_to_graph(G, cong)
lam = compute_lambda(vehicles, candidates, cong)
t1 = time.perf_counter()
Q, offset, meta = build_qubo(vehicles, candidates, cong, lam=lam, mu=120.0)
t_qubo = time.perf_counter() - t1

# Stage 3: QPSO
from phase4_qpso import run_qpso
V, K = meta['V'], meta['K_max']
t2 = time.perf_counter()
result = run_qpso(Q, offset, V, K, seed=0)
t_qpso = time.perf_counter() - t2

# Stage 4: Local Search
from phase5_local_search import run_local_search_and_guard
t3 = time.perf_counter()
ls = run_local_search_and_guard(result.best_assignment, None, Q, offset, meta, candidates)
t_ls = time.perf_counter() - t3

total = t_candidates + t_qubo + t_qpso + t_ls
print("")
print("=== SERVER-SIDE TIMING (your laptop, 10 vehicles, 1000m radius) ===")
print(f"  Candidate Generation (Yen + Bidirectional A*) : {t_candidates:.3f}s")
print(f"  QUBO Matrix Build                             : {t_qubo:.4f}s")
print(f"  QPSO Optimizer (30 particles, 150 max iters)  : {t_qpso:.3f}s")
print(f"  Local Search + Hysteresis Guard               : {t_ls:.4f}s")
print(f"  -----------------------------------------------------------")
print(f"  TOTAL SERVER TIME                             : {total:.3f}s")
print("")
print(f"  Old UI showed only QPSO time : {t_qpso:.3f}s")
print(f"  New UI will show full total  : {total:.3f}s")
