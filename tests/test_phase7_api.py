"""
Phase 7 — Acceptance Test Script
Runs all endpoint checks without needing a browser.
Run AFTER uvicorn is started in a separate process.
"""
import json
import sys
import time

import httpx

BASE = "http://127.0.0.1:8000"
OK   = "[PASS]"
FAIL = "[FAIL]"

def check(label: str, cond: bool) -> bool:
    print(f"  {OK if cond else FAIL} {label}")
    return cond

results = []

sep = "=" * 60
print(f"\n{sep}\nPHASE 7 - ACCEPTANCE TEST\n{sep}")

# Wait for server to be ready
for attempt in range(15):
    try:
        r = httpx.get(f"{BASE}/health", timeout=3)
        if r.status_code == 200:
            break
    except Exception:
        pass
    print(f"  Waiting for server ... attempt {attempt+1}/15")
    time.sleep(2)
else:
    print("  Server did not start. Aborting.")
    sys.exit(1)

# --- Health ---
r = httpx.get(f"{BASE}/health")
results.append(check("GET /health returns 200", r.status_code == 200))
results.append(check("Health note mentions BPR simulation", "SIMULATED" in r.json().get("note", "")))

# --- Create scenario ---
payload = {
    "name": "Phase7Test",
    "area_name": "Navrangpura, Ahmedabad",
    "center_lat": 23.0285,
    "center_lon": 72.5546,
    "radius_m": 1000,
    "num_vehicles": 5,
    "k_candidates": 3,
}
r = httpx.post(f"{BASE}/scenarios", json=payload, timeout=60)
results.append(check("POST /scenarios returns 201", r.status_code == 201))
scenario = r.json()
sid = scenario["id"]
results.append(check("Scenario has id", "id" in scenario and sid is not None))
results.append(check("Scenario has od_pairs", len(json.loads(scenario["od_pairs_json"])) > 0))
print(f"    Created scenario id={sid}, od_pairs={len(json.loads(scenario['od_pairs_json']))}")

# --- List scenarios ---
r = httpx.get(f"{BASE}/scenarios")
results.append(check("GET /scenarios returns list", r.status_code == 200 and isinstance(r.json(), list)))

# --- Get scenario ---
r = httpx.get(f"{BASE}/scenarios/{sid}")
results.append(check(f"GET /scenarios/{sid} returns 200", r.status_code == 200))

# --- Simulate traffic ---
r = httpx.post(f"{BASE}/scenarios/{sid}/simulate-traffic",
               json={"scenario_type": "moderate", "num_vehicles": 50, "seed": 1},
               timeout=30)
results.append(check("POST /scenarios/{id}/simulate-traffic returns 200", r.status_code == 200))
sim_data = r.json()
results.append(check("Simulation note mentions BPR", "SIMULATED" in sim_data.get("note", "")))
results.append(check("Simulation reports loaded_edges > 0", sim_data.get("loaded_edges", 0) > 0))
print(f"    Simulated traffic: loaded_edges={sim_data.get('loaded_edges')}, max_vph={sim_data.get('max_load_vph')}")

# --- Trigger optimisation ---
r = httpx.post(f"{BASE}/scenarios/{sid}/optimize",
               json={"seed": 0, "n_vehicles_congestion": 50},
               timeout=10)
results.append(check("POST /scenarios/{id}/optimize returns 200", r.status_code == 200))
opt_data = r.json()
results.append(check("Optimize response has ws_url", "ws_url" in opt_data))
print(f"    Optimise triggered: status={opt_data.get('status')}, ws_url={opt_data.get('ws_url')}")

# Wait for optimisation to complete
print(f"    Waiting for background optimisation to finish ...")
for i in range(30):
    time.sleep(2)
    r2 = httpx.get(f"{BASE}/scenarios/{sid}")
    st = r2.json().get("status")
    print(f"      status={st}")
    if st in ("complete", "error"):
        break

r = httpx.get(f"{BASE}/scenarios/{sid}")
final = r.json()
results.append(check("Scenario reaches 'complete' status", final.get("status") == "complete"))
results.append(check("Scenario has assignment_json", len(json.loads(final.get("assignment_json", "[]"))) > 0))
results.append(check("Scenario has metrics_json", "total_travel_time_s" in json.loads(final.get("metrics_json", "{}"))))
metrics = json.loads(final.get("metrics_json", "{}"))
print(f"    Metrics: total_time={metrics.get('total_travel_time_s')}s, "
      f"congestion_cost={metrics.get('congestion_cost')}")

# --- Benchmark ---
r = httpx.get(f"{BASE}/scenarios/{sid}/benchmark",
              params={"seed": 0, "n_vehicles_congestion": 50},
              timeout=120)
results.append(check("GET /scenarios/{id}/benchmark returns 200", r.status_code == 200))
bench = r.json()
results.append(check("Benchmark has 3 algorithms", len(bench.get("results", [])) == 3))
results.append(check("Benchmark note mentions BPR", "SIMULATED" in bench.get("note", "")))
algos = [x["algorithm"] for x in bench.get("results", [])]
results.append(check("Benchmark includes dijkstra", any("dijkstra" in a for a in algos)))
results.append(check("Benchmark includes qpso_hybrid", "qpso_hybrid" in algos))
print(f"    Benchmark results (sorted by total_travel_time_s):")
for r_item in bench.get("results", []):
    print(f"      {r_item['algorithm']:25s}  time={r_item['total_travel_time_s']:8.2f}s  "
          f"congestion_cost={r_item['congestion_cost']:.1f}  rt={r_item['runtime_s']:.3f}s")

# --- Delete scenario ---
r = httpx.delete(f"{BASE}/scenarios/{sid}")
results.append(check("DELETE /scenarios/{id} returns 200", r.status_code == 200))

# --- 404 after delete ---
r = httpx.get(f"{BASE}/scenarios/{sid}")
results.append(check("GET /scenarios/{id} after delete returns 404", r.status_code == 404))

# --- Summary ---
passed = sum(results)
total  = len(results)
print(f"\n  {sep}")
print(f"  Results: {passed}/{total} checks passed.")
verdict = "[PASS]" if passed == total else f"[FAIL] ({total - passed} failed)"
print(f"  {verdict} Phase 7 acceptance test.")
print(f"  {sep}\n")

sys.exit(0 if passed == total else 1)
