// Dynamically detect base URL (works locally, on Render, Codespaces, or any host)
const API_BASE = window.location.origin;
const WS_BASE = (window.location.protocol === "https:" ? "wss://" : "ws://") + window.location.host;

let currentScenarioId = null;
let map = null;
let ws = null;
let convergenceChart = null;
let benchmarkChart = null;
let benchmarkCongChart = null;

// Initialize MapLibre
function initMap() {
    map = new maplibregl.Map({
        container: 'map',
        style: 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json', // Free raster/vector hybrid style
        center: [72.5546, 23.0285],
        zoom: 14
    });
    
    map.addControl(new maplibregl.NavigationControl());
}

// Nominatim Search with Throttling
let searchTimeout;
const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('nominatim-results');

searchInput.addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    const q = e.target.value;
    if (q.length < 3) {
        searchResults.style.display = 'none';
        return;
    }
    
    searchTimeout = setTimeout(async () => {
        try {
            // Respect Nominatim Usage Policy (User-Agent header is set via browser, but we provide email/app info in query)
            const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(q)}&limit=5&app=RouteIQ_SIH2026`);
            const data = await res.json();
            
            searchResults.innerHTML = '';
            if (data.length > 0) {
                data.forEach(item => {
                    const div = document.createElement('div');
                    div.className = 'search-result';
                    div.innerText = item.display_name;
                    div.onclick = () => {
                        document.getElementById('lat').value = item.lat;
                        document.getElementById('lon').value = item.lon;
                        map.flyTo({ center: [item.lon, item.lat], zoom: 14 });
                        searchResults.style.display = 'none';
                        searchInput.value = item.display_name;
                    };
                    searchResults.appendChild(div);
                });
                searchResults.style.display = 'block';
            }
        } catch (err) {
            console.error("Nominatim error", err);
        }
    }, 1000); // 1s throttle
});

document.addEventListener('click', (e) => {
    if (e.target !== searchInput && e.target !== searchResults) {
        searchResults.style.display = 'none';
    }
});

// Create Scenario
document.getElementById('btn-create').addEventListener('click', async () => {
    const btn = document.getElementById('btn-create');
    btn.disabled = true;
    btn.innerHTML = '<span class="loader"></span> Creating...';
    
    const payload = {
        name: "UI Scenario",
        area_name: searchInput.value || "Custom Location",
        center_lat: parseFloat(document.getElementById('lat').value),
        center_lon: parseFloat(document.getElementById('lon').value),
        radius_m: parseFloat(document.getElementById('radius').value),
        num_vehicles: parseInt(document.getElementById('num-veh').value),
        k_candidates: 5
    };
    
    try {
        const res = await fetch(`${API_BASE}/scenarios`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error (${res.status})`);
        }
        const data = await res.json();
        currentScenarioId = data.id;
        
        const info = document.getElementById('scenario-info');
        info.innerHTML = `Scenario <b>#${data.id}</b> created.<br>Vehicles: ${payload.num_vehicles}`;
        info.style.display = 'block';
        
        document.getElementById('card-optimizer').classList.remove('hidden');
        document.getElementById('card-benchmarks').classList.remove('hidden');
        
        // Clear old map layers
        if (map.getSource('routes')) {
            map.removeLayer('routes-line');
            map.removeSource('routes');
        }
        
    } catch (err) {
        console.error("Scenario creation error:", err);
        alert(`Failed to create scenario: ${err.message || err}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Create Scenario';
    }
});

// Simulate Traffic
document.getElementById('btn-traffic').addEventListener('click', async () => {
    if (!currentScenarioId) return;
    const btn = document.getElementById('btn-traffic');
    btn.disabled = true;
    
    const type = document.getElementById('traffic-type').value;
    const num_veh = parseInt(document.getElementById('num-veh').value);
    
    try {
        const res = await fetch(`${API_BASE}/scenarios/${currentScenarioId}/simulate-traffic`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ scenario_type: type, num_vehicles: num_veh, seed: Date.now() % 1000 })
        });
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error (${res.status})`);
        }
        const data = await res.json();
        alert(`Injected ${data.congestion_type} congestion (BPR simulated) on ${data.loaded_edges} edges.`);
    } catch(err) {
        console.error("Traffic injection error:", err);
        alert(`Failed to inject traffic: ${err.message || err}`);
    } finally {
        btn.disabled = false;
    }
});

// Run Optimizer (WebSocket)
const stages = ['loading_graph', 'candidate_generation', 'qpso_selecting', 'local_search', 'stability_check', 'complete'];

function resetStages() {
    stages.forEach(s => {
        const el = document.getElementById(`stage-${s}`);
        el.className = '';
        el.querySelector('span').innerText = '○';
    });
    document.getElementById('ws-msg').innerText = '';
}

function updateStage(stage, msg) {
    let passed = true;
    stages.forEach(s => {
        const el = document.getElementById(`stage-${s}`);
        if (s === stage) {
            el.className = 'active';
            el.querySelector('span').innerHTML = '<span class="loader"></span>';
            passed = false;
        } else if (passed) {
            el.className = 'done';
            el.querySelector('span').innerText = '✓';
        } else {
            el.className = '';
            el.querySelector('span').innerText = '○';
        }
    });
    if (stage === 'complete') {
        document.getElementById(`stage-complete`).className = 'done';
        document.getElementById(`stage-complete`).querySelector('span').innerText = '✓';
    }
    
    if (msg) document.getElementById('ws-msg').innerText = msg;
}

document.getElementById('btn-optimize').addEventListener('click', async () => {
    if (!currentScenarioId) return;
    const btn = document.getElementById('btn-optimize');
    btn.disabled = true;
    document.getElementById('progress-container').classList.remove('hidden');
    document.getElementById('card-results').classList.remove('hidden');
    resetStages();
    
    try {
        const num_veh = parseInt(document.getElementById('num-veh').value);
        const res = await fetch(`${API_BASE}/scenarios/${currentScenarioId}/optimize`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ seed: 42, n_vehicles_congestion: num_veh })
        });
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error (${res.status})`);
        }
        
        const data = await res.json();
        if (data.ws_url) {
            connectWebSocket(data.ws_url);
        }
    } catch(err) {
        console.error("Optimizer error:", err);
        alert(`Failed to start optimizer: ${err.message || err}`);
        btn.disabled = false;
    }
});

function connectWebSocket(path) {
    if (ws) ws.close();
    ws = new WebSocket(`${WS_BASE}${path}`);
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.stage === 'ping') return;
        
        updateStage(data.stage, data.msg);
        
        // Handle Convergence Data
        if (data.stage === 'qpso_selecting' && data.convergence) {
            drawConvergenceChart(data.convergence);
        }
        
        // Handle Complete
        if (data.stage === 'complete') {
            document.getElementById('btn-optimize').disabled = false;
            updateKPIs(data.metrics);
            drawRoutesOnMap();
        }
    };
    
    ws.onerror = (err) => {
        console.error("WS error", err);
        document.getElementById('btn-optimize').disabled = false;
    };
}

// Update KPIs
function updateKPIs(metrics) {
    document.getElementById('kpi-time').innerText = metrics.total_travel_time_s.toFixed(1);
    document.getElementById('kpi-dist').innerText = metrics.total_distance_m.toFixed(1);
    document.getElementById('kpi-cost').innerText = metrics.congestion_cost.toFixed(1);
    document.getElementById('kpi-runtime').innerText = metrics.runtime_s.toFixed(3);
}

// Fetch and Draw Routes
async function drawRoutesOnMap() {
    try {
        const res = await fetch(`${API_BASE}/scenarios/${currentScenarioId}/routes`);
        const geojson = await res.json();
        
        if (map.getSource('routes')) {
            map.getSource('routes').setData(geojson);
        } else {
            map.addSource('routes', {
                type: 'geojson',
                data: geojson
            });
            map.addLayer({
                id: 'routes-line',
                type: 'line',
                source: 'routes',
                layout: {
                    'line-join': 'round',
                    'line-cap': 'round'
                },
                paint: {
                    'line-color': ['get', 'color'],
                    'line-width': 4,
                    'line-opacity': 0.8
                }
            });
        }
        
        // Fit bounds
        const coordinates = geojson.features.flatMap(f => f.geometry.coordinates);
        if (coordinates.length > 0) {
            const bounds = coordinates.reduce((b, coord) => {
                return b.extend(coord);
            }, new maplibregl.LngLatBounds(coordinates[0], coordinates[0]));
            map.fitBounds(bounds, { padding: 50 });
        }
        
    } catch (err) {
        console.error("Failed to load routes", err);
    }
}

// Chart.js - Convergence Curve
function drawConvergenceChart(convergenceData) {
    const ctx = document.getElementById('chart-convergence').getContext('2d');
    
    if (convergenceChart) convergenceChart.destroy();
    
    convergenceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: convergenceData.length}, (_, i) => i+1),
            datasets: [{
                label: 'Best Energy (QUBO)',
                data: convergenceData,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { title: { display: true, text: 'Energy' } },
                x: { title: { display: true, text: 'Iteration' } }
            }
        }
    });
}

// Benchmark
document.getElementById('btn-benchmark').addEventListener('click', async () => {
    if (!currentScenarioId) return;
    const btn = document.getElementById('btn-benchmark');
    btn.disabled = true;
    btn.innerHTML = '<span class="loader"></span> Running Benchmarks...';
    
    try {
        const num_veh = parseInt(document.getElementById('num-veh').value);
        const res = await fetch(`${API_BASE}/scenarios/${currentScenarioId}/benchmark?seed=42&n_vehicles_congestion=${num_veh}`);
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error (${res.status})`);
        }
        const data = await res.json();
        
        document.getElementById('benchmark-chart-container').classList.remove('hidden');
        drawBenchmarkChart(data.results);
        
    } catch(err) {
        console.error("Benchmark error:", err);
        alert(`Failed to run benchmark: ${err.message || err}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Run Benchmark Suite';
    }
});

function drawBenchmarkChart(results) {
    // ── Per-algorithm colors (consistent across both charts) ─────────────────
    const ALGO = {
        'dijkstra_all_shortest': { label: 'Dijkstra',    color: '#3b82f6', border: '#1d4ed8' },
        'k_shortest_greedy':     { label: 'Greedy',      color: '#f59e0b', border: '#b45309' },
        'qpso_hybrid':           { label: 'QPSO Hybrid', color: '#10b981', border: '#047857' },
    };

    const DEFAULT = { label: 'Unknown', color: '#6b7280', border: '#374151' };

    const labels      = results.map(r => (ALGO[r.algorithm] || DEFAULT).label);
    const bgColors    = results.map(r => (ALGO[r.algorithm] || DEFAULT).color);
    const borderColors= results.map(r => (ALGO[r.algorithm] || DEFAULT).border);

    // Show HTML legend + notes
    document.getElementById('benchmark-legend').classList.remove('hidden');
    document.getElementById('benchmark-legend').classList.add('flex');

    // Shared chart options (no legend inside chart — we use HTML legend above)
    function makeOptions(yLabel) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },   // hidden — HTML legend is above
                tooltip: {
                    callbacks: {
                        title: (items) => (ALGO[results[items[0].dataIndex].algorithm] || DEFAULT).label,
                        afterBody: (items) => {
                            const r = results[items[0].dataIndex];
                            return [`Runtime: ${r.runtime_s.toFixed(3)}s`, `Vehicles: ${r.n_vehicles}`];
                        }
                    }
                }
            },
            scales: {
                x: { ticks: { font: { size: 12, weight: 'bold' } } },
                y: {
                    beginAtZero: true,
                    title: { display: true, text: yLabel, font: { size: 11 } }
                }
            }
        };
    }

    // ── Chart 1: Travel Time ──────────────────────────────────────────────────
    if (benchmarkChart) benchmarkChart.destroy();
    const ctx1 = document.getElementById('chart-benchmark').getContext('2d');
    benchmarkChart = new Chart(ctx1, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Travel Time (s)',
                data: results.map(r => r.total_travel_time_s),
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 2,
                borderRadius: 4,
            }]
        },
        options: makeOptions('Travel Time (s)')
    });

    // ── Chart 2: Congestion Cost ──────────────────────────────────────────────
    if (benchmarkCongChart) benchmarkCongChart.destroy();
    const ctx2 = document.getElementById('chart-benchmark-cong').getContext('2d');
    benchmarkCongChart = new Chart(ctx2, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Congestion Cost',
                data: results.map(r => r.congestion_cost),
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 2,
                borderRadius: 4,
            }]
        },
        options: makeOptions('Congestion Cost')
    });
}


// Init map on load
initMap();
