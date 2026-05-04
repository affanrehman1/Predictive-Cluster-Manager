// ===================================================================
// Predictive Cluster Manager — Dashboard Logic
// ===================================================================
const API_URL = 'http://127.0.0.1:8000';
const MAX_DATA_POINTS = 40;

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $ = id => document.getElementById(id);
const elStatus = $('apiStatus');
const elStatusText = elStatus.querySelector('span');
const elCpu = $('cpuValue');
const elMem = $('memValue');
const elNodes = $('nodesValue');
const elCost = $('costValue');
const elCpuBar = $('cpuBar');
const elMemBar = $('memBar');
const elNodeBar = $('nodeBar');
const elNodeCountText = $('nodeCountText');
const elNodesGrid = $('nodesGrid');
const elFeed = $('activityFeed');
const elAstarPath = $('astarPath');
const elCpuSlider = $('cpuSlider');
const elMemSlider = $('memSlider');
const elCpuSliderVal = $('cpuSliderVal');
const elMemSliderVal = $('memSliderVal');
const elCpuPred = $('cpuPredBadge');
const elMemPred = $('memPredBadge');
const elNodesSub = $('nodesSub');

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let activeNodesMap = new Map(); // name -> { el, bootTime }
let totalBootCost = 0;
let totalShutdownCost = 0;
let lastPredCpu = null;
let lastPredMem = null;
let lastRequiredNodes = null;
let isConnected = false;

// ---------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------
function updateClock() {
    const now = new Date();
    $('liveClock').textContent = now.toLocaleTimeString('en-US', { hour12: false });
}
setInterval(updateClock, 1000);
updateClock();

// ---------------------------------------------------------------------------
// Slider updates
// ---------------------------------------------------------------------------
elCpuSlider.addEventListener('input', () => {
    elCpuSliderVal.textContent = elCpuSlider.value + '%';
});
elMemSlider.addEventListener('input', () => {
    elMemSliderVal.textContent = elMemSlider.value + '%';
});

// ---------------------------------------------------------------------------
// Chart.js setup
// ---------------------------------------------------------------------------
const ctx = $('workloadChart').getContext('2d');
const gCpu = ctx.createLinearGradient(0, 0, 0, 350);
gCpu.addColorStop(0, 'rgba(76,123,244,0.25)');
gCpu.addColorStop(1, 'rgba(76,123,244,0.0)');
const gMem = ctx.createLinearGradient(0, 0, 0, 350);
gMem.addColorStop(0, 'rgba(155,123,244,0.25)');
gMem.addColorStop(1, 'rgba(155,123,244,0.0)');
const gPredCpu = ctx.createLinearGradient(0, 0, 0, 350);
gPredCpu.addColorStop(0, 'rgba(76,123,244,0.06)');
gPredCpu.addColorStop(1, 'rgba(76,123,244,0.0)');
const gPredMem = ctx.createLinearGradient(0, 0, 0, 350);
gPredMem.addColorStop(0, 'rgba(155,123,244,0.06)');
gPredMem.addColorStop(1, 'rgba(155,123,244,0.0)');

const emptyArr = () => Array(MAX_DATA_POINTS).fill(null);

const chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: Array(MAX_DATA_POINTS).fill(''),
        datasets: [
            { label: 'CPU %', borderColor: '#4C7BF4', backgroundColor: gCpu, borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, fill: true, tension: 0.4, data: emptyArr() },
            { label: 'Memory %', borderColor: '#9B7BF4', backgroundColor: gMem, borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, fill: true, tension: 0.4, data: emptyArr() },
            { label: 'CPU Predicted', borderColor: 'rgba(76,123,244,0.35)', backgroundColor: gPredCpu, borderWidth: 1.5, borderDash: [5, 3], pointRadius: 0, fill: true, tension: 0.4, data: emptyArr() },
            { label: 'Mem Predicted', borderColor: 'rgba(155,123,244,0.35)', backgroundColor: gPredMem, borderWidth: 1.5, borderDash: [5, 3], pointRadius: 0, fill: true, tension: 0.4, data: emptyArr() },
        ]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        animation: { duration: 300 },
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: { labels: { color: '#8b90a8', font: { family: 'Inter', size: 11 }, usePointStyle: true, pointStyle: 'line' } },
            tooltip: { backgroundColor: 'rgba(26,29,46,0.95)', titleColor: '#e8eaf0', bodyColor: '#8b90a8', borderColor: 'rgba(100,120,180,0.2)', borderWidth: 1 }
        },
        scales: {
            x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { display: false } },
            y: { min: 0, max: 120, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#5c6080', font: { family: 'JetBrains Mono', size: 10 }, callback: v => v + '%' } }
        }
    }
});

function pushChartData(cpu, mem, predCpu, predMem) {
    const push = (ds, val) => { ds.push(val); ds.shift(); };
    push(chart.data.datasets[0].data, cpu);
    push(chart.data.datasets[1].data, mem);
    push(chart.data.datasets[2].data, predCpu);
    push(chart.data.datasets[3].data, predMem);
    chart.data.labels.push('');
    chart.data.labels.shift();
    chart.update('none');
}

// ---------------------------------------------------------------------------
// Color coding for metric values
// ---------------------------------------------------------------------------
function colorClass(pct) {
    if (pct < 40) return 'low';
    if (pct < 70) return 'medium';
    return 'high';
}

// ---------------------------------------------------------------------------
// Activity Feed
// ---------------------------------------------------------------------------
function addFeed(msg, type = 'info') {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    const entry = document.createElement('div');
    entry.className = `feed-entry ${type}`;
    entry.innerHTML = `<span class="feed-time">[${time}]</span> ${msg}`;
    elFeed.appendChild(entry);
    // Keep max 80 entries
    while (elFeed.children.length > 80) elFeed.removeChild(elFeed.firstChild);
    elFeed.scrollTop = elFeed.scrollHeight;
}

// ---------------------------------------------------------------------------
// A* Plan Visualizer
// ---------------------------------------------------------------------------
function renderAstarPlan(currentNodes, goalNodes, actions, totalCostVal) {
    elAstarPath.innerHTML = '';
    if (!actions || actions.length === 0) {
        elAstarPath.innerHTML = '<div class="astar-placeholder">Cluster at optimal size — no scaling needed</div>';
        return;
    }
    const isScaleUp = goalNodes > currentNodes;
    let nodeCount = currentNodes;

    // Start node
    const startEl = createAstarNode(nodeCount, 'current', 0);
    elAstarPath.appendChild(startEl);

    let runningCost = 0;
    actions.forEach((action, i) => {
        // Arrow
        const arrow = document.createElement('span');
        arrow.className = 'astar-arrow';
        arrow.textContent = '→';
        elAstarPath.appendChild(arrow);

        const isBoot = action.startsWith('Boot');
        nodeCount += isBoot ? 1 : -1;
        runningCost += isBoot ? 1.0 : 0.5;

        const el = createAstarNode(nodeCount, isBoot ? 'boot' : 'shutdown', runningCost);
        el.style.animationDelay = `${i * 0.1}s`;
        elAstarPath.appendChild(el);
    });
}

function createAstarNode(count, type, cost) {
    const el = document.createElement('div');
    el.className = `astar-node ${type}`;
    el.innerHTML = `${count}<div class="astar-node-cost">${cost.toFixed(1)}</div>`;
    return el;
}

// ---------------------------------------------------------------------------
// Pipeline Animation
// ---------------------------------------------------------------------------
function animatePipeline(stages) {
    // stages: ['metrics','lstm','astar','docker']
    const ids = ['stage-metrics', 'stage-lstm', 'stage-astar', 'stage-docker'];
    const particles = ['particle1', 'particle2', 'particle3'];
    const details = ['stageMetricsDetail', 'stageLstmDetail', 'stageAstarDetail', 'stageDockerDetail'];

    // Reset
    ids.forEach(id => $(id).classList.remove('active'));
    particles.forEach(id => $(id).classList.remove('animating'));

    let delay = 0;
    stages.forEach((info, i) => {
        setTimeout(() => {
            $(ids[i]).classList.add('active');
            $(details[i]).textContent = info;
            if (i > 0) {
                $(particles[i - 1]).classList.remove('animating');
                void $(particles[i - 1]).offsetWidth; // reflow
                $(particles[i - 1]).classList.add('animating');
            }
        }, delay);
        delay += 400;
    });

    // Clear active after animation
    setTimeout(() => {
        ids.forEach(id => $(id).classList.remove('active'));
    }, delay + 1500);
}

// ---------------------------------------------------------------------------
// Node Grid with Uptime
// ---------------------------------------------------------------------------
function updateNodesGrid(nodesFromApi) {
    const currentIds = new Set(nodesFromApi.map(n => n.name));
    // Remove deleted
    for (let [id, data] of activeNodesMap.entries()) {
        if (!currentIds.has(id)) {
            data.el.classList.add('removing');
            setTimeout(() => data.el.remove(), 300);
            activeNodesMap.delete(id);
        }
    }
    // Add new
    nodesFromApi.forEach(node => {
        if (!activeNodesMap.has(node.name)) {
            const card = document.createElement('div');
            card.className = 'node-card';
            const numMatch = node.name.match(/\d+$/);
            const displayNum = numMatch ? numMatch[0].padStart(2, '0') : '??';
            card.innerHTML = `
                <div class="node-card-header">
                    <span class="node-name">Node #${displayNum}</span>
                    <div class="node-status-dot"></div>
                </div>
                <div class="node-id-text">${node.container_id}</div>
                <div class="node-uptime" data-boot="${Date.now()}">⏱ 0s</div>
            `;
            elNodesGrid.appendChild(card);
            activeNodesMap.set(node.name, { el: card, bootTime: Date.now() });
        }
    });
    elNodeCountText.textContent = activeNodesMap.size;
}

// Update uptimes every second
setInterval(() => {
    for (let [, data] of activeNodesMap) {
        const uptimeEl = data.el.querySelector('.node-uptime');
        if (uptimeEl) {
            const secs = Math.floor((Date.now() - data.bootTime) / 1000);
            if (secs < 60) uptimeEl.textContent = `⏱ ${secs}s`;
            else if (secs < 3600) uptimeEl.textContent = `⏱ ${Math.floor(secs / 60)}m ${secs % 60}s`;
            else uptimeEl.textContent = `⏱ ${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
        }
    }
}, 1000);

// ---------------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------------
async function fetchStatus() {
    try {
        const res = await fetch(`${API_URL}/status`);
        if (!res.ok) throw new Error('API error');
        const data = await res.json();

        if (!isConnected) {
            isConnected = true;
            addFeed('Connected to API backend', 'success');
        }
        elStatus.className = 'status-badge';
        elStatusText.textContent = 'Connected';

        const cpuPct = (data.current_cpu * 100).toFixed(1);
        const memPct = (data.current_memory * 100).toFixed(1);

        elCpu.textContent = `${cpuPct}%`;
        elCpu.className = `metric-value ${colorClass(parseFloat(cpuPct))}`;
        elCpuBar.style.width = `${Math.min(parseFloat(cpuPct), 100)}%`;

        elMem.textContent = `${memPct}%`;
        elMem.className = `metric-value ${colorClass(parseFloat(memPct))}`;
        elMemBar.style.width = `${Math.min(parseFloat(memPct), 100)}%`;

        elNodes.textContent = data.running_nodes;
        elNodes.className = `metric-value ${data.running_nodes > 5 ? 'medium' : 'low'}`;
        elNodeBar.style.width = `${Math.min((data.running_nodes / 15) * 100, 100)}%`;
        elNodesSub.textContent = lastRequiredNodes !== null ? `Target: ${lastRequiredNodes} nodes` : `${data.running_nodes} active`;

        const predCpuPct = lastPredCpu !== null ? (lastPredCpu * 100).toFixed(1) : null;
        const predMemPct = lastPredMem !== null ? (lastPredMem * 100).toFixed(1) : null;
        pushChartData(parseFloat(cpuPct), parseFloat(memPct), predCpuPct ? parseFloat(predCpuPct) : null, predMemPct ? parseFloat(predMemPct) : null);

        updateNodesGrid(data.node_details);
    } catch (e) {
        if (isConnected) {
            addFeed('Lost connection to API', 'error');
            isConnected = false;
        }
        elStatus.className = 'status-badge disconnected';
        elStatusText.textContent = 'Disconnected';
    }
}

// ---------------------------------------------------------------------------
// API actions
// ---------------------------------------------------------------------------
function setLoading(btnId, spinId, loading) {
    $(btnId).disabled = loading;
    $(spinId).style.display = loading ? 'inline-block' : 'none';
}

function processScaleResponse(data, source) {
    lastPredCpu = data.predicted_cpu;
    lastPredMem = data.predicted_memory;
    lastRequiredNodes = data.required_nodes;

    elCpuPred.textContent = `AI Pred: ${(data.predicted_cpu * 100).toFixed(1)}%`;
    elMemPred.textContent = `AI Pred: ${(data.predicted_memory * 100).toFixed(1)}%`;

    // Cost tracking
    const bootActions = data.execution_results.filter(r => r.action === 'boot' && r.success).length;
    const shutActions = data.execution_results.filter(r => r.action === 'shutdown' && r.success).length;
    totalBootCost += bootActions * 1.0;
    totalShutdownCost += shutActions * 0.5;
    const totalCost = totalBootCost + totalShutdownCost;
    elCost.textContent = totalCost.toFixed(1);
    elCost.className = `metric-value ${totalCost > 10 ? 'medium' : 'low'}`;

    // Cost bars
    if (totalCost > 0) {
        $('costBarBoot').style.width = `${(totalBootCost / totalCost) * 100}%`;
        $('costBarShutdown').style.width = `${(totalShutdownCost / totalCost) * 100}%`;
    }

    // A* visualization
    renderAstarPlan(data.current_nodes, data.required_nodes, data.scaling_plan, data.total_cost);

    // Pipeline animation
    const cpuPct = (data.predicted_cpu * 100).toFixed(0);
    animatePipeline([
        `CPU: ${cpuPct}%`,
        `→ ${data.required_nodes} nodes`,
        `Cost: ${data.total_cost.toFixed(1)}`,
        `${data.scaling_direction}`
    ]);

    // Activity feed
    addFeed(`${source}: CPU=${(data.predicted_cpu * 100).toFixed(1)}%, Mem=${(data.predicted_memory * 100).toFixed(1)}%`, 'ai');
    if (data.scaling_direction !== 'STABLE') {
        addFeed(`A* Plan: ${data.scaling_direction} ${data.current_nodes}→${data.required_nodes} nodes (cost: ${data.total_cost.toFixed(1)})`, 'warning');
    }
    data.execution_results.forEach(r => {
        const type = r.success ? 'success' : 'error';
        addFeed(`${r.action.toUpperCase()} ${r.node_name} — ${r.message} (${r.duration_ms.toFixed(0)}ms)`, type);
    });

    setTimeout(fetchStatus, 300);
}

async function triggerFromSliders() {
    setLoading('btnSlider', 'spinSlider', true);
    try {
        const cpuSpike = parseInt(elCpuSlider.value) / 100;
        const memSpike = parseInt(elMemSlider.value) / 100;
        addFeed(`Stress test triggered: CPU=${(cpuSpike * 100).toFixed(0)}%, Mem=${(memSpike * 100).toFixed(0)}%`, 'warning');
        const res = await fetch(`${API_URL}/simulate-stress`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cpu_spike: cpuSpike, memory_spike: memSpike, headroom: 1.2 })
        });
        const data = await res.json();
        processScaleResponse(data, 'Stress Test');
    } catch (e) { addFeed(`Stress test failed: ${e.message}`, 'error'); }
    finally { setLoading('btnSlider', 'spinSlider', false); }
}

async function triggerPredict() {
    setLoading('btnPredict', 'spinPredict', true);
    try {
        const cpuVal = parseInt(elCpuSlider.value) / 100;
        const memVal = parseInt(elMemSlider.value) / 100;
        addFeed(`Running LSTM prediction pipeline (starting from CPU=${(cpuVal * 100).toFixed(0)}%)...`, 'ai');
        const res = await fetch(`${API_URL}/predict-and-scale`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                avg_cpu: cpuVal,
                avg_memory: memVal,
                headroom: 1.2
            })
        });
        const data = await res.json();
        processScaleResponse(data, 'LSTM Prediction');
    } catch (e) { addFeed(`Prediction failed: ${e.message}`, 'error'); }
    finally { setLoading('btnPredict', 'spinPredict', false); }
}

async function triggerCleanup() {
    setLoading('btnCleanup', 'spinCleanup', true);
    try {
        addFeed('Destroying all cluster nodes...', 'error');
        const res = await fetch(`${API_URL}/cleanup`, { method: 'POST' });
        const data = await res.json();
        addFeed(`Cleanup complete: ${data.removed} nodes removed`, 'success');
        data.details.forEach(r => {
            addFeed(`SHUTDOWN ${r.node_name} — ${r.message} (${r.duration_ms.toFixed(0)}ms)`, r.success ? 'success' : 'error');
        });
        lastPredCpu = null;
        lastPredMem = null;
        lastRequiredNodes = null;
        elCpuPred.textContent = 'AI Pred: --';
        elMemPred.textContent = 'AI Pred: --';
        elAstarPath.innerHTML = '<div class="astar-placeholder">Trigger a stress test to see the A* plan</div>';
        setTimeout(fetchStatus, 300);
    } catch (e) { addFeed(`Cleanup failed: ${e.message}`, 'error'); }
    finally { setLoading('btnCleanup', 'spinCleanup', false); }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
addFeed('Dashboard initialized. Connecting to API...', 'info');
setInterval(fetchStatus, 1500);
fetchStatus();
