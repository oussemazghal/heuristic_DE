const API_URL = ""; // Relative path since we serve from FastAPI

// State
let currentResults = [];

// Init
document.addEventListener('DOMContentLoaded', async () => {
    await loadFunctions();

    document.getElementById('runBtn').addEventListener('click', runOptimization);
});

async function loadFunctions() {
    try {
        const res = await fetch(`${API_URL}/functions`);
        const data = await res.json();
        const select = document.getElementById('functionSelect');
        data.functions.forEach(f => {
            const opt = document.createElement('option');
            opt.value = f;
            opt.textContent = f;
            select.appendChild(opt);
        });
    } catch (e) {
        console.error("Failed to load functions", e);
    }
}

function toggleParams(algo) {
    const el = document.getElementById(`params-${algo}`);
    el.classList.toggle('active');
}

function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    document.querySelector(`button[onclick="switchTab('${tabName}')"]`).classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');

    // Resize plotly if needed
    if (tabName === 'convergence') {
        Plotly.Plots.resize('convergencePlot');
    }
}

async function runOptimization() {
    const loading = document.getElementById('loading');
    const resultsArea = document.getElementById('results-area');
    const welcome = document.getElementById('welcome-message');

    // Collect params
    const selectedAlgos = Array.from(document.querySelectorAll('.algo-check:checked')).map(cb => cb.value);

    if (selectedAlgos.length === 0) {
        alert("Please select at least one algorithm.");
        return;
    }

    const algorithms = selectedAlgos.map(algo => {
        const inputs = document.querySelectorAll(`#params-${algo} input`);
        const params = {};
        inputs.forEach(input => {
            const key = input.dataset.param;
            params[key] = parseFloat(input.value);
        });
        return { name: algo, params: params };
    });

    const payload = {
        function: document.getElementById('functionSelect').value,
        dimension: parseInt(document.getElementById('dimension').value),
        max_fes: parseInt(document.getElementById('maxFes').value),
        lb: parseFloat(document.getElementById('lb').value),
        ub: parseFloat(document.getElementById('ub').value),
        seed: parseInt(document.getElementById('seed').value),
        algorithms: algorithms
    };

    loading.style.display = 'flex';

    try {
        const res = await fetch(`${API_URL}/optimize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Optimization failed");

        currentResults = await res.json();

        renderResults(currentResults);

        welcome.style.display = 'none';
        resultsArea.style.display = 'block';
        switchTab('convergence');

    } catch (e) {
        alert(`Error: ${e.message}`);
    } finally {
        loading.style.display = 'none';
    }
}


function renderResults(results) {
    renderConvergence(results);
    renderSummary(results);
    renderDetails(results);
}

function renderConvergence(results) {
    // Couleurs ULTRA distinctes - séparation maximale dans l'espace colorimétrique
    const colors = [
        '#FF0000',  // ROUGE PUR - DE
        '#00FFFF',  // CYAN NÉON - jDE
        '#FF00FF',  // MAGENTA NÉON - jDE-Adapted
        '#00FF00',  // VERT NÉON - PSO
        '#FFFF00',  // JAUNE PUR - PSO-H
        '#0000FF',  // BLEU PUR - GA
        '#FF6600',  // ORANGE VIF - GSA
        '#9900FF',  // VIOLET ÉLECTRIQUE - ABC
        '#00FF99',  // Vert turquoise
        '#FF0099'   // Rose néon
    ];

    const traces = results.map((res, idx) => ({
        x: res.fes_history,
        y: res.history,
        mode: 'lines',
        name: res.algorithm,
        line: {
            color: colors[idx % colors.length],
            width: 2.5
        }
    }));

    const layout = {
        title: {
            text: 'Convergence Curves',
            font: { size: 16, color: '#e2e8f0' }
        },
        xaxis: {
            title: { text: 'Function Evaluations', font: { size: 14 } },
            gridcolor: '#334155',
            gridwidth: 1,
            showgrid: true,
            zeroline: false,
            color: '#cbd5e1'
        },
        yaxis: {
            title: { text: 'Best Fitness (Log Scale)', font: { size: 14 } },
            type: 'log',
            gridcolor: '#334155',
            gridwidth: 1,
            showgrid: true,
            zeroline: false,
            color: '#cbd5e1'
        },
        paper_bgcolor: '#1e293b',
        plot_bgcolor: '#0f172a',
        font: { color: '#f8fafc', family: 'Arial, sans-serif' },
        margin: { t: 50, r: 30, b: 60, l: 70 },
        legend: {
            bgcolor: 'rgba(30, 41, 59, 0.8)',
            bordercolor: '#475569',
            borderwidth: 1
        }
    };

    Plotly.newPlot('convergencePlot', traces, layout, { responsive: true });

    // Metrics cards
    const metricsContainer = document.getElementById('metrics-cards');
    metricsContainer.innerHTML = '';
    results.forEach(res => {
        const card = document.createElement('div');
        card.className = 'card';
        card.style.marginBottom = '0';
        card.innerHTML = `
            <div style="color: var(--text-secondary); font-size: 0.8rem;">${res.algorithm}</div>
            <div style="font-size: 1.2rem; font-weight: bold;">${res.best_value.toExponential(4)}</div>
        `;
        metricsContainer.appendChild(card);
    });
}


function renderSummary(results) {
    const tbody = document.querySelector('#summaryTable tbody');
    tbody.innerHTML = '';

    results.forEach(res => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${res.algorithm}</td>
            <td>${res.best_value.toExponential(6)}</td>
            <td>${res.fes_history[res.fes_history.length - 1]}</td>
            <td>${res.execution_time.toFixed(3)}</td>
            <td>${res.parameters.pop_size || '-'}</td>
        `;
        tbody.appendChild(tr);
    });

    // Ranking
    const sorted = [...results].sort((a, b) => a.best_value - b.best_value);
    const rankingList = document.getElementById('rankingList');
    rankingList.innerHTML = '';

    sorted.forEach((res, idx) => {
        const div = document.createElement('div');
        div.className = 'ranking-item';
        let medal = `${idx + 1}.`;
        if (idx === 0) medal = '🥇';
        if (idx === 1) medal = '🥈';
        if (idx === 2) medal = '🥉';

        div.innerHTML = `
            <span class="medal">${medal}</span>
            <div>
                <strong>${res.algorithm}</strong>
                <span style="color: var(--text-secondary)"> - ${res.best_value.toExponential(6)}</span>
            </div>
        `;
        rankingList.appendChild(div);
    });
}

function renderDetails(results) {
    const container = document.getElementById('detailsList');
    container.innerHTML = '';

    results.forEach(res => {
        const el = document.createElement('div');
        el.className = 'card';

        // Stats
        const mean = res.history.reduce((a, b) => a + b, 0) / res.history.length;
        const min = Math.min(...res.history);
        const max = Math.max(...res.history);

        el.innerHTML = `
            <h3>${res.algorithm} Details</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                <div>
                    <p><strong>Best Value:</strong> ${res.best_value.toExponential(6)}</p>
                    <p><strong>Execution Time:</strong> ${res.execution_time.toFixed(3)}s</p>
                    <p><strong>Total FES:</strong> ${res.fes_history[res.fes_history.length - 1]}</p>
                </div>
                <div>
                    <p><strong>Parameters:</strong></p>
                    <ul style="list-style: none; padding-left: 0.5rem; font-size: 0.9rem; color: var(--text-secondary);">
                        ${Object.entries(res.parameters).map(([k, v]) => `<li>${k}: ${v}</li>`).join('')}
                    </ul>
                </div>
            </div>
            <div style="margin-top: 1rem;">
                <p><strong>Best Solution (first 10 dims):</strong></p>
                <code style="display: block; background: rgba(0,0,0,0.3); padding: 0.5rem; border-radius: 0.25rem; margin-top: 0.5rem; overflow-x: auto;">
                    [${res.best_solution.slice(0, 10).map(n => n.toFixed(4)).join(', ')}${res.best_solution.length > 10 ? ', ...' : ''}]
                </code>
            </div>
        `;
        container.appendChild(el);
    });
}
