/**
 * PixelGuard — app.js
 * Handles: drag-and-drop upload, analysis via REST API, results display, history.
 */

const API_BASE = '';  // same origin

/* ── DOM References ── */
const dropZone    = document.getElementById('drop-zone');
const fileInput   = document.getElementById('file-input');
const dropIdle    = document.getElementById('drop-idle');
const dropPreview = document.getElementById('drop-preview');
const previewImg  = document.getElementById('preview-img');
const analyzeBtn  = document.getElementById('analyze-btn');
const errorBox    = document.getElementById('error-box');
const errorMsg    = document.getElementById('error-msg');

const resultEmpty   = document.getElementById('result-empty');
const resultLoading = document.getElementById('result-loading');
const resultContent = document.getElementById('result-content');

const scoreFillCircle = document.getElementById('score-fill-circle');
const scoreValue      = document.getElementById('score-value');
const qualityChip     = document.getElementById('quality-label-chip');
const confBar         = document.getElementById('conf-bar');
const confValue       = document.getElementById('conf-value');
const issuesList      = document.getElementById('issues-list');
const probaGrid       = document.getElementById('proba-grid');
const statsGrid       = document.getElementById('stats-grid');

const historyBody   = document.getElementById('history-body');
const historyEmpty  = document.getElementById('history-empty-row');
const refreshBtn    = document.getElementById('refresh-btn');

const modalOverlay  = document.getElementById('modal-overlay');
const modalTitle    = document.getElementById('modal-title');
const modalBody     = document.getElementById('modal-body');
const modalClose    = document.getElementById('modal-close');

/* ── State ── */
let selectedFile = null;

/* ═══════════════════════════════════════════
   Upload / Drop Zone
═══════════════════════════════════════════ */

function setPreview(file) {
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.onload = () => URL.revokeObjectURL(url);
  dropIdle.classList.add('hidden');
  dropPreview.classList.remove('hidden');
  analyzeBtn.disabled = false;
  hideError();
  showEmpty();
}

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) { selectedFile = file; setPreview(file); }
});

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });

dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) {
    selectedFile = file;
    setPreview(file);
    // sync to input so FormData works
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
  }
});

/* ═══════════════════════════════════════════
   Analyze
═══════════════════════════════════════════ */

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  showLoading();
  hideError();

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const resp = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      body: formData,
    });

    const data = await resp.json();

    if (!resp.ok) {
      showError(data.detail || `Error ${resp.status}`);
      showEmpty();
      return;
    }

    renderResult(data);
    loadHistory();   // auto-refresh history after a new analysis
  } catch (err) {
    showError('Network error — is the server running?');
    showEmpty();
  }
});

/* ═══════════════════════════════════════════
   Render Result
═══════════════════════════════════════════ */

const ISSUE_ICONS = {
  blur:          '🌫',
  underexposure: '🌑',
  overexposure:  '☀️',
  noise:         '📡',
  corruption:    '💀',
  visual_defect: '⚡',
};

function renderResult(data) {
  const score = data.quality_score ?? 0;
  const label = data.quality_label ?? 'UNKNOWN';
  const conf  = data.confidence ?? 0;

  // Score ring
  const circumference = 2 * Math.PI * 42;  // r=42
  const offset = circumference * (1 - score / 100);
  scoreFillCircle.style.strokeDashoffset = offset;
  scoreFillCircle.style.stroke =
    score >= 70 ? 'var(--green)' :
    score >= 40 ? 'var(--yellow)' : 'var(--red)';

  scoreValue.textContent = score;

  // Quality label chip
  qualityChip.textContent = label;
  qualityChip.className = `quality-label quality-label--${label}`;

  // Confidence bar
  confBar.style.width = `${Math.round(conf * 100)}%`;
  confValue.textContent = `${Math.round(conf * 100)}%`;

  // Issues
  issuesList.innerHTML = '';
  if (data.issues && data.issues.length > 0) {
    data.issues.forEach((issue, i) => {
      const el = document.createElement('div');
      el.className = 'issue-item';
      el.style.animationDelay = `${i * 60}ms`;
      el.innerHTML = `
        <div class="issue-type">
          <span class="issue-icon">${ISSUE_ICONS[issue.type] || '🔍'}</span>
          ${issue.type.replace('_', ' ')}
        </div>
        <div class="issue-right">
          <span class="issue-severity severity--${issue.severity}">${issue.severity}</span>
          <span class="issue-conf">${Math.round(issue.confidence * 100)}%</span>
        </div>
      `;
      issuesList.appendChild(el);
    });
  } else {
    issuesList.innerHTML = '<div class="no-issues">✅ No issues detected</div>';
  }

  // Label probabilities
  probaGrid.innerHTML = '';
  const proba = data.label_probabilities ?? {};
  const ORDER = ['ACCEPTABLE', 'DEGRADED', 'DEFECTIVE'];
  ORDER.forEach(k => {
    if (!(k in proba)) return;
    const pct = Math.round(proba[k] * 100);
    const el = document.createElement('div');
    el.className = 'proba-item';
    el.innerHTML = `
      <span class="proba-name">${k.toLowerCase()}</span>
      <div class="proba-bar-wrap">
        <div class="proba-bar proba-bar--${k}" style="width:${pct}%"></div>
      </div>
      <span class="proba-pct">${pct}%</span>
    `;
    probaGrid.appendChild(el);
  });

  // Image stats
  statsGrid.innerHTML = '';
  const stats = data.image_stats ?? {};
  const STAT_LABELS = {
    width: 'Width', height: 'Height', sharpness: 'Sharpness',
    mean_brightness: 'Brightness', contrast: 'Contrast',
    noise: 'Noise', edge_density: 'Edge Density', block_uniformity: 'Uniformity',
  };
  Object.entries(stats).forEach(([k, v]) => {
    const el = document.createElement('div');
    el.className = 'stat-item';
    const display = typeof v === 'number' && !Number.isInteger(v)
      ? v.toFixed(3) : v;
    el.innerHTML = `
      <div class="stat-key">${STAT_LABELS[k] || k}</div>
      <div class="stat-val">${display}</div>
    `;
    statsGrid.appendChild(el);
  });

  showResult();
}

/* ═══════════════════════════════════════════
   UI State Helpers
═══════════════════════════════════════════ */

function showEmpty() {
  resultEmpty.classList.remove('hidden');
  resultLoading.classList.add('hidden');
  resultContent.classList.add('hidden');
}

function showLoading() {
  resultEmpty.classList.add('hidden');
  resultLoading.classList.remove('hidden');
  resultContent.classList.add('hidden');
}

function showResult() {
  resultEmpty.classList.add('hidden');
  resultLoading.classList.add('hidden');
  resultContent.classList.remove('hidden');
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorBox.classList.remove('hidden');
}

function hideError() {
  errorBox.classList.add('hidden');
}

/* ═══════════════════════════════════════════
   History
═══════════════════════════════════════════ */

async function loadHistory() {
  try {
    const resp = await fetch(`${API_BASE}/history?limit=50`);
    if (!resp.ok) return;
    const data = await resp.json();
    renderHistory(data.analyses ?? []);
  } catch (_) {
    // history not critical — fail silently
  }
}

function renderHistory(analyses) {
  // Remove all rows except the empty placeholder
  const rows = historyBody.querySelectorAll('tr:not(#history-empty-row)');
  rows.forEach(r => r.remove());

  if (!analyses || analyses.length === 0) {
    historyEmpty.classList.remove('hidden');
    return;
  }

  historyEmpty.classList.add('hidden');

  analyses.forEach(a => {
    const tr = document.createElement('tr');
    const date = a.created_at
      ? new Date(a.created_at).toLocaleString()
      : '—';
    tr.innerHTML = `
      <td>${a.id}</td>
      <td title="${a.filename || ''}">${truncate(a.filename || '—', 28)}</td>
      <td><span class="tbl-label tbl-label--${a.quality_label}">${a.quality_label}</span></td>
      <td>${a.quality_score}</td>
      <td>${date}</td>
      <td><button class="tbl-view-btn" data-id="${a.id}">View</button></td>
    `;
    tr.querySelector('.tbl-view-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      openModal(a.id);
    });
    tr.addEventListener('click', () => openModal(a.id));
    historyBody.appendChild(tr);
  });
}

function truncate(str, n) {
  return str.length > n ? str.slice(0, n - 1) + '…' : str;
}

/* ── Modal ── */

async function openModal(id) {
  modalTitle.textContent = `Analysis #${id}`;
  modalBody.innerHTML = '<div style="color:var(--text-muted);font-size:0.875rem;padding:0.5rem 0">Loading…</div>';
  modalOverlay.classList.remove('hidden');
  document.body.style.overflow = 'hidden';

  try {
    const resp = await fetch(`${API_BASE}/history/${id}`);
    if (!resp.ok) { modalBody.innerHTML = '<p style="color:var(--red)">Failed to load.</p>'; return; }
    const data = await resp.json();
    renderModal(data);
  } catch (_) {
    modalBody.innerHTML = '<p style="color:var(--red)">Network error.</p>';
  }
}

function renderModal(data) {
  const r = data.result ?? {};
  const rows = [
    ['Filename',  data.filename ?? '—'],
    ['Timestamp', data.created_at ? new Date(data.created_at).toLocaleString() : '—'],
    ['Quality Score', `${r.quality_score ?? '—'} / 100`],
    ['Quality Label', r.quality_label ?? '—'],
    ['Confidence', r.confidence != null ? `${Math.round(r.confidence * 100)}%` : '—'],
    ['Issues Detected', (r.issues?.length ?? 0)],
  ];

  let html = rows.map(([k, v]) => `
    <div class="modal-row">
      <span class="modal-key">${k}</span>
      <span class="modal-val">${v}</span>
    </div>
  `).join('');

  html += `<div style="margin-top:0.5rem;font-size:0.7rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:var(--text-subtle);margin-bottom:0.5rem">Raw JSON</div>`;
  html += `<div class="modal-json">${escapeHtml(JSON.stringify(r, null, 2))}</div>`;

  modalBody.innerHTML = html;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

modalClose.addEventListener('click', closeModal);
modalOverlay.addEventListener('click', (e) => { if (e.target === modalOverlay) closeModal(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

function closeModal() {
  modalOverlay.classList.add('hidden');
  document.body.style.overflow = '';
}

/* ── Refresh button ── */
refreshBtn.addEventListener('click', () => {
  const icon = document.getElementById('refresh-icon');
  icon.style.animation = 'spin 0.6s linear';
  icon.addEventListener('animationend', () => { icon.style.animation = ''; }, { once: true });
  loadHistory();
});

/* ── Init ── */
loadHistory();
