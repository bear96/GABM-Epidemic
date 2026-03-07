// ============================================================
// GABM Epidemic — Town UI  (Dewberry Hollow)
// Charts, bio panel, chat bubbles, hover interaction.
// Depends on: map-layout.js, agent-schedule.js,
//             town-render.js, town-pathfinding.js, town-agents.js
// ============================================================

'use strict';

// ─── Chat bubble state ───────────────────────────────────────
const chatCooldown  = new Map();
const activeBubbles = new Map();
const agentsInBubble = new Set();

// ─── Hover / selection state ─────────────────────────────────
let lastHoveredId = -1;
let lockedAgentId = -1;

// ─── Chart snapshots ─────────────────────────────────────────
let sirChartSnap   = null;
let mobilChartSnap = null;
let hoverRing      = null;  // Phaser graphics for hover highlight

// ═══════════════════════════════════════════════════════════════
// CHAT BUBBLES
// ═══════════════════════════════════════════════════════════════

function convoCooldown(idA, idB) {
  return 3000 + ((idA * 37 + idB * 53 + 7919) >>> 0) % 5000;
}

function restoreStatusEmoji(id) {
  if (!agentStatusTexts[id]) return;
  const key   = agentLocKey[id];
  const emoji = key === 'home' ? '🏠' : (LOCATIONS[key]?.emoji || '');
  agentStatusTexts[id].setText(`${agentInitials[id]}: ${emoji}`);
}

function destroyBubblesForAgent(id) {
  const toDelete = [];
  for (const [key] of activeBubbles) {
    const parts = key.split('-').map(Number);
    if (parts[0] === id || parts[1] === id) toDelete.push([key, parts[0], parts[1]]);
  }
  toDelete.forEach(([key, idA, idB]) => {
    activeBubbles.delete(key);
    agentsInBubble.delete(idA);
    agentsInBubble.delete(idB);
    restoreStatusEmoji(idA);
    restoreStatusEmoji(idB);
  });
}

function showChatBubble(idA, idB) {
  if (!scene) return;
  if (agentsInBubble.has(idA) || agentsInBubble.has(idB)) return;

  const bubbleKey = `${Math.min(idA, idB)}-${Math.max(idA, idB)}`;
  if (activeBubbles.has(bubbleKey)) return;

  agentsInBubble.add(idA);
  agentsInBubble.add(idB);

  // Face each other E/W
  const ca = agentContainers[idA];
  const cb = agentContainers[idB];
  const dirA = cb.x >= ca.x ? 'right' : 'left';
  const dirB = ca.x >= cb.x ? 'right' : 'left';
  agentSprites[idA].anims.play(`${agentCharNames[idA]}_${dirA}`, false);
  agentSprites[idB].anims.play(`${agentCharNames[idB]}_${dirB}`, false);

  // Show speech emoji in status text
  if (agentStatusTexts[idA]) agentStatusTexts[idA].setText(`${agentInitials[idA]}: \u{1F4AC}`);
  if (agentStatusTexts[idB]) agentStatusTexts[idB].setText(`${agentInitials[idB]}: \u{1F4AC}`);

  // Use a sentinel in activeBubbles (no sprite, just tracks active state)
  activeBubbles.set(bubbleKey, null);

  // Restore after delay
  scene.time.delayedCall(3000, () => {
    activeBubbles.delete(bubbleKey);
    agentsInBubble.delete(idA);
    agentsInBubble.delete(idB);
    restoreStatusEmoji(idA);
    restoreStatusEmoji(idB);
  });
}

// ═══════════════════════════════════════════════════════════════
// HOVER → BIO PANEL
// ═══════════════════════════════════════════════════════════════

function onCanvasMouseMove(e) {
  const rect   = game.canvas.getBoundingClientRect();
  const scaleX = VIEW_W / rect.width;
  const scaleY = VIEW_H / rect.height;
  const screenX = (e.clientX - rect.left) * scaleX;
  const screenY = (e.clientY - rect.top)  * scaleY;

  const cam    = scene.cameras.main;
  const wp     = cam.getWorldPoint(screenX, screenY);
  const worldX = wp.x; const worldY = wp.y;

  const hoverR = Math.max(20, 36 / cam.zoom);  // ~36 screen px at any zoom
  let closestId = -1, closestDist = hoverR * hoverR;
  agentContainers.forEach((c, id) => {
    if (!c.visible) return;
    const dx = c.x - worldX, dy = c.y - worldY;
    const d2 = dx * dx + dy * dy;
    if (d2 < closestDist) { closestDist = d2; closestId = id; }
  });


  // Floating canvas tooltip
  const tooltip = document.getElementById('agent-tooltip');
  if (tooltip) {
    if (closestId >= 0) {
      const row = agentRows[closestId]?.[currentStep];
      const name = row?.name || `Agent ${closestId}`;
      const health = row?.health_condition || 'Unknown';
      const hCol = healthCssColor(health);
      tooltip.innerHTML = `<span class="tt-name">${name}</span><span class="tt-health" style="color:${hCol}">${health}</span>`;
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 14) + 'px';
      tooltip.style.top  = (e.clientY - 8) + 'px';
    } else {
      tooltip.style.display = 'none';
    }
  }

  // Hover highlight ring
  if (typeof scene !== 'undefined' && scene) {
    if (!hoverRing) hoverRing = scene.add.graphics().setDepth(14);
    hoverRing.clear();
    if (closestId >= 0 && agentContainers[closestId]?.visible) {
      const ac = agentContainers[closestId];
      hoverRing.lineStyle(2, 0xffffff, 0.7);
      hoverRing.strokeCircle(ac.x, ac.y, 18);
    }
    // Cursor feedback
    game.canvas.style.cursor = closestId >= 0 ? 'pointer' : 'default';
  }

  if (lockedAgentId >= 0) return;
  if (closestId !== lastHoveredId) {
    lastHoveredId = closestId;
    if (closestId >= 0) showBioPanel(closestId);
  }
}

function showBioPanel(id) {
  const row   = agentRows[id]?.[currentStep];
  const panel = document.getElementById('bio-panel');
  if (!row) {
    panel.innerHTML = '<div class="bio-placeholder">no data for<br>this agent</div>';
    return;
  }
  panel.innerHTML = buildBioHtml(id, row);
}

// ═══════════════════════════════════════════════════════════════
// CHARTS (SVG — SIR + Mobility + Scatter)
// ═══════════════════════════════════════════════════════════════

function buildCharts() {
  const steps = Object.keys(popRows).map(Number).sort((a, b) => a - b);
  if (steps.length === 0) return;
  const chartsRow = document.getElementById('charts-row');
  if (chartsRow) chartsRow.style.display = 'flex';
  const scatter = document.getElementById('scatter-chart');
  if (scatter) scatter.style.display = 'block';
  drawSirChart(steps);
  drawMobilityChart(steps);
  updateChartPlayheads();
}

function drawSirChart(steps) {
  const el = document.getElementById('sir-chart');
  if (!el) return;
  const W = 576, H = 84, pad = 18;
  const mxStep = steps[steps.length - 1] || 1;
  const toX = s => (pad + (s / mxStep) * (W - pad * 2)).toFixed(1);
  const toY = v => (H - pad - (Math.max(0, Math.min(100, v)) / 100) * (H - pad * 2)).toFixed(1);

  const grid = [25, 50, 75].map(v =>
    `<line x1="${pad}" y1="${toY(v)}" x2="${W - pad}" y2="${toY(v)}" stroke="#1a2a1a" stroke-width="0.5"/>`
  ).join('');
  const yLabels = [0, 25, 50, 75, 100].map(v =>
    `<text x="2" y="${toY(v)}" dy="3" fill="#3a6a3a" font-size="6" font-family="sans-serif">${v}</text>`
  ).join('');
  const nTicks = Math.min(5, mxStep);
  const xLabels = Array.from({ length: nTicks + 1 }, (_, i) => {
    const s = Math.round((i / nTicks) * mxStep);
    return `<text x="${toX(s)}" y="${H - 2}" fill="#3a6a3a" font-size="6" font-family="sans-serif" text-anchor="middle">d${s}</text>`;
  }).join('');

  const series = [
    { key: 'Susceptible', col: '#4895EF' },
    { key: 'Infected',    col: '#FF6B35' },
    { key: 'Recovered',   col: '#8338EC' },
  ];
  const polylines = series.map(({ key, col }) => {
    const pts = steps
      .map(s => { const p = popRows[s]; return p ? `${toX(s)},${toY(Number(p[key]) || 0)}` : null; })
      .filter(Boolean).join(' ');
    return pts ? `<polyline points="${pts}" stroke="${col}" stroke-width="1.5" fill="none" opacity="0.85"/>` : '';
  }).join('');

  const lgd = [['S', '#4895EF', 0], ['I', '#FF6B35', 14], ['R', '#8338EC', 28]].map(([lbl, col, off]) =>
    `<rect x="${W - 22}" y="${pad + off}" width="5" height="5" fill="${col}"/>` +
    `<text x="${W - 14}" y="${pad + 5 + off}" fill="${col}" font-size="6" font-family="sans-serif">${lbl}</text>`
  ).join('');

  el.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" style="display:block">
    <rect width="${W}" height="${H}" fill="#08100a"/>
    ${grid}${yLabels}${xLabels}
    <text x="${pad + 2}" y="${pad - 3}" fill="#4a7a4a" font-size="6" font-family="sans-serif" font-weight="bold">SIR CURVE</text>
    ${polylines}${lgd}
    <line id="sir-playhead" x1="${toX(0)}" y1="${pad}" x2="${toX(0)}" y2="${H - pad}" stroke="rgba(255,255,255,0.4)" stroke-width="1"/>
  </svg>`;
  sirChartSnap = true;
}

function drawMobilityChart(steps) {
  const el = document.getElementById('mobility-chart');
  if (!el) return;
  const W = 576, H = 84, pad = 18;
  const mxStep = steps[steps.length - 1] || 1;
  const toX = s => (pad + (s / mxStep) * (W - pad * 2)).toFixed(1);
  const toY = v => (H - pad - (Math.max(0, Math.min(100, v)) / 100) * (H - pad * 2)).toFixed(1);

  const grid = [25, 50, 75].map(v =>
    `<line x1="${pad}" y1="${toY(v)}" x2="${W - pad}" y2="${toY(v)}" stroke="#121a24" stroke-width="0.5"/>`
  ).join('');
  const yLabels = [0, 25, 50, 75, 100].map(v =>
    `<text x="2" y="${toY(v)}" dy="3" fill="#2a3a58" font-size="6" font-family="sans-serif">${v}</text>`
  ).join('');
  const nTicks = Math.min(5, mxStep);
  const xLabels = Array.from({ length: nTicks + 1 }, (_, i) => {
    const s = Math.round((i / nTicks) * mxStep);
    return `<text x="${toX(s)}" y="${H - 2}" fill="#2a3a58" font-size="6" font-family="sans-serif" text-anchor="middle">d${s}</text>`;
  }).join('');

  const mobSeries = [
    { key: '# Grid', col: '#00CED1', scale: 1 },
    { key: '# Home', col: '#4a78c8', scale: 1 },
    { key: 'New Infections', col: '#FF6B35', scale: 4 },
  ];
  const polylines = mobSeries.map(({ key, col, scale }) => {
    const pts = steps
      .map(s => { const p = popRows[s]; return p ? `${toX(s)},${toY(Math.min(100, (Number(p[key]) || 0) * scale))}` : null; })
      .filter(Boolean).join(' ');
    return pts ? `<polyline points="${pts}" stroke="${col}" stroke-width="1.5" fill="none" opacity="0.85"/>` : '';
  }).join('');

  const lgd = [['Grid', '#00CED1', 0], ['Home', '#4a78c8', 14], ['New×4', '#FF6B35', 28]].map(([lbl, col, off]) =>
    `<rect x="${W - 32}" y="${pad + off}" width="5" height="5" fill="${col}"/>` +
    `<text x="${W - 24}" y="${pad + 5 + off}" fill="${col}" font-size="6" font-family="sans-serif">${lbl}</text>`
  ).join('');

  el.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" style="display:block">
    <rect width="${W}" height="${H}" fill="#080c14"/>
    ${grid}${yLabels}${xLabels}
    <text x="${pad + 2}" y="${pad - 3}" fill="#3a5878" font-size="6" font-family="sans-serif" font-weight="bold">MOBILITY</text>
    ${polylines}${lgd}
    <line id="mob-playhead" x1="${toX(0)}" y1="${pad}" x2="${toX(0)}" y2="${H - pad}" stroke="rgba(255,255,255,0.4)" stroke-width="1"/>
  </svg>`;
  mobilChartSnap = true;
}

function drawScatterChart(upToStep) {
  const canvas = document.getElementById('scatter-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const padL = 22, padB = 14, padT = 14, padR = 10;

  ctx.fillStyle = '#08100c';
  ctx.fillRect(0, 0, W, H);
  const chartW = W - padL - padR, chartH = H - padT - padB;

  ctx.fillStyle = '#2e4a32';
  ctx.font = '5px "Press Start 2P", monospace';
  ctx.fillText('% INFECTED (X) vs MOBILITY (Y)', padL + 2, padT - 2);

  ctx.fillStyle = '#2a4a2a';
  ctx.font = '4px "Press Start 2P", monospace';
  [0, 25, 50, 75, 100].forEach(v => {
    ctx.fillText(`${v}`, 1, H - padB - (v / 100) * chartH + 2);
    ctx.fillText(`${v}`, padL + (v / 100) * chartW - 3, H - 1);
  });

  ctx.strokeStyle = '#2a4a2a'; ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB);
  ctx.moveTo(padL, H - padB); ctx.lineTo(W - padR, H - padB);
  ctx.stroke();

  const toX = v => padL + (v / 100) * chartW;
  const toY = v => H - padB - (v / 100) * chartH;

  const pts = [];
  for (let s = 0; s <= upToStep; s++) {
    const pop = popRows[s];
    if (!pop) continue;
    pts.push({ x: Number(pop['Infected']) || 0, y: Number(pop['# Grid']) || 0 });
  }
  if (pts.length === 0) return;

  ctx.strokeStyle = '#2a5030'; ctx.lineWidth = 1; ctx.globalAlpha = 0.6;
  ctx.beginPath();
  pts.forEach((p, i) => { i === 0 ? ctx.moveTo(toX(p.x), toY(p.y)) : ctx.lineTo(toX(p.x), toY(p.y)); });
  ctx.stroke();
  ctx.globalAlpha = 1;

  pts.forEach((p, i) => {
    const t = i / Math.max(pts.length - 1, 1);
    ctx.fillStyle = t < 0.33 ? '#4895EF' : t < 0.66 ? '#FF6B35' : '#8338EC';
    ctx.globalAlpha = 0.7 + 0.3 * (i / pts.length);
    ctx.beginPath();
    ctx.arc(toX(p.x), toY(p.y), i === pts.length - 1 ? 4 : 2.5, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;

  const last = pts[pts.length - 1];
  ctx.fillStyle = '#c8d8e8';
  ctx.font = '4px "Press Start 2P", monospace';
  ctx.fillText(`d${upToStep}`, toX(last.x) + 5, toY(last.y) - 3);
}

function updateChartPlayheads() {
  const steps = Object.keys(popRows).map(Number).sort((a, b) => a - b);
  if (steps.length === 0) return;
  const mxStep = steps[steps.length - 1] || 1;
  const W = 576, pad = 18;
  const x = (pad + (currentStep / mxStep) * (W - pad * 2)).toFixed(1);

  const sirPh = document.getElementById('sir-playhead');
  if (sirPh) { sirPh.setAttribute('x1', x); sirPh.setAttribute('x2', x); }
  const mobPh = document.getElementById('mob-playhead');
  if (mobPh) { mobPh.setAttribute('x1', x); mobPh.setAttribute('x2', x); }

  drawScatterChart(currentStep);
}
