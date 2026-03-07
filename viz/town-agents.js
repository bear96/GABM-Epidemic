// ============================================================
// GABM Epidemic — Town Agents  (Dewberry Hollow)
// Agent sprites, animations, movement, infection flash.
// Depends on: map-layout.js, agent-schedule.js,
//             town-render.js, town-pathfinding.js
// ============================================================

'use strict';

// ─── 100 character sprites (Pipoya RPG — 50 male, 50 female, shuffled) ──
const CHAR_NAMES = [
  'Pipoya_F43', 'Pipoya_M16', 'Pipoya_F01', 'Pipoya_M21', 'Pipoya_M19',
  'Pipoya_F09', 'Pipoya_F28', 'Pipoya_M45', 'Pipoya_F33', 'Pipoya_F16',
  'Pipoya_M40', 'Pipoya_M42', 'Pipoya_F22', 'Pipoya_F17', 'Pipoya_F44',
  'Pipoya_M25', 'Pipoya_F47', 'Pipoya_F49', 'Pipoya_M15', 'Pipoya_M22',
  'Pipoya_M38', 'Pipoya_M07', 'Pipoya_F08', 'Pipoya_M17', 'Pipoya_F34',
  'Pipoya_M37', 'Pipoya_F32', 'Pipoya_F12', 'Pipoya_F25', 'Pipoya_M10',
  'Pipoya_M05', 'Pipoya_F05', 'Pipoya_F20', 'Pipoya_F07', 'Pipoya_M27',
  'Pipoya_M31', 'Pipoya_M47', 'Pipoya_M02', 'Pipoya_M08', 'Pipoya_M03',
  'Pipoya_M43', 'Pipoya_F41', 'Pipoya_F37', 'Pipoya_F03', 'Pipoya_M29',
  'Pipoya_F21', 'Pipoya_M46', 'Pipoya_F24', 'Pipoya_M09', 'Pipoya_F23',
  'Pipoya_M04', 'Pipoya_F06', 'Pipoya_M39', 'Pipoya_M13', 'Pipoya_M34',
  'Pipoya_F45', 'Pipoya_F15', 'Pipoya_F14', 'Pipoya_M24', 'Pipoya_M14',
  'Pipoya_F35', 'Pipoya_M01', 'Pipoya_M36', 'Pipoya_M35', 'Pipoya_M06',
  'Pipoya_F50', 'Pipoya_M41', 'Pipoya_F11', 'Pipoya_F36', 'Pipoya_F39',
  'Pipoya_M11', 'Pipoya_M44', 'Pipoya_F46', 'Pipoya_F10', 'Pipoya_M20',
  'Pipoya_F31', 'Pipoya_F40', 'Pipoya_M48', 'Pipoya_M28', 'Pipoya_F26',
  'Pipoya_F29', 'Pipoya_F04', 'Pipoya_F48', 'Pipoya_F13', 'Pipoya_F30',
  'Pipoya_F42', 'Pipoya_F19', 'Pipoya_F38', 'Pipoya_F18', 'Pipoya_M50',
  'Pipoya_M33', 'Pipoya_M12', 'Pipoya_M18', 'Pipoya_M32', 'Pipoya_M30',
  'Pipoya_M23', 'Pipoya_F27', 'Pipoya_M26', 'Pipoya_M49', 'Pipoya_F02',
];

// ─── Agent state arrays ──────────────────────────────────────
const agentContainers  = [];
const agentSprites     = [];
const agentBadges      = [];
const agentCharNames   = [];
const agentStatusTexts = [];
const agentInitials    = [];
const agentLocKey      = [];
const agentDestPos     = [];
const agentSettled     = new Array(100).fill(true);
const agentVisualPartner = new Array(100).fill(-1);  // per-step visual partner id

// ═══════════════════════════════════════════════════════════════
// ANIMATIONS
// ═══════════════════════════════════════════════════════════════

function createAnimations(sc) {
  const DIRS = ['down', 'up', 'left', 'right'];
  CHAR_NAMES.forEach(name => {
    DIRS.forEach(dir => {
      sc.anims.create({
        key: `${name}_${dir}_walk`,
        frames: [
          { key: name, frame: `${dir}-walk.000` },
          { key: name, frame: `${dir}-walk.001` },
          { key: name, frame: `${dir}-walk.002` },
          { key: name, frame: `${dir}-walk.003` },
          { key: name, frame: `${dir}-walk.001` },
        ],
        frameRate: 6, repeat: -1,
      });
      sc.anims.create({
        key: `${name}_${dir}`,
        frames: [{ key: name, frame: dir }],
        frameRate: 1, repeat: 0,
      });
    });
  });
}

// ═══════════════════════════════════════════════════════════════
// AGENT CONTAINERS
// ═══════════════════════════════════════════════════════════════

function buildAgentContainers(sc) {
  for (let id = 0; id < 100; id++) {
    const charName = CHAR_NAMES[id % CHAR_NAMES.length];
    const homePos  = getHomePosT(id);

    const container = sc.add.container(homePos.x, homePos.y);
    container.setDepth(15 + (id % 10) * 0.01);

    const glow = sc.add.graphics();
    glow.fillStyle(0x4895EF, 0.16);
    glow.fillCircle(0, 0, 12);
    container.add(glow);

    const sprite = sc.add.sprite(0, 2, charName, 'down');
    sprite.setScale(1.0);
    container.add(sprite);

    const badge = sc.add.graphics();
    badge.setPosition(9, -13);
    container.add(badge);

    const parts    = charName.split('_');
    const initials = ((parts[0]?.[0] || '') + (parts[1]?.[0] || '')).toUpperCase();
    agentInitials.push(initials);

    const statusTxt = sc.add.text(0, -36, `${initials}: 🏠`, {
      fontSize: '9px', fontFamily: 'monospace',
      color: '#000000', backgroundColor: '#ffffffdd',
      padding: { x: 4, y: 2 },
    }).setOrigin(0.5, 1);
    container.add(statusTxt);

    agentContainers.push(container);
    agentSprites.push(sprite);
    agentBadges.push(badge);
    agentCharNames.push(charName);
    agentStatusTexts.push(statusTxt);
    agentLocKey.push('home');
    agentDestPos.push({ x: homePos.x, y: homePos.y });
  }
}

// ── Badge / glow updates ─────────────────────────────────────

function updateBadge(id, condition) {
  const badge = agentBadges[id];
  badge.clear();
  const col = healthHex(condition);
  if (condition === 'Susceptible') {
    badge.fillStyle(col, 1);
    badge.fillCircle(0, 0, 3.5);
  } else if (condition === 'Exposed') {
    badge.fillStyle(col, 1);
    drawDiamond(badge, 0, 0, 5);
  } else if (condition === 'Infected') {
    badge.fillStyle(col, 1);
    drawStar(badge, 0, 0, 5, 5, 2.5);
  } else {
    badge.fillStyle(col, 1);
    badge.fillRect(-3.5, -3.5, 7, 7);
  }
}

function updateGlow(id, condition) {
  const glow = agentContainers[id].list[0];
  glow.clear();
  const col = condition === 'Exposed'   ? 0xFFB800
            : condition === 'Infected'  ? 0xFF6B35
            : condition === 'Recovered' ? 0x8338EC : 0x4895EF;
  glow.fillStyle(col, 0.18);
  glow.fillCircle(0, 0, 12);
}

function drawStar(gfx, x, y, nPts, outerR, innerR) {
  const step  = Math.PI / nPts;
  const verts = [];
  for (let i = 0; i < nPts * 2; i++) {
    const r = i % 2 === 0 ? outerR : innerR;
    const a = i * step - Math.PI / 2;
    verts.push({ x: x + r * Math.cos(a), y: y + r * Math.sin(a) });
  }
  gfx.fillPoints(verts, true);
}

function drawDiamond(gfx, x, y, r) {
  gfx.fillPoints([
    { x: x, y: y - r },
    { x: x + r * 0.7, y: y },
    { x: x, y: y + r },
    { x: x - r * 0.7, y: y },
  ], true);
}

// ═══════════════════════════════════════════════════════════════
// INFECTION FLASH
// ═══════════════════════════════════════════════════════════════

function flashNewInfections(step) {
  if (step === 0 || !scene) return;
  for (let id = 0; id < 100; id++) {
    const prev = agentRows[id]?.[step - 1]?.health_condition;
    const curr = agentRows[id]?.[step]?.health_condition;
    if (prev === 'Susceptible' && curr === 'Infected') playInfectionFlash(id);
  }
}

function playInfectionFlash(id) {
  const container = agentContainers[id];
  const badge     = agentBadges[id];
  if (!container || !badge || !scene) return;

  scene.tweens.add({
    targets: badge,
    scaleX: 3.5, scaleY: 3.5,
    duration: 300, ease: 'Quad.easeOut', yoyo: true,
    onStart() { badge.clear(); badge.fillStyle(0xFF2200, 1); badge.fillCircle(0, 0, 5); },
    onComplete() { badge.clear(); badge.fillStyle(0xFF6B35, 1); badge.fillCircle(0, 0, 4); },
  });

  const txt = scene.add.text(container.x, container.y - 20, 'INFECTED!', {
    fontSize: '7px', fontFamily: '"Press Start 2P", monospace',
    color: '#FF2200', stroke: '#000000', strokeThickness: 2,
  }).setOrigin(0.5, 1).setDepth(22);

  scene.tweens.add({
    targets: txt, y: container.y - 50, alpha: 0,
    duration: 2200, ease: 'Quad.easeOut',
    onComplete() { txt.destroy(); },
  });
}

// ═══════════════════════════════════════════════════════════════
// AGENT MOVEMENT
// ═══════════════════════════════════════════════════════════════

function killAgentTweens() {
  agentContainers.forEach(c => scene.tweens.killTweensOf(c));
}

function moveAgents(animate) {
  killAgentTweens();
  if (typeof window._pathOverlayBegin === 'function') window._pathOverlayBegin();
  const subStepMs = animate ? Math.max(200, Math.round(10000 / playSpeed)) : 0;

  // Pass 1: compute all plans first
  const nRounds = maxRoundsPerStep[currentStep] || 1;
  const plans = [];
  for (let id = 0; id < agentContainers.length; id++) {
    const row  = agentRows[id]?.[currentStep];
    const contactCount = agentContactCounts[currentStep]?.[id] || 0;
    plans[id] = getAgentDayPlan(id, currentStep, row, contactCount, nRounds);
  }

  // Pass 2: assign unique tile positions per building (no stacking)
  agentVisualPartner.fill(-1);

  // Group agents by building
  const buildingAgents = {};
  for (let id = 0; id < agentContainers.length; id++) {
    const ss = plans[id].subSteps[currentSubStep];
    if (ss.locationKey !== 'home') {
      if (!buildingAgents[ss.locationKey]) buildingAgents[ss.locationKey] = [];
      buildingAgents[ss.locationKey].push(id);
    }
  }

  // Build pair lookup from this round's contact data
  const pairOf = {};
  if (currentSubStep < nRounds) {
    const roundPairs = contactRoundsMap[currentStep]?.[currentSubStep] || [];
    for (const { a, b } of roundPairs) {
      const ssA = plans[a]?.subSteps[currentSubStep];
      const ssB = plans[b]?.subSteps[currentSubStep];
      if (ssA && ssB && ssA.locationKey === ssB.locationKey && ssA.locationKey !== 'home') {
        pairOf[a] = b;
        pairOf[b] = a;
        agentVisualPartner[a] = b;
        agentVisualPartner[b] = a;
      }
    }
  }

  // Assign unique tiles per building (evenly spread via stride)
  for (const [locKey, ids] of Object.entries(buildingAgents)) {
    const tiles = getBuildingTiles(locKey);
    if (tiles.length === 0) continue;

    // Deterministic shuffle so positions vary day to day
    const shuffled = ids.slice().sort((a, b) => {
      const ha = ((a * 31 + currentStep * 7) >>> 0) % 9973;
      const hb = ((b * 31 + currentStep * 7) >>> 0) % 9973;
      return ha - hb;
    });

    // Count how many tile slots we need (pairs share a tile, unpaired get 1 each)
    let nPairs = 0;
    const seenPair = new Set();
    for (const id of shuffled) {
      const p = pairOf[id];
      if (p !== undefined && !seenPair.has(p)) { nPairs++; seenPair.add(id); }
    }
    const nUnpaired = ids.length - nPairs * 2;
    const totalSlots = nPairs + nUnpaired;

    // Stride evenly across row-major tile list (spreads across rows + columns)
    const stride = totalSlots > 0 ? Math.max(1, Math.floor(tiles.length / totalSlots)) : 1;

    let slotIdx = 0;
    const assigned = new Set();

    // First: assign pairs (A left, B right — 28px apart, centered on tile)
    for (const id of shuffled) {
      if (assigned.has(id)) continue;
      const partner = pairOf[id];
      if (partner === undefined || assigned.has(partner)) continue;

      const t = tiles[(slotIdx * stride) % tiles.length];
      slotIdx++;

      plans[id].subSteps[currentSubStep].x = t.x - 14;
      plans[id].subSteps[currentSubStep].y = t.y;
      plans[partner].subSteps[currentSubStep].x = t.x + 14;
      plans[partner].subSteps[currentSubStep].y = t.y;

      assigned.add(id);
      assigned.add(partner);
    }

    // Then: assign unpaired agents
    for (const id of shuffled) {
      if (assigned.has(id)) continue;
      const t = tiles[(slotIdx * stride) % tiles.length];
      slotIdx++;
      plans[id].subSteps[currentSubStep].x = t.x;
      plans[id].subSteps[currentSubStep].y = t.y;
      assigned.add(id);
    }
  }

  // Build partner planned X positions (avoids race condition in markArrived)
  const partnerPlannedX = new Map();
  for (let id = 0; id < agentContainers.length; id++) {
    const vp = agentVisualPartner[id];
    if (vp >= 0) {
      partnerPlannedX.set(id, plans[vp].subSteps[currentSubStep].x);
    }
  }

  // Pass 3: move all agents
  for (let id = 0; id < agentContainers.length; id++) {
    const row  = agentRows[id]?.[currentStep];
    const plan = plans[id];
    const ss   = plan.subSteps[currentSubStep];
    const cond = (typeof getEffectiveHealth === 'function')
      ? getEffectiveHealth(id, currentStep)
      : (row?.health_condition || 'Susceptible');

    updateBadge(id, cond);
    updateGlow(id, cond);

    const toPos = { x: ss.x, y: ss.y };
    const toKey = ss.locationKey;

    // Dim house immediately when agent leaves home
    if (toKey !== 'home') setHomeLit(id, false);

    if (subStepMs === 0) {
      agentContainers[id].x = toPos.x;
      agentContainers[id].y = toPos.y;
      agentLocKey[id]  = toKey;
      agentDestPos[id] = toPos;
      agentSettled[id] = true;
      // Face visual partner E/W, otherwise default facing down
      const vp = agentVisualPartner[id];
      if (vp >= 0 && toKey !== 'home') {
        const vpX = plans[vp].subSteps[currentSubStep].x;
        const dir = vpX > toPos.x ? 'right' : 'left';
        agentSprites[id].anims.play(`${agentCharNames[id]}_${dir}`, false);
      } else {
        agentSprites[id].anims.play(`${agentCharNames[id]}_down`, false);
      }
      if (agentStatusTexts[id]) {
        const emoji = toKey === 'home' ? '🏠' : (LOCATIONS[toKey]?.emoji || '');
        agentStatusTexts[id].setText(`${agentInitials[id]}: ${emoji}`);
      }
      // Hide agent sprite when at home (they're "inside") and brighten house
      agentContainers[id].setVisible(toKey !== 'home');
      if (toKey === 'home') setHomeLit(id, true);
    } else {
      agentSettled[id] = false;
      // Pre-compute paths synchronously (avoids jank from BFS inside setTimeout)
      const fromPos = { x: agentContainers[id].x, y: agentContainers[id].y };
      const fromKey = agentLocKey[id];
      const waypoints = computeWalkPath(id, fromKey, fromPos, toKey, toPos);
      const staggerMs = Math.round(((id * 97 + currentStep * 37) % 3000) / playSpeed);
      setTimeout(() => {
        if (!scene) return;
        destroyBubblesForAgent(id);
        // Show agent when leaving home
        agentContainers[id].setVisible(true);
        // Draw path debug if overlay enabled (press 'P')
        if (typeof window._pathOverlayGet === 'function') {
          const gfx = window._pathOverlayGet();
          if (gfx && waypoints.length > 0) {
            const hue = (id * 37) % 360;
            const col = Phaser.Display.Color.HSLToColor(hue / 360, 0.8, 0.5).color;
            gfx.lineStyle(2, col, 0.7);
            gfx.beginPath();
            gfx.moveTo(waypoints[0].x, waypoints[0].y);
            for (const wp of waypoints) gfx.lineTo(wp.x, wp.y);
            gfx.strokePath();
          }
        }
        agentLocKey[id]  = toKey;
        agentDestPos[id] = toPos;
        if (agentStatusTexts[id]) {
          const destEmoji = toKey === 'home' ? '🏠' : (LOCATIONS[toKey]?.emoji || '');
          agentStatusTexts[id].setText(`${agentInitials[id]}: 🚶${destEmoji}`);
        }
        const budget = Math.max(200, subStepMs - staggerMs);
        chainWalkTimed(id, agentContainers[id], agentSprites[id], agentCharNames[id], waypoints, budget, toKey, partnerPlannedX);
      }, staggerMs);
    }
  }
}

function chainWalkTimed(agentId, container, sprite, charName, waypoints, totalMs, toKey, partnerPlannedX) {
  function markArrived() {
    // Snap to final destination (interior position) so agent ends inside building
    container.x = agentDestPos[agentId].x;
    container.y = agentDestPos[agentId].y;
    agentSettled[agentId] = true;
    // Face visual partner E/W on arrival
    const partner = agentVisualPartner[agentId];
    if (partner >= 0 && toKey !== 'home') {
      const partnerX = partnerPlannedX?.get(agentId) ?? agentDestPos[partner]?.x ?? container.x;
      const dir = partnerX > container.x ? 'right' : 'left';
      sprite.anims.play(`${charName}_${dir}`, false);
    }
    if (agentStatusTexts[agentId]) {
      const emoji = toKey === 'home' ? '🏠' : (LOCATIONS[toKey]?.emoji || '');
      agentStatusTexts[agentId].setText(`${agentInitials[agentId]}: ${emoji}`);
    }
    // Hide agent and brighten house when arriving home
    if (toKey === 'home') {
      container.setVisible(false);
      setHomeLit(agentId, true);
    }
  }

  if (!waypoints || waypoints.length === 0) {
    sprite.anims.play(`${charName}_down`, false);
    markArrived();
    return;
  }

  const starts = [{ x: container.x, y: container.y }, ...waypoints.slice(0, -1)];
  const dists  = waypoints.map((wp, i) => {
    const dx = wp.x - starts[i].x, dy = wp.y - starts[i].y;
    return Math.sqrt(dx * dx + dy * dy);
  });
  const totalDist = dists.reduce((s, d) => s + d, 0);

  if (totalDist < 2) {
    sprite.anims.play(`${charName}_down`, false);
    markArrived();
    return;
  }

  let wpIdx = 0;
  function walkNext() {
    if (wpIdx >= waypoints.length) {
      const prevIdx = waypoints.length - 1;
      const dx = waypoints[prevIdx].x - starts[prevIdx].x;
      const dy = waypoints[prevIdx].y - starts[prevIdx].y;
      sprite.anims.play(`${charName}_${getWalkDir(dx, dy)}`, false);
      markArrived();
      return;
    }

    const wp  = waypoints[wpIdx];
    const dx  = wp.x - container.x;
    const dy  = wp.y - container.y;
    const dist = dists[wpIdx];
    const dur  = Math.max(30, (dist / totalDist) * totalMs);
    const dir  = getWalkDir(dx, dy);

    sprite.anims.play(`${charName}_${dir}_walk`, true);
    scene.tweens.add({
      targets: container, x: wp.x, y: wp.y,
      duration: dur, ease: 'Linear',
      onComplete() { wpIdx++; walkNext(); },
    });
  }
  walkNext();
}
