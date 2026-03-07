// ============================================================
// GABM Epidemic — Agent Schedule  (Circle Map)
// Deterministic daily schedules, bio panel, health utilities.
// Depends on: map-layout.js (LOC_KEYS, BUILDINGS, LOCATIONS, MAP_PX_W/H)
// ============================================================

'use strict';

// ── Deterministic location assignment ────────────────────────
function deterministicLoc(id, step, salt) {
  return ((id * 31 + step * 7 + salt * 53) >>> 0) % LOC_KEYS.length;
}

function getMorningLoc(id, step) {
  return LOC_KEYS[deterministicLoc(id, step, 0)];
}
function getAfternoonLoc(id, step) {
  const morning   = deterministicLoc(id, step, 0);
  const afternoon = deterministicLoc(id, step, 11);
  return LOC_KEYS[afternoon === morning ? (afternoon + 1) % LOC_KEYS.length : afternoon];
}
function getEveningLoc(id, step) {
  const morning   = deterministicLoc(id, step, 0);
  const aftnRaw   = deterministicLoc(id, step, 11);
  const afternoon = aftnRaw === morning ? (aftnRaw + 1) % LOC_KEYS.length : aftnRaw;
  let   evening   = deterministicLoc(id, step, 23);
  if (evening === morning)   evening = (evening + 1) % LOC_KEYS.length;
  if (evening === afternoon) evening = (evening + 1) % LOC_KEYS.length;
  return LOC_KEYS[evening];
}

// ── Get all interior tile positions for a building ───────────
function getBuildingTiles(locationKey) {
  const b = BUILDINGS[locationKey];
  if (!b) return [];
  const tiles = [];
  for (const p of b.parts) {
    const mx = Math.min(1, Math.floor(p.w / 3));
    const my = Math.min(1, Math.floor(p.h / 3));
    for (let dy = my; dy < p.h - my; dy++)
      for (let dx = mx; dx < p.w - mx; dx++) {
        if (b.blockedZones) {
          let blocked = false;
          for (const z of b.blockedZones) {
            if (p.dx + dx >= z.dx && p.dx + dx < z.dx + z.w &&
                p.dy + dy >= z.dy && p.dy + dy < z.dy + z.h) {
              blocked = true; break;
            }
          }
          if (blocked) continue;
        }
        tiles.push({ x: (b.tx + p.dx + dx) * TILE + TILE / 2,
                      y: (b.ty + p.dy + dy) * TILE + TILE / 2 });
      }
  }
  return tiles;
}

// ── Spread agents within a building interior ─────────────────
function getLocPosition(locationKey, id, step, subStep) {
  const b = BUILDINGS[locationKey];
  if (!b) return { x: MAP_PX_W / 2, y: MAP_PX_H / 2 };

  const salt = 99 + (subStep || 0) * 23;
  const hash = ((id * 31 + step * 7 + salt * 53) >>> 0);

  // Build flat list of interior tiles across all parts (1-tile margin from walls)
  const tiles = [];
  for (const p of b.parts) {
    const mx = Math.min(1, Math.floor(p.w / 3));
    const my = Math.min(1, Math.floor(p.h / 3));
    for (let dy = my; dy < p.h - my; dy++)
      for (let dx = mx; dx < p.w - mx; dx++) {
        // Skip tiles in blocked zones (e.g. pond water)
        if (b.blockedZones) {
          let blocked = false;
          for (const z of b.blockedZones) {
            if (p.dx + dx >= z.dx && p.dx + dx < z.dx + z.w &&
                p.dy + dy >= z.dy && p.dy + dy < z.dy + z.h) {
              blocked = true; break;
            }
          }
          if (blocked) continue;
        }
        tiles.push({ x: (b.tx + p.dx + dx) * TILE + TILE / 2,
                      y: (b.ty + p.dy + dy) * TILE + TILE / 2 });
      }
  }

  if (tiles.length === 0) {
    const p = b.parts[0];
    return { x: (b.tx + p.dx + p.w / 2) * TILE, y: (b.ty + p.dy + p.h / 2) * TILE };
  }

  const tile = tiles[hash % tiles.length];
  return { x: tile.x, y: tile.y };
}

// ── Main export: getAgentDayPlan ─────────────────────────────
function getAgentDayPlan(agentId, step, row, contactCount, nRounds) {
  const isHome  = !row || (row.location === 'home');
  const homePos = getHomePosT(agentId);
  nRounds = nRounds || 1;

  // Per-round locations from agentRoundLoc (built during round-assignment)
  const perRoundLocs = (typeof agentRoundLoc !== 'undefined' && agentRoundLoc[step])
    ? agentRoundLoc[step][agentId] : null;

  let subSteps = [];
  let primaryEmoji;

  if (isHome) {
    for (let r = 0; r <= nRounds; r++) {
      subSteps.push({ slot: r, locationKey: 'home', x: homePos.x, y: homePos.y,
                       label: r < nRounds ? 'round' + r : 'evening' });
    }
    primaryEmoji = '🏠';
  } else {
    for (let r = 0; r < nRounds; r++) {
      if (perRoundLocs && perRoundLocs[r]) {
        // Agent has a partner this round — go to assigned meeting building
        const locKey = perRoundLocs[r];
        const pos = getLocPosition(locKey, agentId, step, r);
        subSteps.push({ slot: r, locationKey: locKey, x: pos.x, y: pos.y, label: 'round' + r });
      } else {
        // No partner this round — agent goes home early
        subSteps.push({ slot: r, locationKey: 'home', x: homePos.x, y: homePos.y, label: 'round' + r });
      }
    }
    // Final substep: go home
    subSteps.push({ slot: nRounds, locationKey: 'home', x: homePos.x, y: homePos.y, label: 'evening' });
    // Use first round's location for the primary emoji
    const firstLoc = subSteps[0]?.locationKey || 'home';
    primaryEmoji = LOCATIONS[firstLoc]?.emoji || '🚶';
  }

  const bioHtml = row ? buildBioHtml(agentId, row, contactCount) : '';
  return { subSteps, emoji: primaryEmoji, bioHtml };
}

// ── Bio panel HTML ───────────────────────────────────────────
function buildBioHtml(id, row, contactCount) {
  const hCol  = row.health_condition === 'Exposed'   ? '#FFB800'
              : row.health_condition === 'Infected'  ? '#FF6B35'
              : row.health_condition === 'Recovered' ? '#8338EC' : '#4895EF';
  const hIcon = row.health_condition === 'Exposed'   ? '◆'
              : row.health_condition === 'Infected'  ? '★'
              : row.health_condition === 'Recovered' ? '■' : '●';
  const decision = row.response === 'Yes' ? '🏠 Stayed home' : '🚶 Went out';
  const reasoning = (row.reasoning || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const traits    = (row.traits || '').split(',').map(t => `<span class="trait">${t.trim()}</span>`).join(' ');
  const contacts  = contactCount != null ? `${contactCount} contact${contactCount !== 1 ? 's' : ''}` : '';

  // Extract sprite portrait as data URL from Phaser atlas
  let spriteImg = '';
  if (typeof scene !== 'undefined' && scene && agentCharNames && agentCharNames[id]) {
    const charName = agentCharNames[id];
    const frame = scene.textures.getFrame(charName, 'down');
    if (frame) {
      const cvs = document.createElement('canvas');
      cvs.width = frame.width; cvs.height = frame.height;
      const ctx = cvs.getContext('2d');
      ctx.drawImage(frame.source.image, frame.cutX, frame.cutY, frame.cutWidth, frame.cutHeight, 0, 0, frame.width, frame.height);
      spriteImg = cvs.toDataURL();
    }
  }
  const portraitHtml = spriteImg
    ? `<img src="${spriteImg}" style="width:32px;height:32px;image-rendering:pixelated;image-rendering:crisp-edges;float:right;margin:0 0 4px 8px">`
    : '';

  return `
    <div class="bio-name">${portraitHtml}${row.name || `Agent ${id}`}</div>
    <div class="bio-meta">Age ${row.age || '?'} · ${contacts}</div>
    <div class="bio-traits">${traits}</div>
    <div class="bio-health" style="color:${hCol}">${hIcon} ${row.health_condition}</div>
    <div class="bio-decision">${decision}</div>
    <div class="bio-reasoning-label">Reasoning</div>
    <div class="bio-reasoning">${reasoning || '—'}</div>
  `;
}

// ── Utility: health colour ───────────────────────────────────
function healthHex(condition) {
  return condition === 'Exposed'   ? 0xFFB800
       : condition === 'Infected'  ? 0xFF6B35
       : condition === 'Recovered' ? 0x8338EC
       : 0x4895EF;
}

function healthCssColor(condition) {
  return condition === 'Exposed'   ? '#FFB800'
       : condition === 'Infected'  ? '#FF6B35'
       : condition === 'Recovered' ? '#8338EC'
       : '#4895EF';
}
