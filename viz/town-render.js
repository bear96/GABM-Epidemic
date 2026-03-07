// ============================================================
// GABM Epidemic — Town Rendering  (Circle Map)
// Joon Cream dirt circle with soft feathered edge on grass.
// Compound building shapes + house sprites.
// Depends on: map-layout.js
// ============================================================

'use strict';

// ═══════════════════════════════════════════════════════════════
// TERRAIN — Joon Cream dirt circle with soft feathered edge
// ═══════════════════════════════════════════════════════════════

function renderTerrain(sc, terrainGrid) {
  const cx = CIRCLE_CENTER.tx * TILE + TILE / 2;
  const cy = CIRCLE_CENTER.ty * TILE + TILE / 2;
  const r = CIRCLE_RADIUS * TILE;
  const feather = 6 * TILE;

  // Render onto offscreen canvas for Canvas2D radial gradients
  const canvas = document.createElement('canvas');
  canvas.width = MAP_PX_W;
  canvas.height = MAP_PX_H;
  const ctx = canvas.getContext('2d');

  // 1. Grass base
  ctx.fillStyle = '#6AAF4A';
  ctx.fillRect(0, 0, MAP_PX_W, MAP_PX_H);

  // 2. Soft-edged dirt circle via radial gradient
  const grad = ctx.createRadialGradient(cx, cy, r - feather, cx, cy, r + feather * 0.3);
  grad.addColorStop(0, '#E8D9A0');
  grad.addColorStop(0.6, '#C8B878');
  grad.addColorStop(1, '#6AAF4A');
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, r + feather * 0.3 + 2, 0, Math.PI * 2);
  ctx.fill();

  // 3. Clean solid interior
  ctx.fillStyle = '#E8D9A0';
  ctx.beginPath();
  ctx.arc(cx, cy, r - feather, 0, Math.PI * 2);
  ctx.fill();

  // 4. Nature border (dense forest + themed corners)
  drawNatureBorder(ctx, sc);

  // Add as Phaser texture
  if (sc.textures.exists('terrain_canvas')) sc.textures.remove('terrain_canvas');
  sc.textures.addCanvas('terrain_canvas', canvas);
  sc.add.image(MAP_PX_W / 2, MAP_PX_H / 2, 'terrain_canvas').setDepth(0);
}


// ═══════════════════════════════════════════════════════════════
// NATURE BORDER — Dense pixel art forest around the dirt circle
// Option E: oak-heavy + dewberries, 4 themed corners
// ═══════════════════════════════════════════════════════════════

function drawNatureBorder(ctx, sc) {
  function mulberry32(a) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function getImg(key) {
    return sc.textures.exists(key) ? sc.textures.get(key).getSourceImage() : null;
  }

  function spr(name, x, y, scale) {
    var img = getImg(name);
    if (!img) return;
    ctx.drawImage(img, x - img.width * scale / 2, y - img.height * scale,
                  img.width * scale, img.height * scale);
  }

  function sprC(name, x, y, scale) {
    var img = getImg(name);
    if (!img) return;
    ctx.drawImage(img, x - img.width * scale / 2, y - img.height * scale / 2,
                  img.width * scale, img.height * scale);
  }

  var rng = mulberry32(777);
  var W = MAP_PX_W, H = MAP_PX_H;
  var CX = CIRCLE_CENTER.tx * TILE + TILE / 2;
  var CY = CIRCLE_CENTER.ty * TILE + TILE / 2;
  var CR = CIRCLE_RADIUS * TILE;

  // Tree pool: oaks x4, pines x1, dewberries x2
  var oaks  = ['nat_tree_oak_1', 'nat_tree_oak_2', 'nat_tree_oak_3'];
  var pines = ['nat_pine_1', 'nat_pine_2', 'nat_pine_3'];
  var dews  = ['nat_dew_1', 'nat_dew_2', 'nat_dew_3'];
  var POOL  = [], i;
  for (i = 0; i < 4; i++) POOL.push.apply(POOL, oaks);
  for (i = 0; i < 1; i++) POOL.push.apply(POOL, pines);
  for (i = 0; i < 2; i++) POOL.push.apply(POOL, dews);
  var DEW_SET = {};
  dews.forEach(function(d) { DEW_SET[d] = true; });

  var GRID = 34, JITTER = 14;
  var T_MIN = 0.60, T_RNG = 0.30;
  var D_MIN = 0.30, D_RNG = 0.20;
  var circleMargin = CR * 1.08;

  // Corner features (pixel coords)
  var lkX = 360, lkY = 340, lkRX = 240, lkRY = 165;
  var cabX = 2800, cabY = 380, cabClrR = 130;
  var bX = 340, bY = 2890, bClrR = 140;
  var rX = 2860, rY = 2890, rClrR = 120;

  // ── Lake water ──
  ctx.fillStyle = '#2878A8';
  ctx.beginPath(); ctx.ellipse(lkX, lkY, lkRX, lkRY, -0.12, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#3898C8';
  ctx.beginPath(); ctx.ellipse(lkX - 10, lkY - 15, lkRX * 0.68, lkRY * 0.62, -0.08, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#58B8E8';
  ctx.beginPath(); ctx.ellipse(lkX - 25, lkY - 30, lkRX * 0.32, lkRY * 0.25, 0, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#78C8E8'; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.ellipse(lkX, lkY, lkRX, lkRY, -0.12, 0, Math.PI * 2); ctx.stroke();
  sprC('nat_lily', lkX - 40, lkY - 30, 0.50);
  sprC('nat_lily', lkX + 80, lkY + 40, 0.40);
  spr('nat_cattails', lkX - lkRX - 10, lkY - 40, 0.50);
  spr('nat_cattails', lkX + lkRX + 10, lkY + 50, 0.45);
  spr('nat_cattails', lkX - lkRX + 40, lkY + lkRY - 20, 0.40);

  // ── Cabin trail ──
  var trailHW = 40;
  ctx.fillStyle = '#C4A070';
  ctx.beginPath();
  ctx.moveTo(cabX + 30, cabY + 40);
  ctx.quadraticCurveTo(3025, cabY + 15, W, cabY - 25);
  ctx.lineTo(W, cabY - 25 + trailHW * 2);
  ctx.quadraticCurveTo(3035, cabY + 15 + trailHW * 2, cabX + 30, cabY + 40 + trailHW * 2);
  ctx.closePath(); ctx.fill();
  ctx.strokeStyle = '#A08050'; ctx.lineWidth = 2; ctx.stroke();

  // ── Dense forest grid ──
  var items = [], gy, gx, jx, jy, name, isDew, scale;
  for (gy = -20; gy < H + 40; gy += GRID) {
    for (gx = -20; gx < W + 40; gx += GRID) {
      jx = gx + (rng() - 0.5) * JITTER * 2;
      jy = gy + (rng() - 0.5) * JITTER * 1.2;

      var dcx = jx - CX, dcy = jy - CY;
      if (Math.sqrt(dcx * dcx + dcy * dcy) < circleMargin) continue;

      var dlx = (jx - lkX) / (lkRX + 60), dly = (jy - lkY) / (lkRY + 50);
      if (dlx * dlx + dly * dly < 1.0) continue;

      if (Math.sqrt((jx - cabX) * (jx - cabX) + (jy - (cabY + 40)) * (jy - (cabY + 40))) < cabClrR) continue;
      if (jx > cabX && jy > cabY - 40 && jy < cabY + 40 + trailHW * 2 + 40) continue;

      if (Math.sqrt((jx - bX) * (jx - bX) + (jy - bY) * (jy - bY)) < bClrR) continue;
      if (Math.sqrt((jx - rX) * (jx - rX) + (jy - rY) * (jy - rY)) < rClrR) continue;

      name = POOL[Math.floor(rng() * POOL.length)];
      isDew = !!DEW_SET[name];
      scale = isDew ? D_MIN + rng() * D_RNG : T_MIN + rng() * T_RNG;
      items.push({ n: name, x: jx, y: jy, s: scale, d: isDew });
    }
  }

  items.sort(function(a, b) {
    if (a.d !== b.d) return a.d ? -1 : 1;
    return a.y - b.y;
  });
  for (i = 0; i < items.length; i++) {
    var it = items[i];
    if (it.d) sprC(it.n, it.x, it.y, it.s);
    else spr(it.n, it.x, it.y, it.s);
  }

  // ── Corner overlays ──

  // Cabin + props
  spr('nat_cabin', cabX, cabY, 1.10);
  sprC('nat_campfire', cabX - 40, cabY + 80, 0.70);
  sprC('nat_log_seat_1', cabX - 90, cabY + 60, 0.60);
  sprC('nat_log_seat_2', cabX - 15, cabY + 120, 0.55);
  sprC('nat_woodpile', cabX + 80, cabY + 30, 0.60);
  sprC('nat_fallen_log', cabX - 95, cabY + 130, 0.50);
  sprC('nat_mushroom', cabX - 65, cabY + 155, 0.45);

  // Berry grove — dewberries packed in clearing
  for (var di = 0; di < 22; di++) {
    var ba = rng() * Math.PI * 2, bd = rng() * bClrR * 0.85;
    sprC(dews[Math.floor(rng() * 3)], bX + Math.cos(ba) * bd, bY + Math.sin(ba) * bd, 0.50 + rng() * 0.30);
  }
  sprC('nat_berry_basket', bX + 25, bY + 30, 0.60);
  sprC('nat_mushroom', bX - 75, bY + 60, 0.45);
  sprC('nat_mushroom', bX + 85, bY - 50, 0.40);
  sprC('nat_fallen_log', bX - 50, bY - 75, 0.55);
  sprC('nat_sign', bX + 105, bY + 75, 0.50);

  // Rocky glade — boulders + logs + mushrooms
  sprC('nat_boulder_l', rX + 12, rY - 18, 0.70);
  sprC('nat_boulder_l', rX - 45, rY + 12, 0.60);
  sprC('nat_boulder_m', rX + 50, rY + 25, 0.55);
  sprC('nat_boulder_m', rX - 18, rY - 50, 0.50);
  sprC('nat_boulder_s', rX + 70, rY - 30, 0.48);
  sprC('nat_boulder_s', rX - 65, rY + 45, 0.45);
  sprC('nat_boulder_s', rX + 25, rY + 60, 0.42);
  sprC('nat_fallen_log', rX - 70, rY - 30, 0.55);
  sprC('nat_fallen_log', rX + 55, rY + 55, 0.50);
  sprC('nat_mushroom', rX - 38, rY + 68, 0.48);
  sprC('nat_mushroom', rX + 75, rY - 12, 0.42);
  sprC('nat_mushroom', rX - 78, rY - 6, 0.38);

  // Lake shore boulders
  sprC('nat_boulder_m', lkX + lkRX + 30, lkY - 70, 0.50);
  sprC('nat_boulder_s', lkX - lkRX - 20, lkY + 95, 0.42);

  // ── Bush rings & flowers ──
  var bushes = ['nat_bush_1', 'nat_bush_2'];
  var flowers = ['nat_flowers_1', 'nat_flowers_2', 'nat_flowers_3'];

  for (i = 0; i < 10; i++) {
    var a1 = rng() * Math.PI * 2, d1 = cabClrR * (0.82 + rng() * 0.18);
    sprC(bushes[Math.floor(rng() * 2)], cabX + Math.cos(a1) * d1, cabY + 40 + Math.sin(a1) * d1, 0.35 + rng() * 0.18);
  }
  for (i = 0; i < 5; i++) {
    var a2 = rng() * Math.PI * 2, d2 = rng() * cabClrR * 0.75;
    sprC(flowers[Math.floor(rng() * 3)], cabX + Math.cos(a2) * d2, cabY + 40 + Math.sin(a2) * d2, 0.22 + rng() * 0.12);
  }

  for (i = 0; i < 8; i++) {
    var a3 = rng() * Math.PI * 2, d3 = rClrR * (0.82 + rng() * 0.18);
    sprC(bushes[Math.floor(rng() * 2)], rX + Math.cos(a3) * d3, rY + Math.sin(a3) * d3, 0.35 + rng() * 0.18);
  }
  for (i = 0; i < 4; i++) {
    var a4 = rng() * Math.PI * 2, d4 = rng() * rClrR * 0.6;
    sprC(flowers[Math.floor(rng() * 3)], rX + Math.cos(a4) * d4, rY + Math.sin(a4) * d4, 0.22 + rng() * 0.12);
  }

  for (i = 0; i < 8; i++) {
    var a5 = rng() * Math.PI * 2, d5 = rng() * bClrR * 0.8;
    sprC(flowers[Math.floor(rng() * 3)], bX + Math.cos(a5) * d5, bY + Math.sin(a5) * d5, 0.22 + rng() * 0.12);
  }
}


// ═══════════════════════════════════════════════════════════════
// BUILDINGS — compound floor plans (L, T, U shapes)
// ═══════════════════════════════════════════════════════════════

function renderBuildings(sc) {
  // Floor layer (depth 2, under walls)
  const floorGfx = sc.add.graphics().setDepth(2);
  // Wall layer (depth 3)
  const wallGfx = sc.add.graphics().setDepth(3);

  const WT = TILE * 0.22;  // wall thickness (top surface)
  const WF = TILE * 0.14;  // front face height (3D depth)
  const WO = 1.5;          // outline width

  for (const [key, b] of Object.entries(BUILDINGS)) {
    // -- Floor fills --
    const floorColor = b.floor || FLOOR_WOOD_LIGHT;
    floorGfx.fillStyle(floorColor, 1);
    for (const p of b.parts) {
      const px = (b.tx + p.dx) * TILE;
      const py = (b.ty + p.dy) * TILE;
      floorGfx.fillRect(px, py, p.w * TILE, p.h * TILE);
    }

    // Subtle plank lines on wood floors
    if (floorColor !== FLOOR_GREEN_PARK) {
      floorGfx.lineStyle(0.5, 0x000000, 0.06);
      for (const p of b.parts) {
        const px = (b.tx + p.dx) * TILE;
        const py = (b.ty + p.dy) * TILE;
        const plankH = TILE * 0.5;
        for (let ly = 0; ly < p.h * TILE; ly += plankH) {
          floorGfx.beginPath();
          floorGfx.moveTo(px, py + ly);
          floorGfx.lineTo(px + p.w * TILE, py + ly);
          floorGfx.strokePath();
        }
      }
    }

    // -- 3D Dollhouse Walls (only render for buildings with doors) --
    if (!b.doors || b.doors.length === 0) continue;
    const edges = BUILDING_WALLS[key];
    for (let i = 0; i < edges.length; i++) {
      const edge = edges[i];
      const pal = WALL_OUTER;
      const wt = WT;
      const x = (b.tx + edge.cx) * TILE;
      const y = (b.ty + edge.cy) * TILE;

      if (edge.side === 'N') {
        wallGfx.fillStyle(pal.outline, 1);
        wallGfx.fillRect(x - WO, y - wt - WO, TILE + WO * 2, wt + WF + WO * 2);
        wallGfx.fillStyle(pal.top, 1);
        wallGfx.fillRect(x, y - wt, TILE, wt);
        wallGfx.fillStyle(pal.body, 1);
        wallGfx.fillRect(x, y, TILE, WF);
        wallGfx.fillStyle(pal.hl, 1);
        wallGfx.fillRect(x, y - wt, TILE, 1);
      }
      else if (edge.side === 'S') {
        const by = y + TILE;
        wallGfx.fillStyle(pal.outline, 1);
        wallGfx.fillRect(x - WO, by - WF - WO, TILE + WO * 2, wt + WF + WO * 2);
        wallGfx.fillStyle(pal.top, 1);
        wallGfx.fillRect(x, by, TILE, wt);
        wallGfx.fillStyle(pal.body, 1);
        wallGfx.fillRect(x, by - WF, TILE, WF);
        wallGfx.fillStyle(pal.hl, 1);
        wallGfx.fillRect(x, by, TILE, 1);
      }
      else if (edge.side === 'W') {
        wallGfx.fillStyle(pal.outline, 1);
        wallGfx.fillRect(x - wt - WO, y - WO, wt + WF + WO * 2, TILE + WO * 2);
        wallGfx.fillStyle(pal.top, 1);
        wallGfx.fillRect(x - wt, y, wt, TILE);
        wallGfx.fillStyle(pal.body, 1);
        wallGfx.fillRect(x, y, WF, TILE);
        wallGfx.fillStyle(pal.hl, 1);
        wallGfx.fillRect(x - wt, y, 1, TILE);
      }
      else if (edge.side === 'E') {
        const rx = x + TILE;
        wallGfx.fillStyle(pal.outline, 1);
        wallGfx.fillRect(rx - WF - WO, y - WO, wt + WF + WO * 2, TILE + WO * 2);
        wallGfx.fillStyle(pal.top, 1);
        wallGfx.fillRect(rx, y, wt, TILE);
        wallGfx.fillStyle(pal.body, 1);
        wallGfx.fillRect(rx - WF, y, WF, TILE);
        wallGfx.fillStyle(pal.hl, 1);
        wallGfx.fillRect(rx + wt - 1, y, 1, TILE);
      }
    }

    // -- Door gaps: clear wall area and draw frame --
    const doors = b.doors || [];
    for (const d of doors) {
      const floorColor = b.floor || FLOOR_WOOD_LIGHT;
      if (d.y1 === d.y2) {
        // Horizontal door
        const dx = (b.tx + d.x1) * TILE;
        const dw = (d.x2 - d.x1) * TILE;
        const dy = (b.ty + d.y1) * TILE;
        // Clear wall zone with floor color
        floorGfx.fillStyle(floorColor, 1);
        floorGfx.fillRect(dx, dy - WT - WF, dw, WT + WF * 2 + WT);
        // Door frame top bar
        wallGfx.fillStyle(WALL_OUTER.top, 1);
        wallGfx.fillRect(dx, dy - WO, dw, WO * 2);
        // Side posts
        wallGfx.fillStyle(WALL_OUTER.body, 1);
        wallGfx.fillRect(dx - WO, dy - WT, WO * 2, WT + WF);
        wallGfx.fillRect(dx + dw - WO, dy - WT, WO * 2, WT + WF);
      } else {
        // Vertical door
        const dx = (b.tx + d.x1) * TILE;
        const dy = (b.ty + d.y1) * TILE;
        const dh = (d.y2 - d.y1) * TILE;
        // Clear wall zone with floor color
        floorGfx.fillStyle(floorColor, 1);
        floorGfx.fillRect(dx - WT - WF, dy, WT + WF * 2 + WT, dh);
        // Door frame side bar
        wallGfx.fillStyle(WALL_OUTER.top, 1);
        wallGfx.fillRect(dx - WO, dy, WO * 2, dh);
        // Top/bottom posts
        wallGfx.fillStyle(WALL_OUTER.body, 1);
        wallGfx.fillRect(dx - WT, dy - WO, WT + WF, WO * 2);
        wallGfx.fillRect(dx - WT, dy + dh - WO, WT + WF, WO * 2);
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// FURNITURE — pixel art sprites inside buildings (depth 4, above walls)
// ═══════════════════════════════════════════════════════════════

function renderFurniture(sc) {
  for (const [key, b] of Object.entries(BUILDINGS)) {
    if (!b.furniture) continue;
    for (const f of b.furniture) {
      const px = (b.tx + f.x) * TILE;
      const py = (b.ty + f.y) * TILE;
      const fw = f.w * TILE;
      const fh = f.h * TILE;

      if (sc.textures.exists(f.asset)) {
        const spr = sc.add.image(px + fw / 2, py + fh / 2, f.asset);
        spr.setDisplaySize(fw, fh);
        spr.setDepth(4);
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// TOWN DECORATIONS — landscaping sprites between buildings (depth 16, above agents)
// ═══════════════════════════════════════════════════════════════

function renderTownDecorations(sc) {
  for (const d of TOWN_DECORATIONS) {
    const px = d.tx * TILE;
    const py = d.ty * TILE;
    const fw = d.w * TILE;
    const fh = d.h * TILE;
    if (sc.textures.exists(d.asset)) {
      const spr = sc.add.image(px + fw / 2, py + fh / 2, d.asset);
      spr.setDisplaySize(fw, fh);
      spr.setDepth(16);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// HOME HOUSES — pixel art sprites around the circle perimeter
// ═══════════════════════════════════════════════════════════════

const homeSprites = [];
const HOUSE_SCALE = 0.95;

function renderHomes(sc) {
  for (let i = 0; i < HOME_POSITIONS.length; i++) {
    const h = HOME_POSITIONS[i];
    const hh = HOME_HOUSES[i];
    const textureKey = 'house_' + hh.folder + '_' + hh.sprite;
    const px = h.tx * TILE + TILE / 2;
    const py = h.ty * TILE + TILE / 2;

    const sprite = sc.add.image(px, py, textureKey);
    sprite.setScale(HOUSE_SCALE);
    sprite.setOrigin(0.5, 0.7);
    sprite.setDepth(5);

    homeSprites.push(sprite);
  }
}

function setHomeLit(agentId, lit) {
  const sprite = homeSprites[agentId];
  if (!sprite) return;
  if (lit) {
    sprite.setAlpha(1.0);
    sprite.clearTint();
  } else {
    sprite.setAlpha(0.4);
    sprite.setTint(0x888888);
  }
}

// ═══════════════════════════════════════════════════════════════
// DECORATIONS
// ═══════════════════════════════════════════════════════════════

function placeDecorations(sc, terrainGrid) {
  // Decorations are drawn on the terrain canvas in renderTerrain
}

// ═══════════════════════════════════════════════════════════════
// LOCATION LABELS
// ═══════════════════════════════════════════════════════════════

function drawLocationLabels(sc) {
  for (const [key, b] of Object.entries(BUILDINGS)) {
    const labelY = (b.ty + b._minDy) * TILE - 8;

    sc.add.text(b.tx * TILE + TILE / 2, labelY, b.emoji + ' ' + b.name, {
      fontSize: '7px',
      fontFamily: '"Press Start 2P", monospace',
      color: '#fffde8',
      stroke: '#000000',
      strokeThickness: 3,
      backgroundColor: '#00000088',
      padding: { x: 5, y: 3 },
    }).setOrigin(0.5, 1).setDepth(17).setAlpha(0.9);
  }
}
