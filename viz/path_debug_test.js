#!/usr/bin/env node
/**
 * Path Debug Test — runs the same BFS logic as the viz to check
 * whether agent paths go through buildings they shouldn't.
 *
 * Usage: node viz/path_debug_test.js
 */

'use strict';

// ---- Replicate map-layout.js constants ----
const MAP_TILES_W = 100, MAP_TILES_H = 100;
const TILE = 32;
const TERRAIN = { GRASS: 0, DIRT: 1 };
const CIRCLE_CENTER = { tx: 50, ty: 50 };
const CIRCLE_RADIUS = 40;

// ---- Building definitions (exact copy from map-layout.js) ----
const BUILDINGS = {
  coliving:   { name: 'Co-Living Space', tx: 28, ty: 22,
    parts: [{ dx: 0, dy: 0, w: 6, h: 4 }, { dx: -2, dy: 4, w: 4, h: 4 }],
    doors: [{ x1: -1, y1: 8, x2: 1, y2: 8 }] },
  college:    { name: 'Ridgeway College', tx: 50, ty: 20,
    parts: [{ dx: -4, dy: 0, w: 4, h: 8 }, { dx: 0, dy: 0, w: 4, h: 4 }, { dx: 4, dy: 0, w: 4, h: 8 }],
    rooms: [
      { id: 'lectA', dx: -4, dy: 0, w: 4, h: 8 },
      { id: 'lib',   dx:  0, dy: 0, w: 4, h: 4 },
      { id: 'lectB', dx:  4, dy: 0, w: 4, h: 8 },
    ],
    doors: [
      { x1: -3, y1: 8, x2: -1, y2: 8 },
      { x1:  5, y1: 8, x2:  7, y2: 8 },
    ] },
  cafe:       { name: 'Bramble Cafe', tx: 66, ty: 22,
    parts: [{ dx: 0, dy: 0, w: 8, h: 5 }],
    doors: [{ x1: 3, y1: 5, x2: 5, y2: 5 }] },
  garden:     { name: 'Community Garden', tx: 24, ty: 38,
    parts: [{ dx: 0, dy: 0, w: 10, h: 4 }, { dx: 6, dy: 4, w: 4, h: 4 }],
    doors: [{ x1: 10, y1: 3, x2: 10, y2: 5 }] },
  postoffice: { name: 'Post Office', tx: 66, ty: 36,
    parts: [{ dx: 0, dy: 0, w: 5, h: 5 }],
    doors: [{ x1: 0, y1: 1, x2: 0, y2: 3 }] },
  uptown:     { name: 'Uptown Apts', tx: 22, ty: 54,
    parts: [{ dx: 0, dy: 0, w: 5, h: 10 }, { dx: 5, dy: 2, w: 4, h: 6 }],
    doors: [{ x1: 9, y1: 4, x2: 9, y2: 6 }] },
  pond:       { name: 'Village Green', tx: 50, ty: 50,
    parts: [{ dx: 0, dy: 0, w: 8, h: 8 }],
    doors: [{ x1: 3, y1: 0, x2: 5, y2: 0 }] },
  pub:        { name: "The Ploughman's Rest", tx: 74, ty: 50,
    parts: [{ dx: 0, dy: 0, w: 5, h: 7 }],
    doors: [{ x1: 0, y1: 2, x2: 0, y2: 4 }] },
  park:       { name: 'Lindenfield Park', tx: 34, ty: 66,
    parts: [{ dx: 0, dy: 0, w: 10, h: 5 }],
    doors: [{ x1: 4, y1: 0, x2: 6, y2: 0 }] },
  supply:     { name: 'Copper Beech Supply', tx: 56, ty: 64,
    parts: [{ dx: 0, dy: 0, w: 4, h: 6 }, { dx: 4, dy: 2, w: 4, h: 4 }],
    doors: [{ x1: 1, y1: 0, x2: 3, y2: 0 }] },
  market:     { name: 'Hawthorn Market', tx: 76, ty: 62,
    parts: [{ dx: 0, dy: 0, w: 6, h: 5 }],
    doors: [{ x1: 0, y1: 1, x2: 0, y2: 3 }] },
  farm:       { name: 'Orchard Farm', tx: 36, ty: 76,
    parts: [{ dx: 0, dy: 0, w: 8, h: 4 }, { dx: 0, dy: 4, w: 5, h: 4 }],
    doors: [{ x1: 3, y1: 0, x2: 5, y2: 0 }] },
  chapel:     { name: 'Old Chapel', tx: 62, ty: 76,
    parts: [{ dx: 1, dy: 0, w: 3, h: 3 }, { dx: 0, dy: 3, w: 5, h: 5 }],
    doors: [{ x1: 2, y1: 0, x2: 3, y2: 0 }] },
};

// ---- Wall computation (exact copy) ----
function computeBuildingWalls(b) {
  const source = b.rooms || b.parts;
  const cells = {};
  for (let pi = 0; pi < source.length; pi++) {
    const p = source[pi];
    const rid = p.id || pi;
    for (let y = p.dy; y < p.dy + p.h; y++)
      for (let x = p.dx; x < p.dx + p.w; x++)
        cells[x + ',' + y] = rid;
  }
  const edges = [];
  for (const key in cells) {
    const sp = key.split(','), cx = +sp[0], cy = +sp[1], rid = cells[key];
    const n = cells[cx + ',' + (cy - 1)];
    if (n === undefined) edges.push({ cx, cy, side: 'N', type: 'outer' });
    else if (n !== rid)  edges.push({ cx, cy, side: 'N', type: 'inner' });
    const s = cells[cx + ',' + (cy + 1)];
    if (s === undefined) edges.push({ cx, cy, side: 'S', type: 'outer' });
    const w = cells[(cx - 1) + ',' + cy];
    if (w === undefined) edges.push({ cx, cy, side: 'W', type: 'outer' });
    else if (w !== rid)  edges.push({ cx, cy, side: 'W', type: 'inner' });
    const e = cells[(cx + 1) + ',' + cy];
    if (e === undefined) edges.push({ cx, cy, side: 'E', type: 'outer' });
  }
  const doors = b.doors || [];
  const result = [];
  for (let ei = 0; ei < edges.length; ei++) {
    const e = edges[ei]; let blocked = false;
    for (const d of doors) {
      if (d.y1 === d.y2) {
        if (e.side === 'S' && e.cy + 1 === d.y1 && e.cx >= d.x1 && e.cx < d.x2) { blocked = true; break; }
        if (e.side === 'N' && e.cy === d.y1 && e.cx >= d.x1 && e.cx < d.x2) { blocked = true; break; }
      }
      if (d.x1 === d.x2) {
        if (e.side === 'E' && e.cx + 1 === d.x1 && e.cy >= d.y1 && e.cy < d.y2) { blocked = true; break; }
        if (e.side === 'W' && e.cx === d.x1 && e.cy >= d.y1 && e.cy < d.y2) { blocked = true; break; }
      }
    }
    if (!blocked) result.push(e);
  }
  return result;
}

const WALL_EDGES = new Set();
for (const [key, b] of Object.entries(BUILDINGS)) {
  if (!b.doors || b.doors.length === 0) continue;
  const walls = computeBuildingWalls(b);
  for (const w of walls) {
    const ax = b.tx + w.cx, ay = b.ty + w.cy;
    let nx, ny;
    if      (w.side === 'N') { nx = ax;     ny = ay - 1; }
    else if (w.side === 'S') { nx = ax;     ny = ay + 1; }
    else if (w.side === 'E') { nx = ax + 1; ny = ay;     }
    else                     { nx = ax - 1; ny = ay;     }
    WALL_EDGES.add(ax + ',' + ay + '>' + nx + ',' + ny);
    WALL_EDGES.add(nx + ',' + ny + '>' + ax + ',' + ay);
  }
}

// ---- Terrain + collision grid ----
function buildTerrainGrid() {
  const grid = Array.from({ length: 100 }, () => new Array(100).fill(TERRAIN.GRASS));
  const cx = CIRCLE_CENTER.tx, cy = CIRCLE_CENTER.ty, r2 = CIRCLE_RADIUS * CIRCLE_RADIUS;
  for (let y = 0; y < 100; y++)
    for (let x = 0; x < 100; x++)
      if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2)
        grid[y][x] = TERRAIN.DIRT;
  return grid;
}

const terrainGrid = buildTerrainGrid();

function buildCollisionGrid() {
  const coll = Array.from({ length: 100 }, () => new Array(100).fill(true));
  for (let y = 0; y < 100; y++)
    for (let x = 0; x < 100; x++)
      if (terrainGrid[y][x] === TERRAIN.DIRT) coll[y][x] = false;

  // FIX: building interiors are BLOCKED — BFS routes around
  for (const b of Object.values(BUILDINGS)) {
    for (const p of b.parts) {
      const sx = b.tx + p.dx, sy = b.ty + p.dy;
      for (let y = sy; y < sy + p.h; y++)
        for (let x = sx; x < sx + p.w; x++)
          if (x >= 0 && x < 100 && y >= 0 && y < 100) coll[y][x] = true;
    }
  }

  // Home tiles
  const HOME_POSITIONS = [];
  for (let i = 0; i < 100; i++) {
    const angle = (i / 100) * 2 * Math.PI - Math.PI / 2;
    const tx = Math.round(CIRCLE_CENTER.tx + CIRCLE_RADIUS * Math.cos(angle));
    const ty = Math.round(CIRCLE_CENTER.ty + CIRCLE_RADIUS * Math.sin(angle));
    HOME_POSITIONS.push({ tx, ty });
  }
  for (const h of HOME_POSITIONS) {
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const nx = h.tx + dx, ny = h.ty + dy;
        if (nx >= 0 && nx < 100 && ny >= 0 && ny < 100) coll[ny][nx] = false;
      }
  }
  return { coll, HOME_POSITIONS };
}

const { coll: collisionGrid, HOME_POSITIONS } = buildCollisionGrid();

// ---- BFS ----
function bfsTilePath(fromTile, toTile) {
  const [fx, fy] = fromTile, [tx, ty] = toTile;
  if (fx === tx && fy === ty) return [];
  const dist = new Int16Array(100 * 100).fill(-1);
  const queue = [];
  let qi = 0;
  dist[fy * 100 + fx] = 0;
  queue.push([fx, fy]);
  const DIRS = [[1,0],[-1,0],[0,1],[0,-1]];
  while (qi < queue.length) {
    const [cx, cy] = queue[qi++];
    if (cx === tx && cy === ty) break;
    for (const [ddx, ddy] of DIRS) {
      const nx = cx + ddx, ny = cy + ddy;
      if (nx < 0 || nx >= 100 || ny < 0 || ny >= 100) continue;
      if (collisionGrid[ny][nx] || dist[ny * 100 + nx] >= 0) continue;
      if (WALL_EDGES.has(cx + ',' + cy + '>' + nx + ',' + ny)) continue;
      dist[ny * 100 + nx] = dist[cy * 100 + cx] + 1;
      queue.push([nx, ny]);
    }
  }
  if (dist[ty * 100 + tx] < 0) return [];
  const path = [];
  let cx = tx, cy = ty;
  while (cx !== fx || cy !== fy) {
    path.push([cx, cy]);
    const d = dist[cy * 100 + cx];
    for (const [ddx, ddy] of DIRS) {
      const nx = cx - ddx, ny = cy - ddy;
      if (nx >= 0 && nx < 100 && ny >= 0 && ny < 100 && dist[ny * 100 + nx] === d - 1) {
        cx = nx; cy = ny; break;
      }
    }
  }
  path.reverse();
  return path;
}

function findNearestWalkable(tx, ty) {
  if (tx >= 0 && tx < 100 && ty >= 0 && ty < 100 && !collisionGrid[ty][tx]) return [tx, ty];
  for (let r = 1; r < 12; r++)
    for (let dy = -r; dy <= r; dy++)
      for (let dx = -r; dx <= r; dx++) {
        const nx = tx + dx, ny = ty + dy;
        if (nx >= 0 && nx < 100 && ny >= 0 && ny < 100 && !collisionGrid[ny][nx]) return [nx, ny];
      }
  return [tx, ty];
}

// ---- Door tile lookup (mirrors map-layout.js) ----
function getBuildingDoorTile(buildingKey) {
  const b = BUILDINGS[buildingKey];
  if (!b || !b.doors || b.doors.length === 0) return null;
  const d = b.doors[0];
  if (d.y1 === d.y2) {
    const midX = Math.floor((d.x1 + d.x2) / 2);
    const absX = b.tx + midX;
    const absYbelow = b.ty + d.y1;
    const absYabove = b.ty + d.y1 - 1;
    let belowInside = false;
    for (const p of b.parts) {
      const sx = b.tx + p.dx, sy = b.ty + p.dy;
      if (absX >= sx && absX < sx + p.w && absYbelow >= sy && absYbelow < sy + p.h) {
        belowInside = true; break;
      }
    }
    return belowInside ? [absX, absYabove] : [absX, absYbelow];
  } else {
    const midY = Math.floor((d.y1 + d.y2) / 2);
    const absY = b.ty + midY;
    const absXright = b.tx + d.x1;
    const absXleft  = b.tx + d.x1 - 1;
    let rightInside = false;
    for (const p of b.parts) {
      const sx = b.tx + p.dx, sy = b.ty + p.dy;
      if (absXright >= sx && absXright < sx + p.w && absY >= sy && absY < sy + p.h) {
        rightInside = true; break;
      }
    }
    return rightInside ? [absXleft, absY] : [absXright, absY];
  }
}

const BUILDING_DOOR_TILES = {};
for (const key of Object.keys(BUILDINGS)) {
  BUILDING_DOOR_TILES[key] = getBuildingDoorTile(key);
}

// ---- Helpers ----
function tileInBuilding(tx, ty) {
  for (const [key, b] of Object.entries(BUILDINGS)) {
    for (const p of b.parts) {
      const sx = b.tx + p.dx, sy = b.ty + p.dy;
      if (tx >= sx && tx < sx + p.w && ty >= sy && ty < sy + p.h) return key;
    }
  }
  return null;
}

function getLocPosition(locationKey, id, step, subStep) {
  const b = BUILDINGS[locationKey];
  if (!b) return null;
  const salt = 99 + (subStep || 0) * 23;
  const hash = ((id * 31 + step * 7 + salt * 53) >>> 0);
  const tiles = [];
  for (const p of b.parts) {
    const mx = Math.min(1, Math.floor(p.w / 3));
    const my = Math.min(1, Math.floor(p.h / 3));
    for (let dy = my; dy < p.h - my; dy++)
      for (let dx = mx; dx < p.w - mx; dx++)
        tiles.push([b.tx + p.dx + dx, b.ty + p.dy + dy]);
  }
  if (tiles.length === 0) return [b.tx, b.ty];
  return tiles[hash % tiles.length];
}

// ---- Tests ----
console.log('');
console.log('='.repeat(60));
console.log('  PATH DIAGNOSTIC TEST');
console.log('  Do BFS paths go through buildings?');
console.log('='.repeat(60));
console.log('');

// Stats
let totalTests = 0, totalViolations = 0;
const violationsByBuilding = {};

function testPath(fromTile, toTile, label) {
  const walkFrom = findNearestWalkable(fromTile[0], fromTile[1]);
  const walkTo = findNearestWalkable(toTile[0], toTile[1]);
  const path = bfsTilePath(walkFrom, walkTo);
  totalTests++;

  if (path.length === 0) {
    console.log(`  [NO PATH] ${label}`);
    return null;
  }

  const srcBldg = tileInBuilding(walkFrom[0], walkFrom[1]);
  const dstBldg = tileInBuilding(walkTo[0], walkTo[1]);
  const violations = {};

  for (const [tx, ty] of path) {
    const bldg = tileInBuilding(tx, ty);
    if (bldg && bldg !== srcBldg && bldg !== dstBldg) {
      if (!violations[bldg]) violations[bldg] = 0;
      violations[bldg]++;
      if (!violationsByBuilding[bldg]) violationsByBuilding[bldg] = 0;
      violationsByBuilding[bldg]++;
    }
  }

  if (Object.keys(violations).length > 0) {
    totalViolations++;
    const vList = Object.entries(violations).map(([b, n]) => `${b}(${n}tiles)`).join(', ');
    console.log(`  \x1b[31m[VIOLATION]\x1b[0m ${label}`);
    console.log(`             Passes through: ${vList}`);
  }

  return violations;
}

// Test 1: Home → Building
console.log('--- Test 1: Home → Building (agents near College) ---');
for (const id of [0, 1, 2, 3, 4, 95, 96, 97, 98, 99]) {
  const h = HOME_POSITIONS[id];
  for (const tgt of ['pub', 'market', 'farm', 'supply', 'uptown', 'chapel']) {
    const b = BUILDINGS[tgt];
    testPath([h.tx, h.ty], [b.tx, b.ty], `Home[${id}](${h.tx},${h.ty}) → ${tgt}(${b.tx},${b.ty})`);
  }
}

// Test 2: Building → Building
console.log('');
console.log('--- Test 2: Building → Building ---');
const bKeys = Object.keys(BUILDINGS);
for (let i = 0; i < bKeys.length; i++) {
  for (let j = i + 1; j < bKeys.length; j++) {
    const a = BUILDINGS[bKeys[i]], b = BUILDINGS[bKeys[j]];
    testPath([a.tx, a.ty], [b.tx, b.ty], `${bKeys[i]} → ${bKeys[j]}`);
  }
}

// Test 3: Realistic agent positions (interior tiles)
console.log('');
console.log('--- Test 3: Actual agent schedule positions (20 agents, step 5) ---');
for (let id = 0; id < 20; id++) {
  const step = 5;
  const locKeys = Object.keys(BUILDINGS);
  const mornKey = locKeys[((id * 31 + step * 7) >>> 0) % locKeys.length];
  const aftnKey = locKeys[((id * 31 + step * 7 + 11 * 53) >>> 0) % locKeys.length];
  const homeH = HOME_POSITIONS[id];
  const homeTile = [homeH.tx, homeH.ty];
  const mornTile = getLocPosition(mornKey, id, step, 0);
  const aftnTile = getLocPosition(aftnKey, id, step, 1);

  testPath(homeTile, mornTile, `Agent${id}: home → ${mornKey}`);
  testPath(mornTile, aftnTile, `Agent${id}: ${mornKey} → ${aftnKey}`);
  testPath(aftnTile, homeTile, `Agent${id}: ${aftnKey} → home`);
}

// Test 4: Door tile verification
console.log('');
console.log('--- Test 4: Door tile walkability ---');
let doorErrors = 0;
for (const [key, tile] of Object.entries(BUILDING_DOOR_TILES)) {
  if (!tile) { console.log(`  [WARN] ${key}: no door tile`); doorErrors++; continue; }
  const [tx, ty] = tile;
  if (collisionGrid[ty][tx]) {
    console.log(`  \x1b[31m[ERROR]\x1b[0m ${key} door tile (${tx},${ty}) is BLOCKED`);
    doorErrors++;
  }
  const bldg = tileInBuilding(tx, ty);
  if (bldg) {
    console.log(`  \x1b[31m[ERROR]\x1b[0m ${key} door tile (${tx},${ty}) is INSIDE ${bldg}`);
    doorErrors++;
  }
}
if (doorErrors === 0) {
  console.log(`  \x1b[32mAll 13 door tiles are walkable and outside buildings\x1b[0m`);
}

// Test 5: Door-to-door BFS paths
console.log('');
console.log('--- Test 5: Door-to-door BFS ---');
for (let i = 0; i < bKeys.length; i++) {
  for (let j = i + 1; j < bKeys.length; j++) {
    const tileA = BUILDING_DOOR_TILES[bKeys[i]];
    const tileB = BUILDING_DOOR_TILES[bKeys[j]];
    if (!tileA || !tileB) continue;
    testPath(tileA, tileB, `door:${bKeys[i]} -> door:${bKeys[j]}`);
  }
}

// Summary
console.log('');
console.log('='.repeat(60));
console.log(`  RESULTS: ${totalTests} paths tested`);
if (totalViolations > 0) {
  console.log(`  \x1b[31m${totalViolations} paths go through buildings!\x1b[0m`);
  console.log('');
  console.log('  Buildings most frequently cut through:');
  const sorted = Object.entries(violationsByBuilding).sort((a, b) => b[1] - a[1]);
  for (const [bldg, count] of sorted) {
    const hasDoors = BUILDINGS[bldg].doors && BUILDINGS[bldg].doors.length > 0;
    console.log(`    ${bldg}: ${count} tile violations ${hasDoors ? '(HAS DOORS/WALLS)' : '(NO WALLS - doorless)'}`);
  }
  console.log('');
  console.log('  \x1b[33mROOT CAUSE: Doorless building interiors are walkable in');
  console.log('  the collision grid AND have no WALL_EDGES, so BFS treats');
  console.log('  them as open space and routes straight through.\x1b[0m');
  console.log('');
  console.log('  FIX: Make building interiors BLOCKED in collision grid.');
  console.log('  Agents will pathfind around buildings on dirt paths,');
  console.log('  then smoothly enter/exit at the edges.');
} else {
  console.log('  \x1b[32mAll paths clean — no building pass-throughs!\x1b[0m');
}
console.log('='.repeat(60));
console.log('');
