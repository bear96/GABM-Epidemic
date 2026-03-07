// ============================================================
// GABM Epidemic — Pathfinding  (Dewberry Hollow)
// BFS tile pathfinding, waypoint compression, walk path computation.
// Depends on: map-layout.js
// ============================================================

'use strict';

// ── Collision maze aliases ───────────────────────────────────
const MAZE_W = MAP_TILES_W, MAZE_H = MAP_TILES_H, TILE_SZ = TILE;
let   collisionGrid = null;
const pathCache     = new Map();

// ── Coordinate conversion ────────────────────────────────────
function pixelToTile(px, py) {
  return [Math.floor(px / TILE_SZ), Math.floor(py / TILE_SZ)];
}
function tileToPixel(tx, ty) {
  return { x: tx * TILE_SZ + TILE_SZ / 2, y: ty * TILE_SZ + TILE_SZ / 2 };
}

// ── Find nearest walkable tile ───────────────────────────────
function findNearestWalkable(tx, ty) {
  if (tx >= 0 && tx < MAZE_W && ty >= 0 && ty < MAZE_H && !collisionGrid[ty][tx]) return [tx, ty];
  for (let r = 1; r < 12; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const nx = tx + dx, ny = ty + dy;
        if (nx >= 0 && nx < MAZE_W && ny >= 0 && ny < MAZE_H && !collisionGrid[ny][nx]) {
          return [nx, ny];
        }
      }
    }
  }
  return [tx, ty];
}

// ── BFS path on tile grid ────────────────────────────────────
function bfsTilePath(fromTile, toTile) {
  const [fx, fy] = fromTile, [tx, ty] = toTile;
  if (fx === tx && fy === ty) return [];

  const dist  = new Int16Array(MAZE_W * MAZE_H).fill(-1);
  const queue = [];
  let qi = 0;

  dist[fy * MAZE_W + fx] = 0;
  queue.push([fx, fy]);

  const DIRS = [[1,0],[-1,0],[0,1],[0,-1]];
  while (qi < queue.length) {
    const [cx, cy] = queue[qi++];
    if (cx === tx && cy === ty) break;
    for (const [ddx, ddy] of DIRS) {
      const nx = cx + ddx, ny = cy + ddy;
      if (nx < 0 || nx >= MAZE_W || ny < 0 || ny >= MAZE_H) continue;
      if (collisionGrid[ny][nx] || dist[ny * MAZE_W + nx] >= 0) continue;
      if (WALL_EDGES.has(cx + ',' + cy + '>' + nx + ',' + ny)) continue;
      dist[ny * MAZE_W + nx] = dist[cy * MAZE_W + cx] + 1;
      queue.push([nx, ny]);
    }
  }

  if (dist[ty * MAZE_W + tx] < 0) return [];

  const path = [];
  let cx = tx, cy = ty;
  while (cx !== fx || cy !== fy) {
    path.push([cx, cy]);
    const d = dist[cy * MAZE_W + cx];
    for (const [ddx, ddy] of DIRS) {
      const nx = cx - ddx, ny = cy - ddy;
      if (nx >= 0 && nx < MAZE_W && ny >= 0 && ny < MAZE_H && dist[ny * MAZE_W + nx] === d - 1) {
        cx = nx; cy = ny; break;
      }
    }
  }
  path.reverse();
  return path;
}

// ── Compress tile path to direction-change waypoints ─────────
// Emits the END of each straight segment (the corner tile) so that
// tween segments are always axis-aligned and never cut diagonals
// through walls.
function tilePathToWaypoints(tilePath) {
  if (tilePath.length === 0) return [];
  const wps = [];
  let lastDx = null, lastDy = null;
  for (let i = 1; i < tilePath.length; i++) {
    const dx = tilePath[i][0] - tilePath[i - 1][0];
    const dy = tilePath[i][1] - tilePath[i - 1][1];
    if (dx !== lastDx || dy !== lastDy) {
      // Direction changed — emit the end of the previous segment (tile i-1)
      wps.push(tileToPixel(tilePath[i - 1][0], tilePath[i - 1][1]));
      lastDx = dx; lastDy = dy;
    }
  }
  // Always emit the final tile
  const last = tilePath[tilePath.length - 1];
  wps.push(tileToPixel(last[0], last[1]));
  return wps;
}

// ── Walk direction from delta ────────────────────────────────
function getWalkDir(dx, dy) {
  if (Math.abs(dx) >= Math.abs(dy)) return dx >= 0 ? 'right' : 'left';
  return dy >= 0 ? 'down' : 'up';
}

// ── Interior BFS for routing inside non-rectangular buildings ──
// BFS on building interior tiles only. Returns pixel waypoints or null.
function buildingInteriorPath(buildingKey, fromPixel, toPixel) {
  const cells = BUILDING_CELLS[buildingKey];
  if (!cells) return null;
  const [fx, fy] = pixelToTile(fromPixel.x, fromPixel.y);
  const [tx, ty] = pixelToTile(toPixel.x, toPixel.y);
  // Clamp to nearest interior tile if agent pixel is slightly outside
  let startKey = fx + ',' + fy;
  if (!cells.has(startKey)) {
    for (let r = 1; r <= 3; r++) {
      let found = false;
      for (let dy = -r; dy <= r && !found; dy++)
        for (let dx = -r; dx <= r && !found; dx++) {
          const k = (fx+dx) + ',' + (fy+dy);
          if (cells.has(k)) { startKey = k; found = true; }
        }
      if (found) break;
    }
  }
  let endKey = tx + ',' + ty;
  if (!cells.has(endKey)) {
    for (let r = 1; r <= 3; r++) {
      let found = false;
      for (let dy = -r; dy <= r && !found; dy++)
        for (let dx = -r; dx <= r && !found; dx++) {
          const k = (tx+dx) + ',' + (ty+dy);
          if (cells.has(k)) { endKey = k; found = true; }
        }
      if (found) break;
    }
  }
  const [sx, sy] = startKey.split(',').map(Number);
  const [ex, ey] = endKey.split(',').map(Number);
  if (sx === ex && sy === ey) return [fromPixel, toPixel];
  // BFS on interior cells
  const visited = new Map();
  visited.set(startKey, null);
  const queue = [[sx, sy]];
  let qi = 0;
  const DIRS = [[1,0],[-1,0],[0,1],[0,-1]];
  while (qi < queue.length) {
    const [cx, cy] = queue[qi++];
    if (cx === ex && cy === ey) break;
    for (const [ddx, ddy] of DIRS) {
      const nx = cx + ddx, ny = cy + ddy;
      const nk = nx + ',' + ny;
      if (cells.has(nk) && !visited.has(nk)) {
        visited.set(nk, cx + ',' + cy);
        queue.push([nx, ny]);
      }
    }
  }
  if (!visited.has(endKey)) return null;
  // Reconstruct tile path
  const tilePath = [];
  let cur = endKey;
  while (cur !== null) {
    const [px, py] = cur.split(',').map(Number);
    tilePath.unshift([px, py]);
    cur = visited.get(cur);
  }
  // Compress to waypoints (direction changes only)
  const wps = [fromPixel];
  if (tilePath.length > 1) {
    const compressed = tilePathToWaypoints(tilePath);
    for (const wp of compressed) wps.push(wp);
  }
  wps.push(toPixel);
  return wps;
}

// ── Compute full walk path between locations ─────────────────
function computeWalkPath(agentId, fromKey, fromPos, toKey, toPos) {
  if (fromKey === toKey) {
    // Same location — use interior BFS for within-building partner swaps
    if (fromKey !== 'home' && BUILDING_CELLS[fromKey]) {
      const interior = buildingInteriorPath(fromKey, fromPos, toPos);
      if (interior && interior.length > 1) return interior;
    }
    return [toPos];  // fallback: direct walk
  }

  if (collisionGrid) {
    // Use building door tiles for buildings, spiral search fallback for homes
    let walkFrom;
    if (fromKey !== 'home' && BUILDING_DOOR_TILES[fromKey]) {
      walkFrom = BUILDING_DOOR_TILES[fromKey];
    } else {
      const fromTile = pixelToTile(fromPos.x, fromPos.y);
      walkFrom = findNearestWalkable(fromTile[0], fromTile[1]);
    }
    let walkTo;
    if (toKey !== 'home' && BUILDING_DOOR_TILES[toKey]) {
      walkTo = BUILDING_DOOR_TILES[toKey];
    } else {
      const toTile = pixelToTile(toPos.x, toPos.y);
      walkTo = findNearestWalkable(toTile[0], toTile[1]);
    }
    const cacheKey = `${walkFrom[0]},${walkFrom[1]}-${walkTo[0]},${walkTo[1]}`;

    if (!pathCache.has(cacheKey)) {
      const tilePath  = bfsTilePath(walkFrom, walkTo);
      const waypoints = tilePathToWaypoints(tilePath);
      pathCache.set(cacheKey, waypoints);
    }

    const waypoints = pathCache.get(cacheKey).slice();
    if (waypoints.length > 0) {
      // Interior BFS for exit: route from interior position to door tile pixel
      if (fromKey !== 'home' && BUILDING_CELLS[fromKey]) {
        const doorPixel = tileToPixel(walkFrom[0], walkFrom[1]);
        const exitPath = buildingInteriorPath(fromKey, fromPos, doorPixel);
        if (exitPath && exitPath.length > 2) {
          // Insert interior waypoints (skip first=fromPos and last=doorPixel, those are in the chain)
          for (let i = 0; i < exitPath.length - 1; i++) waypoints.unshift(exitPath[exitPath.length - 2 - i]);
        } else {
          waypoints.unshift(fromPos);
        }
      } else {
        waypoints.unshift(fromPos);
      }
      // Interior BFS for entry: route from door tile pixel to interior position
      if (toKey !== 'home' && BUILDING_CELLS[toKey]) {
        const doorPixel = tileToPixel(walkTo[0], walkTo[1]);
        const entryPath = buildingInteriorPath(toKey, doorPixel, toPos);
        if (entryPath && entryPath.length > 2) {
          // Append interior waypoints (skip first=doorPixel already in chain)
          for (let i = 1; i < entryPath.length; i++) waypoints.push(entryPath[i]);
        } else {
          waypoints.push(toPos);
        }
      } else {
        waypoints.push(toPos);
      }
      return waypoints;
    }
  }

  console.warn('[PATH FALLBACK] agent ' + agentId + ': ' + fromKey + ' → ' + toKey + ' — BFS found no path, teleporting');
  return [toPos];  // fallback direct
}
