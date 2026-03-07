# Pathfinding: Bugs Found and Fixed

This documents the pathfinding issues discovered and resolved in March 2026.
Useful context for future sessions working on viz movement.

## The Problem

Agents were visually walking straight through buildings on their way to
other locations. User reported seeing agents clip through Ridgeway College
walls repeatedly.

## Root Cause (Confirmed with Diagnostic)

**Building interiors were walkable in the collision grid.**

The collision grid had three layers:
1. Dirt circle = walkable (correct)
2. Building interiors = walkable (BUG)
3. Home tiles = walkable (correct)

Because building interiors were walkable, BFS treated them as open space.
Only Ridgeway College had `WALL_EDGES` (because it has doors), so BFS could
freely route through the other 12 doorless buildings.

### Diagnostic Results (path_debug_test.js)

```
RESULTS: 198 paths tested
85 paths go through buildings!

Buildings most frequently cut through:
  pond: 128 tile violations (NO WALLS - doorless)
  supply: 93 tile violations
  uptown: 87 tile violations
  postoffice: 58 tile violations
  park: 49 tile violations
  garden: 48 tile violations
  farm: 43 tile violations
  college: 40 tile violations (HAS DOORS/WALLS)
  cafe: 30 tile violations
  ...
```

43% of all tested paths went through at least one building.

## The Fix

**One-line change in map-layout.js**: Changed building interior tiles from
`false` (walkable) to `true` (blocked) in `buildCollisionGrid()`.

```javascript
// BEFORE (bug):
if (x >= 0 && x < W && y >= 0 && y < H) coll[y][x] = false;

// AFTER (fix):
if (x >= 0 && x < W && y >= 0 && y < H) coll[y][x] = true;
```

### How Movement Works Now

1. Agent is at interior pixel position inside Building A
2. `findNearestWalkable()` snaps to nearest dirt tile just outside the building
3. BFS routes on dirt paths AROUND all buildings
4. Last waypoint is the destination's interior pixel position
5. Tween smoothly moves agent from dirt edge into building interior

This creates a natural "walk to building edge, enter building" visual.

## Other Fixes Applied in Same Session

### Waypoint Compression Diagonal Bug

**Problem**: `tilePathToWaypoints()` emitted the FIRST tile of a new direction
as a waypoint, skipping the corner tile. This created diagonal tween segments
that could visually cross walls.

**Example**: BFS path A(10,5) -> B(10,6) -> C(10,7) -> D(11,7)
- Old waypoints: B, D -> tween from B to D is diagonal
- Fixed waypoints: A, C, D -> tween from A to C (straight down), C to D (straight right)

**Fix**: Emit the END of each straight segment (tile i-1 on direction change)
instead of the start of the next segment.

### WALL_EDGES Applied to Doorless Buildings

**Problem**: Initial WALL_EDGES implementation blocked all 13 buildings'
edges. But only College has doors, so doorless buildings had every edge
blocked, making BFS unable to reach their interiors at all. This triggered
the fallback teleport for many paths.

**Fix**: Skip doorless buildings in the WALL_EDGES loop:
```javascript
if (!b.doors || b.doors.length === 0) continue;
```

This is now moot since building interiors are blocked anyway, but the
skip remains to prevent unnecessary WALL_EDGES entries.

## Testing

Run the diagnostic anytime to verify:
```bash
node viz/path_debug_test.js
```

Expected output: "198 paths tested, All paths clean"

The test checks:
- Home -> Building paths (agents near College perimeter)
- All Building -> Building pairs
- Realistic agent schedule positions (20 agents, step 5)

For each path, it verifies no tile belongs to a building that isn't the
source or destination.

## Debug Overlays

| Key | Overlay | What to look for |
|-----|---------|-----------------|
| C | Collision grid | Buildings should be RED (blocked) |
| W | Wall edges | Only College shows red walls + green doors |
| P | Path lines | Colored lines should go AROUND buildings |
