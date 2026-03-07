# viz/ — CLAUDE.md

For full project context, see @../CLAUDE.md

## What This Is

Phaser 3 (WebGL) town visualization of the GABM epidemic simulation.
100 agents move between 13 buildings on a circular dirt path, with
health states (S/I/R) visualized via badges, glows, and infection flashes.

## Serving

```bash
cd "GABM 3rd paper/GABM-Epidemic"
python3 -m http.server 8000
# open http://localhost:8000/viz/town.html
```

## 7-File Architecture

All files use browser globals. Load order is set by `<script>` tags in town.html.

```
town.html
  1. map-layout.js      # Constants, buildings, homes, terrain, collision grid, WALL_EDGES
  2. agent-schedule.js   # Deterministic schedules, getAgentDayPlan(), bio HTML
  3. town-render.js      # Phaser rendering: terrain, buildings, walls, homes, furniture
  4. town-pathfinding.js # BFS pathfinding, waypoint compression, path caching
  5. town-agents.js      # Agent sprites, animations, movement tweens, infection flash
  6. town-ui.js          # Charts, bio panel, chat bubbles
  7. town.js             # Phaser boot, camera, keyboard, data loading, playback loop
```

Dependency flow:
```
map-layout.js
     |
     v
agent-schedule.js  town-render.js  town-pathfinding.js
     |                  |                  |
     +------------------+------------------+
                        |
                        v
                  town-agents.js
                        |
                        v
                    town-ui.js
                        |
                        v
                     town.js
```

## Key Data Flow: Agent Movement

```
moveAgents(animate)                        [town-agents.js]
  for each agent:
    plan = getAgentDayPlan(id, step)       [agent-schedule.js]
      -> returns subSteps with pixel positions inside buildings
    toPos = plan.subSteps[currentSubStep]
    fromPos = agent's current pixel position

    computeWalkPath(id, fromKey, fromPos, toKey, toPos)  [town-pathfinding.js]
      -> pixelToTile(fromPos)              # convert to tile coords
      -> findNearestWalkable(fromTile)     # snap to walkable dirt tile
      -> findNearestWalkable(toTile)       # snap to walkable dirt tile
      -> bfsTilePath(walkFrom, walkTo)     # BFS on collision grid + WALL_EDGES
      -> tilePathToWaypoints(tilePath)     # compress to direction-change points
      -> append toPos as final waypoint    # agent enters building interior

    chainWalkTimed(waypoints, budget)      [town-agents.js]
      -> tweens linearly between waypoints
      -> walk animation plays during movement
      -> markArrived() on completion (hides agent if home)
```

## Collision Grid Rules

Building interiors are **BLOCKED** in the collision grid. BFS routes around
buildings on dirt-only paths. Agents enter/exit buildings via the first/last
tween segment (smooth visual transition from building edge to interior position).

- Dirt circle: walkable
- Grass (outside circle): blocked
- Building interiors: **blocked** (agents pathfind around)
- Home perimeter tiles: walkable (1-tile radius)

## Debug Keys

| Key | What it does |
|-----|-------------|
| C | Toggle collision grid overlay (red=blocked, green=walkable) |
| W | Toggle wall-edge overlay (red=walls, green=doors) — College only |
| P | Toggle path overlay (colored lines showing BFS paths per agent) |
| Space | Play/pause simulation |
| Shift+Left/Right | Step backward/forward |
| Arrow keys | Pan camera |
| Scroll wheel | Zoom |

## Testing

```bash
node viz/path_debug_test.js   # Verify 0 path violations (runs 198 BFS tests)
```

## Common Mistakes (Lessons Learned)

See [docs/PATHFINDING_HISTORY.md](docs/PATHFINDING_HISTORY.md) for the full story.

**TL;DR**: Building interiors must be blocked in the collision grid. If they're
walkable, BFS takes shortcuts through buildings. This was the root cause of
agents walking through walls — not a wall-edge or waypoint issue.
