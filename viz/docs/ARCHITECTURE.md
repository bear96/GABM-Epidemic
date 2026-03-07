# Viz Architecture

## Overview

Phaser 3.55.2 WebGL application. Seven JavaScript files loaded as browser
globals (no bundler). All inter-file communication is through global functions
and variables.

## File Responsibilities

### map-layout.js — The World Definition
Defines everything about the physical world. No Phaser code here.

- **Constants**: MAP_TILES_W/H (100x100), TILE (32px), colors
- **BUILDINGS**: 13 buildings with `tx/ty` center, `parts` (floor plan rectangles),
  optional `rooms`, `doors`, `furniture`
- **HOME_POSITIONS**: 100 home plots evenly spaced around circle perimeter
- **buildTerrainGrid()**: GRASS/DIRT grid based on circle distance
- **buildCollisionGrid()**: Walkability grid (dirt=walkable, buildings=blocked, homes=walkable)
- **WALL_EDGES**: Set of "x1,y1>x2,y2" strings for wall-blocked tile transitions
- **BUILDING_WALLS**: Precomputed wall edges per building (for rendering)
- **computeBuildingWalls()**: Generates wall edges from parts/rooms, filters out door gaps

### agent-schedule.js — What Agents Do Each Day
Purely deterministic scheduling. No Phaser code.

- **getAgentDayPlan(id, step, row, contactCount)**: Returns 3 sub-steps (morning/afternoon/evening)
  with pixel positions inside buildings
- **getLocPosition(locationKey, id, step, subStep)**: Assigns agents to specific interior tiles
  within building parts (1-tile margin from walls)
- **buildBioHtml()**: Generates HTML for the agent bio panel
- **healthHex/healthCssColor**: S/I/R color utilities

### town-render.js — Drawing the World
All Phaser rendering for static elements.

- **renderTerrain()**: Draws grass background + dirt circle + edge ring
- **renderBuildings()**: Draws 3D dollhouse walls (outer/inner) + floors + labels
- **renderFurniture()**: Places furniture sprites inside buildings
- **renderHomes()**: Places house sprites around the perimeter
- **placeDecorations()**: Trees, bushes, nature border elements
- **setHomeLit(id, lit)**: Toggles house brightness (dim when agent is away)

### town-pathfinding.js — Getting From A to B
BFS pathfinding on the tile grid.

- **bfsTilePath(from, to)**: BFS with collision grid + WALL_EDGES check
- **tilePathToWaypoints(tilePath)**: Compresses to direction-change waypoints
  (emits END of each straight segment to prevent diagonal shortcuts)
- **computeWalkPath(id, from, to)**: Full pipeline: snap to walkable tiles,
  BFS, compress, cache, append destination
- **pathCache**: Map from "fromTile-toTile" to waypoints (session-scoped)
- **findNearestWalkable(tx, ty)**: Spiral search for nearest walkable tile

### town-agents.js — Moving Characters
Sprite management and movement tweens.

- **buildAgentContainers()**: Creates 100 agent containers (sprite + badge + glow + label)
- **moveAgents(animate)**: Main per-step function. Gets plan, computes path, starts tweens
- **chainWalkTimed(waypoints, budget)**: Chains Phaser tweens between waypoints
  with proportional duration per segment
- **flashNewInfections()**: Dramatic star flash + "INFECTED!" text for new cases

### town-ui.js — Interface
Charts, panels, chat bubbles.

- **buildCharts()**: Chart.js S/I/R line chart
- **showBioPanel(id)**: Agent detail panel with health, reasoning, traits
- **showChatBubble(i, j)**: Speech bubbles when agents are near each other

### town.js — Entry Point
Phaser boot and lifecycle. Ties everything together.

- **preload()**: Loads character sprites, house sprites, nature assets, furniture
- **create()**: Builds terrain/collision, renders world, creates agents, sets up camera/keyboard
- **update()**: Camera panning, proximity-based chat bubbles
- **loadSimData()**: Parses CSV data files (individual + population + contacts)
- **Playback**: goToStep(), stepBy(), togglePlay(), scheduleNextSubStep()
- **Debug overlays**: C (collision), W (walls), P (paths)

## Coordinate Systems

| System | Units | Conversion |
|--------|-------|-----------|
| Tile | 0-99 integer | `tx, ty` |
| Pixel | 0-3199 float | `tx * 32 + 16` (center) |
| Screen | viewport coords | Phaser camera transforms |

```
pixelToTile(px, py) → [Math.floor(px/32), Math.floor(py/32)]
tileToPixel(tx, ty) → { x: tx*32+16, y: ty*32+16 }
```

## Building Coordinate System

Buildings use `tx/ty` as an anchor point. Parts are defined with `dx/dy` offsets:

```
College (tx:50, ty:20):
  parts: [
    { dx:-4, dy:0, w:4, h:8 },  // left wing  -> tiles (46,20)-(49,27)
    { dx: 0, dy:0, w:4, h:4 },  // center     -> tiles (50,20)-(53,23)
    { dx: 4, dy:0, w:4, h:8 },  // right wing -> tiles (54,20)-(57,27)
  ]
```

World tile = `(b.tx + p.dx + localX, b.ty + p.dy + localY)`

## Rendering Layers (Phaser depth)

| Depth | What |
|-------|------|
| 0-5 | Terrain, dirt, edge ring |
| 6-8 | Building floors, furniture |
| 9-12 | Walls, decorations, houses |
| 13-14 | Location labels |
| 15-16 | Agent containers |
| 20-22 | Infection flash text |
| 24 | Path debug overlay |
| 25 | Collision debug overlay |
| 26 | Wall debug overlay |
