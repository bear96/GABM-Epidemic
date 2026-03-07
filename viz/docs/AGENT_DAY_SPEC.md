# A Day in the Life of a Viz Agent

This document is the single source of truth for how agent movement, pairing,
and rendering works across a simulated day. All code changes must preserve
every invariant listed here.

---

## Overview

Each simulation day (timestep) is divided into N+1 substeps:
- Rounds 0..N-1: contact meeting rounds (agents walk to buildings, meet partners)
- Round N: evening (all agents walk home)

N = the maximum number of contacts any agent has that day (typically 5-7).

---

## Data Pipeline (runs once at load time)

```
CSV files
  |
  v
contacts.csv  -->  contactPairsMap[step] = [{a, b}, ...]
                          |
                          v
                   greedy edge-coloring
                          |
                          v
                   contactRoundsMap[step][round] = [{a, b}, ...]
                   maxRoundsPerStep[step] = N
                          |
                          v
                   constraint-based location assignment
                          |
                          v
                   agentRoundLoc[step][agentId] = [locKey, locKey, ...]
```

### Round Assignment (greedy edge-coloring)

Splits the flat pair list into sequential rounds where each agent appears
in at most 1 pair per round:

```
for each round r = 0..N-1:
  busy = new Set()
  for each unassigned pair {a, b}:
    if neither a nor b in busy:
      assign pair to round r
      mark pair as assigned
```

### Location Assignment (constraint-based)

For each pair in each round, picks a building where:
1. Neither agent has been in any prior round this day
2. The building has fewer than 5 pairs assigned this round
3. Fallback: least-loaded building if no fresh candidates

Both agents in a pair are assigned the SAME building. This is critical
for the pairing/facing logic.

---

## Per-Substep Rendering (runs each time substep changes)

### Step 1: Build day plans  [agent-schedule.js: getAgentDayPlan()]

For each agent, produce an array of N+1 substeps:

```
if agent stayed home (row.location === 'home'):
  all substeps = home position

if agent went out:
  for round r = 0..N-1:
    if agentRoundLoc[step][agentId][r] exists:
      substep.locationKey = that building
      substep.x/y = getLocPosition(building, agentId, step, r)
    else:
      substep = home (agent has no partner this round, goes home early)
  final substep = home (evening)
```

### Step 2: Group agents by building  [town-agents.js: moveAgents() Pass 2]

Group all agents going to the same building this substep.

### Step 3: Build pair lookup from round data  [town-agents.js: moveAgents() Pass 2]

```
for each pair {a, b} in contactRoundsMap[step][currentSubStep]:
  if both agents' plans have the same locationKey AND it's not 'home':
    pairOf[a] = b
    pairOf[b] = a
    agentVisualPartner[a] = b
    agentVisualPartner[b] = a
```

INVARIANT: Both agents in a pair MUST have the same locationKey.
This is guaranteed by the location assignment (both get the same building).

### Step 4: Assign tile positions (stride-based)  [town-agents.js: moveAgents() Pass 2]

For each building, spread agents across interior tiles:
- Paired agents: A gets tile position, B gets tile position + 18px right
- Unpaired agents: get their own tile

This ensures pairs stand next to each other (A left, B right).

### Step 5: Build partnerPlannedX  [town-agents.js: moveAgents()]

Before starting walks, pre-compute each agent's partner's planned X:
```
partnerPlannedX[agentId] = plans[partner].subSteps[currentSubStep].x
```
This avoids the race condition where the partner hasn't arrived yet.

### Step 6: Walk agents  [town-agents.js: moveAgents() Pass 3]

For each agent:
- Compute BFS path from current position to destination
- Animate walk along waypoints over ~10s budget
- On arrival (markArrived):
  - Snap to exact destination
  - Face partner E/W using partnerPlannedX (not live position)
  - Update status text with building emoji

### FACING INVARIANT

When an agent arrives at a building and has a visual partner:
- Agent whose X < partner's X faces RIGHT
- Agent whose X > partner's X faces LEFT
- This is determined by partnerPlannedX, not live position

Since pairs are positioned as (tile.x, tile.x + 18), the left agent
always faces right and the right agent always faces left.

---

## Timing

| Value | Formula | At 1x speed |
|-------|---------|-------------|
| Walk budget per substep | 10000 / playSpeed | 10s |
| Stagger per agent | (id * 97 + step * 37) % 3000 / playSpeed | 0-3s |
| Actual walk time | budget - stagger | 7-10s |
| Auto-advance delay | subStepMs + 200ms hold | ~10.2s |

Total day at 1x with 8 substeps = ~82s. Use 2x or 4x for faster playback.

---

## Pathfinding

### Between different buildings (building A -> building B)
Uses BFS on collision grid. Building interiors are blocked.
Path: interior exit -> dirt path -> BFS route -> dirt path -> interior entry.

### Same building (partner swap within building)
Uses interior BFS on BUILDING_CELLS. Routes through interior tiles
to avoid walking through walls in non-rectangular buildings.

### Building -> Home or Home -> Building
BFS from building door tile to nearest walkable tile to home position.

---

## Chat Bubbles

During contact rounds (substep < nRounds), chat bubbles appear above
paired agents who have both settled (arrived at destination).
Only the current round's pairs trigger bubbles.

---

## Day Counter Display

Format: "Day {step} . {label} . {time}"

- Rounds: "Round 1" through "Round N", times spread 8:00 AM to 6:00 PM
- Evening: "Evening", time = 8:00 PM
- Times snap to :00 or :30

---

## Files Involved

| File | Responsibility |
|------|---------------|
| town.js | Round assignment, location assignment, playback loop, substep helpers |
| agent-schedule.js | getAgentDayPlan() — per-agent substep locations |
| town-agents.js | moveAgents() — pairing, tile assignment, walking, facing |
| town-pathfinding.js | computeWalkPath(), BFS, interior routing |
| town-ui.js | Chat bubbles, bio panel |

---

## Known Failure Modes (things that have broken before)

1. **Teleporting**: computeWalkPath returned [] for same-building moves.
   Fix: interior BFS for fromKey === toKey.

2. **Wrong facing**: markArrived() used live partner position (agentDestPos)
   before partner arrived. Fix: use pre-computed partnerPlannedX.

3. **Too-fast animation**: subStepMs scaled by 1/nSubs, giving ~3s with 8
   substeps. Fix: fixed 10s per substep regardless of count.

4. **Solo agents at buildings**: greedy first-match only paired each agent
   with 1 of their ~5 contacts. Fix: multi-round system with edge-coloring.

5. **Agents stuck at same building**: getAgentDayPlan used single row.location
   for all rounds. Fix: per-round locations from agentRoundLoc.
