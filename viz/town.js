// ============================================================
// GABM Epidemic — Town.js  (Circle Map)
// Slim entry point: Phaser boot, lifecycle, data loading, playback.
//
// Depends on (loaded before this file):
//   papaparse.min.js   -> Papa
//   config.js           -> CONFIG
//   phaser.min.js       -> Phaser
//   map-layout.js       -> BUILDINGS, HOME_POSITIONS, LOCATIONS, LOC_KEYS,
//                          getHomePosT, buildTerrainGrid, buildCollisionGrid,
//                          MAP_TILES_W/H, TILE, MAP_PX_W/H, TERRAIN,
//                          CIRCLE_CENTER, CIRCLE_RADIUS, GRASS_COLOR
//   agent-schedule.js   -> getAgentDayPlan, buildBioHtml, healthHex, healthCssColor
//   town-render.js      -> renderTerrain, renderBuildings, renderHomes,
//                          placeDecorations, drawLocationLabels, setHomeLit,
//                          homeSprites
//   town-pathfinding.js -> collisionGrid (let), pathCache, computeWalkPath,
//                          pixelToTile, tileToPixel, findNearestWalkable
//   town-agents.js      -> CHAR_NAMES, agentContainers, agentSprites, agentBadges,
//                          agentCharNames, agentStatusTexts, agentInitials,
//                          agentLocKey, agentDestPos, agentSettled,
//                          createAnimations, buildAgentContainers,
//                          updateBadge, updateGlow, moveAgents, killAgentTweens,
//                          flashNewInfections
//   town-ui.js          -> chatCooldown, activeBubbles, convoCooldown,
//                          showChatBubble, destroyBubblesForAgent,
//                          onCanvasMouseMove, showBioPanel,
//                          buildCharts, updateChartPlayheads
// ============================================================

'use strict';

// --- Viewport constants ---
const VIEW_W = 960;
const VIEW_H = 640;
const DEFAULT_ZOOM = 0.35;
const MIN_ZOOM = Math.max(VIEW_W / MAP_PX_W, VIEW_H / MAP_PX_H);

// --- Simulation state ---
let agentRows  = {};
let popRows    = {};
let contactMap = {};
let contactPairsMap = {};  // timestep -> [{a, b}, ...]
let contactRoundsMap = {};  // timestep -> round -> [{a, b}, ...]
let maxRoundsPerStep = {};  // timestep -> max contacts any agent has
let agentRoundLoc = {};     // timestep -> agentId -> [locKey per round]
let infectedByMap   = {};  // agentId -> { infector, step }
let maxStep    = 0;

let currentStep    = 0;
let currentSubStep = 0;
let isPlaying  = false;
let playSpeed  = 1;
let playTimer  = null;

// Dynamic substep helpers (N contact rounds + 1 go-home)
function getSubStepCount(step) {
  return (maxRoundsPerStep[step] || 0) + 1;
}
function getSubStepLabel(step, subStep) {
  const nRounds = maxRoundsPerStep[step] || 0;
  if (subStep < nRounds) return 'Round ' + (subStep + 1);
  return 'Evening';
}
function getClockTime(step, subStep) {
  const nRounds = maxRoundsPerStep[step] || 0;
  if (subStep >= nRounds) return '8:00 PM';
  // Spread rounds evenly from 8:00 AM to 6:00 PM (leaving 8 PM for go-home)
  const startMin = 8 * 60;       // 8:00 AM in minutes
  const endMin   = 18 * 60;      // 6:00 PM in minutes
  const interval = nRounds > 1 ? (endMin - startMin) / (nRounds - 1) : 0;
  const totalMin = Math.round(startMin + subStep * interval);
  const h = Math.floor(totalMin / 60);
  const m = Math.round((totalMin % 60) / 30) * 30;  // snap to :00 or :30
  const ampm = h < 12 ? 'AM' : 'PM';
  const h12 = h > 12 ? h - 12 : (h === 0 ? 12 : h);
  return h12 + ':' + String(m).padStart(2, '0') + ' ' + ampm;
}

// --- Phaser scene ref ---
let scene;
let contactGfx;

// --- Camera drag state ---
let isDragging  = false;
let dragStartX  = 0, dragStartY = 0;
let camScrollX0 = 0, camScrollY0 = 0;

// --- Camera follow state ---
// lockedAgentId is declared in town-ui.js (shared global)
const FOLLOW_ZOOM = 0.85;

// --- Per-agent contact counts ---
const agentContactCounts = {};

// --- Arrow-key pan state ---
const keysDown = new Set();

// ===============================================================
// PHASER BOOT
// ===============================================================

const game = window.__game = new Phaser.Game({
  type:            Phaser.WEBGL,
  width:           VIEW_W,
  height:          VIEW_H,
  backgroundColor: '#6AAF4A',  // grass green
  parent:          'game-container',
  render: { antialias: false, roundPixels: false, pixelArt: true },
  scene:           { preload, create, update },
});

// ===============================================================
// PHASER LIFECYCLE
// ===============================================================

function preload() {
  // Character sprites (reuse Park's until custom chars generated)
  CHAR_NAMES.forEach(name => {
    this.load.atlas(name, `assets/characters/${name}.png`, 'assets/characters/atlas.json');
  });

  // Chat bubble sprite
  this.load.image('bubble', 'assets/tiles/bubble.png');

  // House sprites
  const loaded = new Set();
  HOME_HOUSES.forEach(h => {
    const key = `house_${h.folder}_${h.sprite}`;
    if (!loaded.has(key)) {
      loaded.add(key);
      this.load.image(key, `assets/dewberry/houses/${h.folder}/${h.sprite}.png`);
    }
  });

  // Nature border assets
  const natAssets = {
    nat_tree_oak_1: 'assets/dewberry/trees/tree_oak_1.png',
    nat_tree_oak_2: 'assets/dewberry/trees/tree_oak_2.png',
    nat_tree_oak_3: 'assets/dewberry/trees/tree_oak_3.png',
    nat_pine_1: 'assets/dewberry/nature/pine_tree_1.png',
    nat_pine_2: 'assets/dewberry/nature/pine_tree_2.png',
    nat_pine_3: 'assets/dewberry/nature/pine_tree_3.png',
    nat_dew_1: 'assets/dewberry/nature/dewberry_bush_1.png',
    nat_dew_2: 'assets/dewberry/nature/dewberry_bush_2.png',
    nat_dew_3: 'assets/dewberry/nature/dewberry_bush_3.png',
    nat_lily: 'assets/dewberry/nature/lily_pads.png',
    nat_cattails: 'assets/dewberry/nature/cattails.png',
    nat_cabin: 'assets/dewberry/nature/cabin.png',
    nat_campfire: 'assets/dewberry/nature/campfire.png',
    nat_log_seat_1: 'assets/dewberry/nature/log_seat_1.png',
    nat_log_seat_2: 'assets/dewberry/nature/log_seat_2.png',
    nat_woodpile: 'assets/dewberry/nature/woodpile.png',
    nat_boulder_l: 'assets/dewberry/nature/boulder_large.png',
    nat_boulder_m: 'assets/dewberry/nature/boulder_medium.png',
    nat_boulder_s: 'assets/dewberry/nature/boulder_small.png',
    nat_berry_basket: 'assets/dewberry/nature/berry_basket.png',
    nat_mushroom: 'assets/dewberry/nature/mushroom_cluster.png',
    nat_fallen_log: 'assets/dewberry/nature/fallen_log.png',
    nat_sign: 'assets/dewberry/decorations/sign.png',
    nat_bush_1: 'assets/dewberry/decorations/bush_1.png',
    nat_bush_2: 'assets/dewberry/decorations/bush_2.png',
    nat_flowers_1: 'assets/dewberry/decorations/flowers_1.png',
    nat_flowers_2: 'assets/dewberry/decorations/flowers_2.png',
    nat_flowers_3: 'assets/dewberry/decorations/flowers_3.png',
  };
  for (const [key, path] of Object.entries(natAssets)) {
    this.load.image(key, path);
  }

  // Interior furniture sprites
  const intAssets = [
    'desk_student', 'chair_wooden', 'blackboard', 'podium_teacher', 'desk_teacher',
    'bookshelf_tall', 'bookshelf_wide', 'reading_table', 'globe',
    'table_round', 'table_long', 'chair_cushioned', 'sofa', 'potted_plant_indoor',
    'rug_ornate', 'counter_shop', 'cabinet_medicine', 'bed_single', 'bed_hospital',
    'stove_kitchen', 'barrel', 'crate_wooden', 'fireplace', 'piano', 'pulpit', 'pew_church', 'altar',
    'coffee_machine', 'pastry_display', 'cafe_menu_board', 'coffee_cups',
    'cafe_table_set', 'sink_kitchen', 'shelf_bottles',
    'market_shelf', 'produce_basket', 'cash_register',
    'workbench', 'tool_rack',
    'hay_bale', 'water_trough', 'pitchfork_rack', 'sack_grain', 'milk_pail',
  ];
  intAssets.forEach(name => {
    this.load.image('int_' + name, `assets/dewberry/interiors/${name}.png`);
  });
  // Alias for potted_plant (shorthand used in furniture defs)
  this.load.image('int_potted_plant', 'assets/dewberry/interiors/potted_plant_indoor.png');

  // Outdoor / decoration sprites (ext_ prefix)
  const decoAssets = [
    'garden_plot', 'garden_arch', 'watering_can', 'scarecrow',
    'pond_water', 'duck', 'picnic_blanket',
    'swing_set', 'picnic_table', 'gazebo',
    'bench_1', 'bench_2', 'flowers_1', 'flowers_2', 'flowers_3',
    'fountain', 'bush_1', 'bush_2', 'lamp_1',
  ];
  decoAssets.forEach(name => {
    this.load.image('ext_' + name, `assets/dewberry/decorations/${name}.png`);
  });
  // Nature assets (different directory)
  const natDecoAssets = ['lily_pads', 'boulder_small', 'log_seat_1'];
  natDecoAssets.forEach(name => {
    this.load.image('ext_' + name, `assets/dewberry/nature/${name}.png`);
  });
}

function create() {
  scene = this;

  // -- Build terrain and render --
  const terrainGrid = buildTerrainGrid();
  renderTerrain(scene, terrainGrid);
  renderBuildings(scene);
  renderFurniture(scene);
  renderTownDecorations(scene);
  placeDecorations(scene, terrainGrid);
  renderHomes(scene);
  drawLocationLabels(scene);

  // -- Build collision grid --
  collisionGrid = buildCollisionGrid(terrainGrid);

  // -- Agents --
  createAnimations(scene);
  buildAgentContainers(scene);
  contactGfx = scene.add.graphics().setDepth(10);

  // -- Camera --
  const cam = scene.cameras.main;
  cam.setBounds(0, 0, MAP_PX_W, MAP_PX_H);
  cam.setZoom(DEFAULT_ZOOM);
  cam.setScroll(
    MAP_PX_W / 2 - VIEW_W / 2,
    MAP_PX_H / 2 - VIEW_H / 2
  );

  // Mouse-drag pan
  scene.input.on('pointerdown', ptr => {
    isDragging  = true;
    dragStartX  = ptr.x; dragStartY = ptr.y;
    camScrollX0 = cam.scrollX; camScrollY0 = cam.scrollY;
  });
  scene.input.on('pointermove', ptr => {
    if (!isDragging) return;
    // Manual drag unlocks camera follow
    const dx = Math.abs(ptr.x - dragStartX), dy = Math.abs(ptr.y - dragStartY);
    if ((dx > 5 || dy > 5) && lockedAgentId >= 0) unlockFollow();
    cam.setScroll(
      camScrollX0 - (ptr.x - dragStartX) / cam.zoom,
      camScrollY0 - (ptr.y - dragStartY) / cam.zoom
    );
  });
  scene.input.on('pointerup', () => { isDragging = false; });
  window.addEventListener('mouseup', () => { isDragging = false; });

  // Scroll-wheel zoom
  game.canvas.addEventListener('wheel', e => {
    if (lockedAgentId >= 0) unlockFollow();
    const z = scene.cameras.main.zoom;
    scene.cameras.main.setZoom(Phaser.Math.Clamp(z - e.deltaY * 0.001, MIN_ZOOM, 2.0));
    updateZoomLabel();
    e.preventDefault();
  }, { passive: false });

  // Zoom buttons
  ['btn-zoom-in', 'btn-zoom-out', 'btn-zoom-reset'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('click', () => {
      const z = scene.cameras.main.zoom;
      if (id === 'btn-zoom-in')    scene.cameras.main.setZoom(Phaser.Math.Clamp(z * 1.25, MIN_ZOOM, 2.0));
      if (id === 'btn-zoom-out')   scene.cameras.main.setZoom(Phaser.Math.Clamp(z * 0.8,  MIN_ZOOM, 2.0));
      if (id === 'btn-zoom-reset') {
        scene.cameras.main.setZoom(DEFAULT_ZOOM);
        cam.setScroll(
          MAP_PX_W / 2 - VIEW_W / 2,
          MAP_PX_H / 2 - VIEW_H / 2
        );
      }
      updateZoomLabel();
    });
  });

  // Agent click
  game.canvas.addEventListener('mousemove', onCanvasMouseMove);
  game.canvas.addEventListener('click', e => {
    const rect = game.canvas.getBoundingClientRect();
    const sx   = (e.clientX - rect.left) * (VIEW_W / rect.width);
    const sy   = (e.clientY - rect.top)  * (VIEW_H / rect.height);
    const wp   = cam.getWorldPoint(sx, sy);
    const wx   = Math.round(wp.x); const wy = Math.round(wp.y);

    const clickR = Math.max(24, 40 / cam.zoom);  // ~40 screen px at any zoom
    let closestId = -1, closestDist = clickR * clickR;
    agentContainers.forEach((c, id) => {
      const dx = c.x - wx, dy = c.y - wy;
      const d2 = dx * dx + dy * dy;
      if (d2 < closestDist) { closestDist = d2; closestId = id; }
    });

    if (closestId >= 0) {
      lockedAgentId = closestId;
      showBioPanel(closestId);
      scene.tweens.add({
        targets: agentContainers[closestId],
        scaleX: 1.8, scaleY: 1.8,
        duration: 120, ease: 'Quad.Out', yoyo: true,
        onComplete() { agentContainers[closestId].setScale(1.0); },
      });
      // Zoom in and pan to the followed agent
      const targetZoom = Math.max(FOLLOW_ZOOM, cam.zoom);
      scene.tweens.add({
        targets: cam, zoom: targetZoom,
        duration: 400, ease: 'Quad.Out',
        onUpdate() { updateZoomLabel(); },
      });
      centerCameraOnAgent(closestId, 400);
    } else {
      unlockFollow();
    }
  });

  // Keyboard
  scene.input.keyboard.on('keydown-SPACE', togglePlay);
  scene.input.keyboard.on('keydown-ESC', () => { if (lockedAgentId >= 0) unlockFollow(); });
  scene.input.keyboard.on('keydown', e => {
    keysDown.add(e.key);
    if (e.shiftKey && e.key === 'ArrowLeft')  { pausePlay(); stepBy(-1); }
    if (e.shiftKey && e.key === 'ArrowRight') { pausePlay(); stepBy(1); }
    // Arrow pan unlocks follow
    if (!e.shiftKey && ['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key) && lockedAgentId >= 0) unlockFollow();
    if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)) e.preventDefault();
  });
  scene.input.keyboard.on('keyup', e => keysDown.delete(e.key));

  // Debug collision overlay (press 'C' to toggle)
  let collOverlay = null;
  scene.input.keyboard.on('keydown-C', () => {
    if (collOverlay) { collOverlay.destroy(); collOverlay = null; return; }
    collOverlay = scene.add.graphics().setDepth(25).setAlpha(0.35);
    for (let y = 0; y < MAP_TILES_H; y++) {
      for (let x = 0; x < MAP_TILES_W; x++) {
        collOverlay.fillStyle(collisionGrid[y][x] ? 0xFF0000 : 0x00FF00, 1);
        collOverlay.fillRect(x * TILE, y * TILE, TILE, TILE);
      }
    }
  });

  // Debug wall-edge overlay (press 'W' to toggle)
  // Red lines = blocked wall edges, green lines = door gaps
  let wallOverlay = null;
  scene.input.keyboard.on('keydown-W', () => {
    if (wallOverlay) { wallOverlay.destroy(); wallOverlay = null; return; }
    wallOverlay = scene.add.graphics().setDepth(26).setAlpha(0.8);
    const LW = 3;
    for (const [bkey, b] of Object.entries(BUILDINGS)) {
      if (!b.doors || b.doors.length === 0) continue;
      const walls = BUILDING_WALLS[bkey];
      for (const w of walls) {
        const px = (b.tx + w.cx) * TILE, py = (b.ty + w.cy) * TILE;
        wallOverlay.fillStyle(0xFF0000, 1);
        if (w.side === 'N') wallOverlay.fillRect(px, py - LW / 2, TILE, LW);
        if (w.side === 'S') wallOverlay.fillRect(px, py + TILE - LW / 2, TILE, LW);
        if (w.side === 'W') wallOverlay.fillRect(px - LW / 2, py, LW, TILE);
        if (w.side === 'E') wallOverlay.fillRect(px + TILE - LW / 2, py, LW, TILE);
      }
      for (const d of (b.doors || [])) {
        wallOverlay.fillStyle(0x00FF00, 1);
        if (d.y1 === d.y2) {
          wallOverlay.fillRect((b.tx + d.x1) * TILE, (b.ty + d.y1) * TILE - LW / 2, (d.x2 - d.x1) * TILE, LW);
        } else {
          wallOverlay.fillRect((b.tx + d.x1) * TILE - LW / 2, (b.ty + d.y1) * TILE, LW, (d.y2 - d.y1) * TILE);
        }
      }
    }
  });

  // Debug path overlay (press 'P' to toggle)
  // Shows BFS paths for all agents as colored lines
  let pathOverlay = null;
  let pathOverlayEnabled = false;
  scene.input.keyboard.on('keydown-P', () => {
    pathOverlayEnabled = !pathOverlayEnabled;
    if (!pathOverlayEnabled && pathOverlay) {
      pathOverlay.destroy(); pathOverlay = null;
    } else if (pathOverlayEnabled && !pathOverlay) {
      pathOverlay = scene.add.graphics().setDepth(24).setAlpha(0.6);
    }
    console.log('[DEBUG] Path overlay:', pathOverlayEnabled ? 'ON - advance a step to see paths' : 'OFF');
  });
  // Called once at start of moveAgents to clear and return graphics
  window._pathOverlayBegin = () => {
    if (!pathOverlayEnabled) return null;
    if (pathOverlay) pathOverlay.clear();
    else pathOverlay = scene.add.graphics().setDepth(24).setAlpha(0.6);
    return pathOverlay;
  };
  // Called per-agent to get graphics object (no clear)
  window._pathOverlayGet = () => {
    if (!pathOverlayEnabled) return null;
    return pathOverlay;
  };

  loadSimData();
}

function updateZoomLabel() {
  const el = document.getElementById('zoom-label');
  if (el) el.textContent = Math.round(scene.cameras.main.zoom * 100) + '%';
}

// -- Camera follow helpers --
function centerCameraOnAgent(id, durationMs) {
  const cam = scene.cameras.main;
  const c = agentContainers[id];
  const targetX = c.x - VIEW_W / 2;
  const targetY = c.y - VIEW_H / 2;
  if (durationMs > 0) {
    scene.tweens.add({
      targets: cam, scrollX: targetX, scrollY: targetY,
      duration: durationMs, ease: 'Quad.Out',
    });
  } else {
    cam.setScroll(targetX, targetY);
  }
}

function unlockFollow() {
  lockedAgentId = -1;
  const bioPanel = document.getElementById('bio-panel');
  if (bioPanel) bioPanel.innerHTML = '<div class="bio-placeholder">hover an agent<br>to see their<br>bio &amp; reasoning</div>';
}

// -- Update loop: camera pan + chat bubbles --
function update() {
  // Camera follow: smoothly track the locked agent every frame
  if (lockedAgentId >= 0 && scene && agentContainers[lockedAgentId]) {
    const cam = scene.cameras.main;
    const c = agentContainers[lockedAgentId];
    const targetX = c.x - VIEW_W / 2;
    const targetY = c.y - VIEW_H / 2;
    // Lerp for smooth tracking
    cam.setScroll(
      cam.scrollX + (targetX - cam.scrollX) * 0.12,
      cam.scrollY + (targetY - cam.scrollY) * 0.12
    );
    // Keep bio panel updated during follow
    showBioPanel(lockedAgentId);
  }

  if (keysDown.size > 0 && scene && lockedAgentId < 0) {
    const cam = scene.cameras.main;
    const panPx = 8 / cam.zoom;
    if (keysDown.has('ArrowLeft'))  cam.setScroll(cam.scrollX - panPx, cam.scrollY);
    if (keysDown.has('ArrowRight')) cam.setScroll(cam.scrollX + panPx, cam.scrollY);
    if (keysDown.has('ArrowUp'))    cam.setScroll(cam.scrollX, cam.scrollY - panPx);
    if (keysDown.has('ArrowDown'))  cam.setScroll(cam.scrollX, cam.scrollY + panPx);
  }

  // Contact-data-driven chat bubbles: show for current round's pairs only
  const now = Date.now();
  const nRounds = maxRoundsPerStep[currentStep] || 0;
  if (currentSubStep < nRounds) {
    const roundPairs = contactRoundsMap[currentStep]?.[currentSubStep] || [];
    for (const { a, b } of roundPairs) {
      if (!agentSettled[a] || !agentSettled[b]) continue;
      if (agentLocKey[a] === 'home' || agentLocKey[b] === 'home') continue;
      const key = `${a}-${b}-${currentSubStep}`;
      if (!chatCooldown.has(key) || now > chatCooldown.get(key)) {
        chatCooldown.set(key, now + convoCooldown(a, b));
        showChatBubble(a, b);
      }
    }
  }
}

// ===============================================================
// DATA LOADING
// ===============================================================

let csvLoaded = { indiv: false, contacts: false };
function tryInit() {
  if (csvLoaded.indiv && csvLoaded.contacts) {
    document.getElementById('loading').style.display = 'none';
    initViz();
  }
}

function loadSimData() {
  const base = CONFIG.DATA_RUN;
  const pre  = CONFIG.DATA_PREFIX;
  const suf  = CONFIG.RUN_SUFFIX;

  document.getElementById('world-name').textContent  = CONFIG.WORLD_NAME;
  document.getElementById('model-label').textContent = CONFIG.MODEL_LABEL;

  Papa.parse(`${base}/${pre}indiv_data${suf}.csv`, {
    download: true, header: true, dynamicTyping: true, skipEmptyLines: true,
    complete({ data }) {
      data.forEach(row => {
        if (!agentRows[row.agent_id]) agentRows[row.agent_id] = {};
        agentRows[row.agent_id][row.timestep] = row;
        if (row.timestep > maxStep) maxStep = row.timestep;
        // Build infectedByMap from infected_by column
        if (row.infected_by !== '' && row.infected_by != null) {
          const victim = row.agent_id;
          const infector = Number(row.infected_by);
          if (!infectedByMap[victim]) {
            infectedByMap[victim] = { infector, step: row.timestep };
          }
        }
      });
      csvLoaded.indiv = true;
      tryInit();
    },
    error() { showError('Failed to load indiv_data CSV.\nCheck config.js DATA_RUN path.'); },
  });

  Papa.parse(`${base}/${pre}population_data${suf}.csv`, {
    download: true, header: true, dynamicTyping: true, skipEmptyLines: true,
    complete({ data }) { data.forEach(row => { popRows[row.Step] = row; }); buildCharts(); },
    error() {},
  });

  Papa.parse(`${base}/${pre}contacts${suf}.csv`, {
    download: true, header: true, dynamicTyping: true, skipEmptyLines: true,
    complete({ data }) {
      // Read pre-computed round + location data directly from CSV
      data.forEach(row => {
        const t = row.timestep, r = row.round, a = row.agent_a_id, b = row.agent_b_id, loc = row.location;

        // contactMap (for infection lookup)
        if (!contactMap[t]) contactMap[t] = new Set();
        contactMap[t].add(`${Math.min(a, b)}-${Math.max(a, b)}`);

        // contactPairsMap (flat list, kept for compatibility)
        if (!contactPairsMap[t]) contactPairsMap[t] = [];
        contactPairsMap[t].push({ a, b });

        // agentContactCounts
        if (!agentContactCounts[t]) agentContactCounts[t] = {};
        agentContactCounts[t][a] = (agentContactCounts[t][a] || 0) + 1;
        agentContactCounts[t][b] = (agentContactCounts[t][b] || 0) + 1;

        // contactRoundsMap — direct from CSV (no computation needed)
        if (!contactRoundsMap[t]) contactRoundsMap[t] = [];
        if (!contactRoundsMap[t][r]) contactRoundsMap[t][r] = [];
        contactRoundsMap[t][r].push({ a, b });

        // maxRoundsPerStep
        maxRoundsPerStep[t] = Math.max(maxRoundsPerStep[t] || 0, r + 1);

        // agentRoundLoc — direct from CSV
        if (!agentRoundLoc[t]) agentRoundLoc[t] = {};
        if (!agentRoundLoc[t][a]) agentRoundLoc[t][a] = [];
        if (!agentRoundLoc[t][b]) agentRoundLoc[t][b] = [];
        agentRoundLoc[t][a][r] = loc;
        agentRoundLoc[t][b][r] = loc;
      });
      csvLoaded.contacts = true;
      tryInit();
    },
    error() {},
  });
}

function showError(msg) {
  document.getElementById('loading').style.display = 'none';
  const el = document.getElementById('error-msg');
  el.style.display = 'block';
  el.textContent   = msg;
}

// ===============================================================
// VIZ INIT
// ===============================================================

function placeAgentsAtHome() {
  for (let id = 0; id < agentContainers.length; id++) {
    const homePos = getHomePosT(id);
    agentContainers[id].x = homePos.x;
    agentContainers[id].y = homePos.y;
    agentLocKey[id]  = 'home';
    agentDestPos[id] = homePos;
    agentSettled[id] = true;
    const row  = agentRows[id]?.[0];
    const cond = row?.health_condition || 'Susceptible';
    updateBadge(id, cond);
    updateGlow(id, cond);
    agentSprites[id].anims.play(`${agentCharNames[id]}_down`, false);
    if (agentStatusTexts[id]) agentStatusTexts[id].setText(`${agentInitials[id]}: home`);
    agentContainers[id].setVisible(false); // agents start inside their houses
    setHomeLit(id, true);
  }
}

function initViz() {
  const scrubber = document.getElementById('scrubber');
  scrubber.max = maxStep;
  scrubber.addEventListener('input', () => {
    pausePlay();
    goToStep(parseInt(scrubber.value, 10), 0, false);
  });

  document.getElementById('controls').style.display     = 'flex';
  document.getElementById('scrubber-row').style.display = 'block';
  document.getElementById('hint').style.display         = 'block';

  document.getElementById('btn-play-pause').addEventListener('click', togglePlay);

  document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      playSpeed = parseFloat(btn.dataset.speed);
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      // Reschedule with new speed immediately if playing
      if (isPlaying && playTimer) {
        clearTimeout(playTimer);
        killAgentTweens();
        scheduleNextSubStep();
      }
    });
  });

  placeAgentsAtHome();

  // Update agent initials from CSV names (replaces Pipoya sprite prefix initials)
  for (let id = 0; id < agentContainers.length; id++) {
    const row = agentRows[id]?.[0];
    if (row && row.name) {
      const parts = row.name.trim().split(/\s+/);
      const ini = parts.map(p => p[0]).join('').toUpperCase().slice(0, 2);
      agentInitials[id] = ini;
    }
  }

  currentStep = 0;
  currentSubStep = 0;
  document.getElementById('scrubber').value = 0;
  updateDayCounter();
  updateSIRCounts();
  updateChartPlayheads();
}

// ===============================================================
// STEP NAVIGATION
// ===============================================================

function goToStep(step, subStep, animate) {
  currentStep    = Math.max(0, Math.min(step, maxStep));
  currentSubStep = Math.max(0, Math.min(subStep || 0, getSubStepCount(currentStep) - 1));
  document.getElementById('scrubber').value = currentStep;
  updateDayCounter();
  updateSIRCounts();
  moveAgents(animate !== false);
  drawContactLines();
  updateChartPlayheads();
  if (currentSubStep === 0) flashNewInfections(currentStep);
}

function updateDayCounter() {
  const label = getSubStepLabel(currentStep, currentSubStep);
  const time = getClockTime(currentStep, currentSubStep);
  document.getElementById('day-counter').textContent =
    `Day ${currentStep}  ·  ${label}  ·  ${time}`;
}

function updateSIRCounts() {
  const pop = popRows[currentStep];
  if (!pop) return;
  document.getElementById('sir-s').textContent = `S: ${pop.Susceptible}`;
  document.getElementById('sir-i').textContent = `I: ${pop.Infected}`;
  document.getElementById('sir-r').textContent = `R: ${pop.Recovered}`;
}

function stepBy(delta) {
  let newSub = currentSubStep + delta;
  let newStep = currentStep;
  if (newSub < 0) {
    newStep = Math.max(0, newStep - 1);
    newSub = getSubStepCount(newStep) - 1;
  } else if (newSub >= getSubStepCount(newStep)) {
    newStep = Math.min(maxStep, newStep + 1);
    newSub = 0;
  }
  goToStep(newStep, newSub, true);
}

// ===============================================================
// CONTACT LINES
// ===============================================================

function drawContactLines() {
  contactGfx.clear();
}

// Compute effective health condition including viz-only Exposed state
function getEffectiveHealth(id, step) {
  const row = agentRows[id]?.[step];
  if (!row) return 'Susceptible';
  const cond = row.health_condition;

  // Check if this agent will become infected NEXT step (i.e., they're exposed now)
  if (cond === 'Susceptible') {
    const nextRow = agentRows[id]?.[step + 1];
    if (nextRow && nextRow.health_condition === 'Infected') {
      const info = infectedByMap[id];
      if (info) return 'Exposed';
    }
  }
  return cond;
}

// ===============================================================
// PLAYBACK
// ===============================================================

function togglePlay() { isPlaying ? pausePlay() : startPlay(); }

function startPlay() {
  isPlaying = true;
  document.getElementById('btn-play-pause').textContent = '⏸ PAUSE';
  scheduleNextSubStep();
}

function pausePlay() {
  isPlaying = false;
  document.getElementById('btn-play-pause').textContent = '▶ PLAY';
  if (playTimer) { clearTimeout(playTimer); playTimer = null; }
  killAgentTweens();
}

function scheduleNextSubStep() {
  if (!isPlaying) return;
  moveAgents(true);
  drawContactLines();
  updateDayCounter();
  updateSIRCounts();
  updateChartPlayheads();

  const subStepMs = Math.max(200, Math.round(10000 / playSpeed));
  const holdMs    = Math.max(30, Math.round(2000 / playSpeed));
  playTimer = setTimeout(() => { if (isPlaying) advanceSubStep(); }, subStepMs + holdMs);
}

function advanceSubStep() {
  const maxSub = getSubStepCount(currentStep) - 1;
  if (currentSubStep < maxSub) {
    currentSubStep++;
    scheduleNextSubStep();
  } else if (currentStep < maxStep) {
    currentStep++;
    currentSubStep = 0;
    document.getElementById('scrubber').value = currentStep;
    scheduleNextSubStep();
  } else {
    pausePlay();
  }
}
