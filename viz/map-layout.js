// ============================================================
// GABM Epidemic — Map Layout  (Circle Map)
// Simplified layout: grass field with dirt circle, buildings inside,
// home plots around the perimeter.
// ============================================================

'use strict';

// ── Map constants ────────────────────────────────────────────
const MAP_TILES_W = 100;   // tiles
const MAP_TILES_H = 100;
const TILE = 32;            // px per tile
const MAP_PX_W = MAP_TILES_W * TILE;  // 3200
const MAP_PX_H = MAP_TILES_H * TILE;  // 3200

// Terrain types (simplified — only GRASS and DIRT used)
const TERRAIN = { GRASS: 0, DIRT: 1 };

// ── Circle layout ──────────────────────────────────────────
const CIRCLE_CENTER = { tx: 50, ty: 50 };
const CIRCLE_RADIUS = 40;  // tiles

// Colors (Joon Cream palette)
const GRASS_COLOR = 0x6AAF4A;
const DIRT_COLOR  = 0xE8D9A0;
const DIRT_EDGE_COLOR = 0xC8B878;
const BLDG_COLOR  = 0x3A5A7A;  // legacy fallback
const BLDG_BORDER = 0x2A4A6A;  // legacy fallback

// 3D Dollhouse wall palette (warm brown, Joon CuteRPG style)
const WALL_OUTER  = { outline: 0x3A2820, body: 0x6B4E38, top: 0x8A6A50, hl: 0x9A7A5A };

// Floor colors (Phaser hex)
const FLOOR_WOOD_LIGHT = 0xC8AD82;
const FLOOR_WOOD_DARK  = 0xB09A74;
const FLOOR_STONE      = 0xB8AFA0;
const FLOOR_STONE_WARM = 0xC0B090;
const FLOOR_GREEN_PARK = 0x8BC06A;

// ── Building definitions ─────────────────────────────────────
// tx/ty = center tile (used for pathfinding target + label placement)
// parts = array of { dx, dy, w, h } relative to tx/ty — compound floor plans
// iw/ih = bounding box (computed below from parts)
const BUILDINGS = {
  // ── North band ──
  coliving:   { name: 'Co-Living Space',    emoji: '🎨', floor: FLOOR_WOOD_LIGHT, tx: 28, ty: 22,
    parts: [{ dx: 0, dy: 0, w: 6, h: 4 }, { dx: -2, dy: 4, w: 4, h: 4 }],
    doors: [{ x1: -1, y1: 8, x2: 1, y2: 8 }],
    furniture: [
      { asset: 'int_sofa', x: 0.3, y: 0.3, w: 1.5, h: 0.8 },
      { asset: 'int_table_round', x: 2.5, y: 0.5, w: 1.0, h: 0.8 },
      { asset: 'int_chair_cushioned', x: 2.0, y: 1.5, w: 0.5, h: 0.5 },
      { asset: 'int_chair_cushioned', x: 3.3, y: 1.5, w: 0.5, h: 0.5 },
      { asset: 'int_bookshelf_tall', x: 5.2, y: 0.15, w: 0.7, h: 0.9 },
      { asset: 'int_bookshelf_tall', x: 5.2, y: 1.2, w: 0.7, h: 0.9 },
      { asset: 'int_stove_kitchen', x: 4.0, y: 0.15, w: 1.0, h: 0.8 },
      { asset: 'int_counter_shop',  x: 4.0, y: 2.5, w: 1.8, h: 0.7 },
      { asset: 'int_bed_single', x: -1.7, y: 4.3, w: 1.2, h: 1.8 },
      { asset: 'int_desk_student', x: 0.5, y: 4.3, w: 1.0, h: 0.7 },
      { asset: 'int_chair_wooden', x: 0.7, y: 5.2, w: 0.5, h: 0.5 },
      { asset: 'int_bookshelf_tall', x: -1.7, y: 6.5, w: 0.7, h: 0.9 },
      { asset: 'int_potted_plant_indoor', x: 0.2, y: 3.3, w: 0.5, h: 0.5 },
      { asset: 'int_potted_plant_indoor', x: -1.7, y: 7.3, w: 0.5, h: 0.5 },
    ] },  // L-shape, door S
  college:    { name: 'Ridgeway College',    emoji: '🎓', floor: FLOOR_STONE, tx: 50, ty: 20,
    parts: [{ dx: -4, dy: 0, w: 4, h: 8 }, { dx: 0, dy: 0, w: 4, h: 4 }, { dx: 4, dy: 0, w: 4, h: 8 }],  // U-shape
    rooms: [
      { id: 'lectA', dx: -4, dy: 0, w: 4, h: 8 },
      { id: 'lib',   dx:  0, dy: 0, w: 4, h: 4 },
      { id: 'lectB', dx:  4, dy: 0, w: 4, h: 8 },
    ],
    furniture: [
      // Left Wing: Lecture Hall A
      { asset: 'int_blackboard',     x: -3.5, y: 0.15, w: 3.0, h: 0.5 },
      { asset: 'int_podium_teacher', x: -2.5, y: 1.0,  w: 1.0, h: 0.9 },
      { asset: 'int_desk_student', x: -3.7, y: 3.0,  w: 1.4, h: 0.7 },
      { asset: 'int_desk_student', x: -1.9, y: 3.0,  w: 1.4, h: 0.7 },
      { asset: 'int_desk_student', x: -3.7, y: 5.2,  w: 1.4, h: 0.7 },
      { asset: 'int_desk_student', x: -1.9, y: 5.2,  w: 1.4, h: 0.7 },
      { asset: 'int_chair_wooden', x: -3.3, y: 3.75, w: 0.5, h: 0.5 },
      { asset: 'int_chair_wooden', x: -1.5, y: 3.75, w: 0.5, h: 0.5 },
      { asset: 'int_chair_wooden', x: -3.3, y: 5.95, w: 0.5, h: 0.5 },
      { asset: 'int_chair_wooden', x: -1.5, y: 5.95, w: 0.5, h: 0.5 },
      // Center: Library
      { asset: 'int_bookshelf_tall', x: 0.2,  y: 0.1,  w: 1.5, h: 0.6 },
      { asset: 'int_bookshelf_tall', x: 2.3,  y: 0.1,  w: 1.5, h: 0.6 },
      { asset: 'int_reading_table',  x: 1.0,  y: 1.4,  w: 2.0, h: 1.1 },
      { asset: 'int_chair_wooden',   x: 1.5,  y: 0.9,  w: 0.4, h: 0.4 },
      { asset: 'int_chair_wooden',   x: 2.2,  y: 0.9,  w: 0.4, h: 0.4 },
      { asset: 'int_chair_wooden',   x: 1.5,  y: 2.55, w: 0.4, h: 0.4 },
      { asset: 'int_chair_wooden',   x: 2.2,  y: 2.55, w: 0.4, h: 0.4 },
      { asset: 'int_potted_plant', x: 0.15, y: 3.3, w: 0.5, h: 0.5 },
      { asset: 'int_potted_plant', x: 3.35, y: 3.3, w: 0.5, h: 0.5 },
      // Right Wing: Lecture Hall B
      { asset: 'int_blackboard',     x: 4.5,  y: 0.15, w: 3.0, h: 0.5 },
      { asset: 'int_podium_teacher', x: 5.5,  y: 1.0,  w: 1.0, h: 0.9 },
      { asset: 'int_desk_student', x: 4.3, y: 3.0,  w: 1.4, h: 0.7 },
      { asset: 'int_desk_student', x: 6.3, y: 3.0,  w: 1.4, h: 0.7 },
      { asset: 'int_desk_student', x: 4.3, y: 5.2,  w: 1.4, h: 0.7 },
      { asset: 'int_desk_student', x: 6.3, y: 5.2,  w: 1.4, h: 0.7 },
      { asset: 'int_chair_wooden', x: 4.7,  y: 3.75, w: 0.5, h: 0.5 },
      { asset: 'int_chair_wooden', x: 6.7,  y: 3.75, w: 0.5, h: 0.5 },
      { asset: 'int_chair_wooden', x: 4.7,  y: 5.95, w: 0.5, h: 0.5 },
      { asset: 'int_chair_wooden', x: 6.7,  y: 5.95, w: 0.5, h: 0.5 },
    ],
    doors: [
      { x1: -3, y1: 8, x2: -1, y2: 8 },  // left wing south
      { x1:  5, y1: 8, x2:  7, y2: 8 },   // right wing south
    ] },
  cafe:       { name: 'Bramble Cafe',          emoji: '☕', floor: FLOOR_WOOD_DARK, tx: 66, ty: 22,
    parts: [{ dx: 0, dy: 0, w: 8, h: 5 }],
    doors: [{ x1: 3, y1: 5, x2: 5, y2: 5 }],
    furniture: [
      { asset: 'int_coffee_machine', x: 0.2, y: 0.15, w: 1.0, h: 1.0 },
      { asset: 'int_sink_kitchen',   x: 1.4, y: 0.15, w: 1.0, h: 1.0 },
      { asset: 'int_shelf_bottles',  x: 2.8, y: 0.1, w: 2.0, h: 0.8 },
      { asset: 'int_pastry_display', x: 5.2, y: 0.1, w: 2.0, h: 0.8 },
      { asset: 'int_counter_shop', x: 0.5, y: 1.6, w: 4.0, h: 0.9 },
      { asset: 'int_coffee_cups',  x: 1.0, y: 1.65, w: 0.8, h: 0.6 },
      { asset: 'int_cafe_menu_board', x: 7.2, y: 0.3, w: 0.7, h: 0.7 },
      { asset: 'int_cafe_table_set', x: 0.5, y: 2.8, w: 2.0, h: 2.0 },
      { asset: 'int_cafe_table_set', x: 3.0, y: 2.8, w: 2.0, h: 2.0 },
      { asset: 'int_sofa', x: 5.8, y: 1.8, w: 2.0, h: 1.0 },
      { asset: 'int_cafe_table_set', x: 5.8, y: 3.0, w: 2.0, h: 2.0 },
      { asset: 'int_potted_plant_indoor', x: 0.1, y: 4.2, w: 0.6, h: 0.6 },
      { asset: 'int_potted_plant_indoor', x: 7.2, y: 4.2, w: 0.6, h: 0.6 },
    ] },  // wide rectangle, door S

  // ── Upper band ──
  garden:     { name: 'Community Garden',    emoji: '🌻', floor: FLOOR_GREEN_PARK, tx: 24, ty: 38,
    open: true,
    parts: [{ dx: 0, dy: 0, w: 10, h: 4 }, { dx: 6, dy: 4, w: 4, h: 4 }],
    doors: [],
    furniture: [
      { asset: 'ext_garden_arch', x: 9.2, y: 3.5, w: 1.0, h: 2.0 },
      { asset: 'ext_garden_plot', x: 0.5, y: 0.5, w: 2.0, h: 1.0 },
      { asset: 'ext_garden_plot', x: 3.0, y: 0.5, w: 2.0, h: 1.0 },
      { asset: 'ext_garden_plot', x: 5.5, y: 0.5, w: 2.0, h: 1.0 },
      { asset: 'ext_garden_plot', x: 0.5, y: 2.2, w: 2.0, h: 1.0 },
      { asset: 'ext_garden_plot', x: 3.0, y: 2.2, w: 2.0, h: 1.0 },
      { asset: 'ext_garden_plot', x: 5.5, y: 2.2, w: 2.0, h: 1.0 },
      { asset: 'ext_watering_can', x: 8.2, y: 1.5, w: 0.8, h: 0.8 },
      { asset: 'ext_bench_1', x: 8.2, y: 0.3, w: 1.2, h: 0.7 },
      { asset: 'ext_flowers_1', x: 0.2, y: 3.3, w: 0.7, h: 0.5 },
      { asset: 'ext_flowers_2', x: 2.8, y: 3.3, w: 0.7, h: 0.5 },
      { asset: 'ext_flowers_3', x: 5.2, y: 3.3, w: 0.7, h: 0.5 },
      { asset: 'ext_scarecrow', x: 7.0, y: 4.5, w: 1.0, h: 2.0 },
      { asset: 'ext_bush_1', x: 6.3, y: 4.3, w: 0.8, h: 0.8 },
      { asset: 'ext_bush_2', x: 6.3, y: 6.5, w: 0.8, h: 0.8 },
      { asset: 'ext_flowers_1', x: 8.5, y: 6.5, w: 0.7, h: 0.5 },
      { asset: 'ext_boulder_small', x: 8.5, y: 4.5, w: 0.6, h: 0.6 },
    ] },  // L-shape, open-air
  postoffice: { name: 'Post Office',         emoji: '📬', floor: FLOOR_STONE_WARM, tx: 66, ty: 36,
    parts: [{ dx: 0, dy: 0, w: 5, h: 5 }],
    doors: [{ x1: 0, y1: 1, x2: 0, y2: 3 }],
    furniture: [
      { asset: 'int_cabinet_medicine', x: 1.2, y: 0.15, w: 1.2, h: 0.6 },
      { asset: 'int_cabinet_medicine', x: 3.2, y: 0.15, w: 1.2, h: 0.6 },
      { asset: 'int_counter_shop', x: 1.2, y: 2.0, w: 2.8, h: 0.7 },
      { asset: 'int_crate_wooden', x: 3.8, y: 3.2, w: 0.8, h: 0.7 },
      { asset: 'int_crate_wooden', x: 3.8, y: 4.0, w: 0.8, h: 0.7 },
      { asset: 'int_barrel',       x: 1.5, y: 3.8, w: 0.7, h: 0.7 },
      { asset: 'int_potted_plant_indoor', x: 0.2, y: 0.15, w: 0.5, h: 0.5 },
      { asset: 'int_potted_plant_indoor', x: 0.2, y: 4.3,  w: 0.5, h: 0.5 },
    ] },  // small square, door W

  // ── Center band ──
  uptown:     { name: 'Uptown Apts',         emoji: '🏢', floor: FLOOR_WOOD_LIGHT, tx: 22, ty: 54,
    parts: [{ dx: 0, dy: 0, w: 5, h: 10 }, { dx: 5, dy: 2, w: 4, h: 6 }],
    doors: [{ x1: 9, y1: 4, x2: 9, y2: 6 }],
    furniture: [
      { asset: 'int_bed_single', x: 0.3, y: 0.3, w: 1.2, h: 1.8 },
      { asset: 'int_desk_student', x: 2.0, y: 0.3, w: 1.0, h: 0.7 },
      { asset: 'int_chair_wooden', x: 2.2, y: 1.2, w: 0.5, h: 0.5 },
      { asset: 'int_bookshelf_tall', x: 3.5, y: 0.15, w: 0.7, h: 0.9 },
      { asset: 'int_sofa', x: 0.3, y: 3.5, w: 1.5, h: 0.8 },
      { asset: 'int_table_round', x: 2.5, y: 3.8, w: 1.0, h: 0.8 },
      { asset: 'int_chair_cushioned', x: 3.5, y: 3.9, w: 0.5, h: 0.5 },
      { asset: 'int_bed_single', x: 0.3, y: 6.5, w: 1.2, h: 1.8 },
      { asset: 'int_bed_single', x: 2.0, y: 6.5, w: 1.2, h: 1.8 },
      { asset: 'int_desk_student', x: 0.3, y: 8.8, w: 1.0, h: 0.7 },
      { asset: 'int_stove_kitchen', x: 5.3, y: 2.3, w: 1.0, h: 0.8 },
      { asset: 'int_counter_shop', x: 5.3, y: 3.5, w: 3.0, h: 0.7 },
      { asset: 'int_table_long', x: 5.5, y: 5.5, w: 2.5, h: 1.0 },
      { asset: 'int_chair_cushioned', x: 5.3, y: 6.8, w: 0.5, h: 0.5 },
      { asset: 'int_chair_cushioned', x: 7.0, y: 6.8, w: 0.5, h: 0.5 },
      { asset: 'int_potted_plant_indoor', x: 4.2, y: 0.15, w: 0.5, h: 0.5 },
      { asset: 'int_potted_plant_indoor', x: 4.2, y: 9.3,  w: 0.5, h: 0.5 },
    ] },  // T-shape, door E
  pond:       { name: 'Village Green',       emoji: '💧', floor: FLOOR_GREEN_PARK, tx: 50, ty: 50,
    open: true,
    blockedZones: [{ dx: 2, dy: 2, w: 4, h: 4 }],  // central pond water — not walkable
    parts: [{ dx: 0, dy: 0, w: 8, h: 8 }],
    doors: [],
    furniture: [
      { asset: 'ext_pond_water', x: 2.0, y: 2.0, w: 4.0, h: 4.0 },
      { asset: 'ext_duck', x: 3.2, y: 3.0, w: 0.8, h: 0.8 },
      { asset: 'ext_duck', x: 3.8, y: 4.0, w: 0.8, h: 0.8 },
      { asset: 'ext_lily_pads', x: 1.8, y: 4.5, w: 1.0, h: 0.8 },
      { asset: 'ext_bench_1', x: 0.3, y: 0.3, w: 1.2, h: 0.7 },
      { asset: 'ext_bench_1', x: 6.5, y: 0.3, w: 1.2, h: 0.7 },
      { asset: 'ext_bench_2', x: 0.3, y: 7.0, w: 1.2, h: 0.7 },
      { asset: 'ext_picnic_blanket', x: 5.8, y: 5.8, w: 2.0, h: 2.0 },
      { asset: 'ext_flowers_1', x: 2.5, y: 0.3, w: 0.7, h: 0.5 },
      { asset: 'ext_flowers_2', x: 5.0, y: 0.3, w: 0.7, h: 0.5 },
      { asset: 'ext_flowers_3', x: 0.3, y: 3.5, w: 0.7, h: 0.5 },
      { asset: 'ext_flowers_1', x: 7.0, y: 3.5, w: 0.7, h: 0.5 },
      { asset: 'ext_bush_1', x: 0.3, y: 5.5, w: 0.8, h: 0.8 },
      { asset: 'ext_bush_2', x: 7.0, y: 1.5, w: 0.8, h: 0.8 },
      { asset: 'ext_lamp_1', x: 3.5, y: 0.2, w: 0.6, h: 0.8 },
      { asset: 'ext_boulder_small', x: 1.5, y: 2.5, w: 0.6, h: 0.6 },
    ] },  // square, open-air
  pub:        { name: "The Ploughman's Rest",        emoji: '🍺', floor: FLOOR_WOOD_DARK, tx: 74, ty: 50,
    parts: [{ dx: 0, dy: 0, w: 5, h: 7 }],
    doors: [{ x1: 0, y1: 2, x2: 0, y2: 4 }],
    furniture: [
      { asset: 'int_counter_shop', x: 3.0, y: 0.3, w: 1.5, h: 3.0 },
      { asset: 'int_barrel', x: 3.8, y: 3.5, w: 0.7, h: 0.7 },
      { asset: 'int_barrel', x: 3.8, y: 4.4, w: 0.7, h: 0.7 },
      { asset: 'int_fireplace', x: 0.3, y: 0.1, w: 1.2, h: 1.0 },
      { asset: 'int_table_round', x: 0.8, y: 2.0, w: 1.0, h: 0.8 },
      { asset: 'int_chair_cushioned', x: 0.1, y: 2.1, w: 0.5, h: 0.5 },
      { asset: 'int_chair_cushioned', x: 1.6, y: 2.1, w: 0.5, h: 0.5 },
      { asset: 'int_table_round', x: 0.8, y: 4.0, w: 1.0, h: 0.8 },
      { asset: 'int_chair_cushioned', x: 0.1, y: 4.1, w: 0.5, h: 0.5 },
      { asset: 'int_chair_cushioned', x: 1.6, y: 4.1, w: 0.5, h: 0.5 },
      { asset: 'int_sofa', x: 0.3, y: 5.8, w: 1.5, h: 0.8 },
      { asset: 'int_potted_plant_indoor', x: 4.3, y: 0.15, w: 0.5, h: 0.5 },
      { asset: 'int_potted_plant_indoor', x: 4.3, y: 6.3,  w: 0.5, h: 0.5 },
    ] },  // tall rectangle, door W

  // ── Lower band ──
  park:       { name: 'Lindenfield Park',        emoji: '🌳', floor: FLOOR_GREEN_PARK, tx: 34, ty: 66,
    open: true,
    parts: [{ dx: 0, dy: 0, w: 10, h: 5 }],
    doors: [],
    furniture: [
      { asset: 'ext_gazebo', x: 1.0, y: 1.0, w: 3.0, h: 3.0 },
      { asset: 'ext_fountain', x: 6.5, y: 1.5, w: 1.5, h: 1.5 },
      { asset: 'ext_swing_set', x: 4.5, y: 0.3, w: 2.0, h: 2.0 },
      { asset: 'ext_picnic_table', x: 5.0, y: 3.5, w: 2.0, h: 1.0 },
      { asset: 'ext_picnic_table', x: 8.0, y: 1.5, w: 2.0, h: 1.0 },
      { asset: 'ext_bench_1', x: 0.2, y: 0.2, w: 1.2, h: 0.7 },
      { asset: 'ext_bench_2', x: 8.5, y: 4.0, w: 1.2, h: 0.7 },
      { asset: 'ext_flowers_1', x: 0.2, y: 4.3, w: 0.7, h: 0.5 },
      { asset: 'ext_flowers_2', x: 4.5, y: 4.3, w: 0.7, h: 0.5 },
      { asset: 'ext_flowers_3', x: 9.0, y: 0.2, w: 0.7, h: 0.5 },
      { asset: 'ext_bush_1', x: 7.8, y: 3.2, w: 0.8, h: 0.8 },
      { asset: 'ext_bush_2', x: 0.2, y: 2.0, w: 0.8, h: 0.8 },
      { asset: 'ext_lamp_1', x: 4.3, y: 2.5, w: 0.6, h: 0.8 },
      { asset: 'ext_lamp_1', x: 8.5, y: 0.2, w: 0.6, h: 0.8 },
      { asset: 'ext_log_seat_1', x: 7.5, y: 4.0, w: 0.8, h: 0.6 },
    ] },  // wide rectangle, open-air
  supply:     { name: 'Copper Beech Supply',   emoji: '🔧', floor: FLOOR_STONE, tx: 56, ty: 64,
    parts: [{ dx: 0, dy: 0, w: 4, h: 6 }, { dx: 4, dy: 2, w: 4, h: 4 }],
    doors: [{ x1: 1, y1: 0, x2: 3, y2: 0 }],
    furniture: [
      { asset: 'int_counter_shop', x: 0.5, y: 1.5, w: 3.0, h: 0.7 },
      { asset: 'int_cabinet_medicine', x: 0.15, y: 3.0, w: 0.8, h: 1.2 },
      { asset: 'int_cabinet_medicine', x: 0.15, y: 4.5, w: 0.8, h: 1.2 },
      { asset: 'int_crate_wooden', x: 4.3, y: 2.3, w: 0.8, h: 0.7 },
      { asset: 'int_crate_wooden', x: 5.3, y: 2.3, w: 0.8, h: 0.7 },
      { asset: 'int_crate_wooden', x: 6.3, y: 2.3, w: 0.8, h: 0.7 },
      { asset: 'int_crate_wooden', x: 4.3, y: 3.3, w: 0.8, h: 0.7 },
      { asset: 'int_crate_wooden', x: 5.3, y: 3.3, w: 0.8, h: 0.7 },
      { asset: 'int_barrel', x: 6.5, y: 3.5, w: 0.7, h: 0.7 },
      { asset: 'int_barrel', x: 7.0, y: 4.5, w: 0.7, h: 0.7 },
      { asset: 'int_barrel', x: 4.5, y: 4.8, w: 0.7, h: 0.7 },
      { asset: 'int_workbench', x: 1.2, y: 5.0, w: 2.0, h: 0.8 },
      { asset: 'int_tool_rack', x: 3.1, y: 4.2, w: 0.7, h: 1.5 },
    ] },  // L-shape, door N
  market:     { name: 'Hawthorn Market',      emoji: '🛒', floor: FLOOR_STONE_WARM, tx: 76, ty: 62,
    parts: [{ dx: 0, dy: 0, w: 6, h: 5 }],
    doors: [{ x1: 0, y1: 1, x2: 0, y2: 3 }],
    furniture: [
      { asset: 'int_counter_shop',   x: 0.5, y: 3.5, w: 2.0, h: 0.7 },
      { asset: 'int_cash_register',  x: 0.8, y: 3.3, w: 0.7, h: 0.6 },
      { asset: 'int_market_shelf', x: 2.2, y: 0.15, w: 1.8, h: 0.7 },
      { asset: 'int_market_shelf', x: 4.2, y: 0.15, w: 1.8, h: 0.7 },
      { asset: 'int_market_shelf', x: 2.2, y: 1.5, w: 1.8, h: 0.7 },
      { asset: 'int_market_shelf', x: 4.2, y: 1.5, w: 1.8, h: 0.7 },
      { asset: 'int_produce_basket', x: 2.5, y: 2.8, w: 0.8, h: 0.8 },
      { asset: 'int_produce_basket', x: 3.5, y: 2.8, w: 0.8, h: 0.8 },
      { asset: 'int_crate_wooden',   x: 4.5, y: 2.9, w: 0.8, h: 0.7 },
      { asset: 'int_barrel',       x: 5.2, y: 3.5, w: 0.7, h: 0.7 },
      { asset: 'int_crate_wooden', x: 5.0, y: 4.2, w: 0.8, h: 0.7 },
      { asset: 'int_potted_plant_indoor', x: 0.2, y: 0.15, w: 0.5, h: 0.5 },
    ] },  // rectangle, door W

  // ── South band ──
  farm:       { name: 'Orchard Farm',        emoji: '🌾', floor: FLOOR_WOOD_LIGHT, tx: 36, ty: 76,
    parts: [{ dx: 0, dy: 0, w: 8, h: 4 }, { dx: 0, dy: 4, w: 5, h: 4 }],
    doors: [{ x1: 3, y1: 0, x2: 5, y2: 0 }],
    furniture: [
      { asset: 'int_crate_wooden', x: 0.3, y: 0.3, w: 0.8, h: 0.7 },
      { asset: 'int_crate_wooden', x: 1.3, y: 0.3, w: 0.8, h: 0.7 },
      { asset: 'int_barrel', x: 0.3, y: 1.3, w: 0.7, h: 0.7 },
      { asset: 'int_barrel', x: 1.2, y: 1.3, w: 0.7, h: 0.7 },
      { asset: 'int_counter_shop', x: 0.3, y: 2.8, w: 2.5, h: 0.7 },
      { asset: 'int_crate_wooden', x: 6.5, y: 0.3, w: 0.8, h: 0.7 },
      { asset: 'int_crate_wooden', x: 6.5, y: 1.3, w: 0.8, h: 0.7 },
      { asset: 'int_barrel', x: 7.0, y: 2.5, w: 0.7, h: 0.7 },
      { asset: 'int_pitchfork_rack', x: 7.2, y: 0.15, w: 0.7, h: 1.2 },
      { asset: 'int_stove_kitchen', x: 0.3, y: 4.2, w: 1.0, h: 0.8 },
      { asset: 'int_table_long', x: 1.8, y: 4.3, w: 2.0, h: 1.0 },
      { asset: 'int_chair_wooden', x: 1.5, y: 5.5, w: 0.5, h: 0.5 },
      { asset: 'int_chair_wooden', x: 3.2, y: 5.5, w: 0.5, h: 0.5 },
      { asset: 'int_bookshelf_tall', x: 3.8, y: 4.15, w: 0.7, h: 0.9 },
      { asset: 'int_hay_bale', x: 0.3, y: 6.3, w: 0.9, h: 0.8 },
      { asset: 'int_hay_bale', x: 1.4, y: 6.3, w: 0.9, h: 0.8 },
      { asset: 'int_water_trough', x: 0.3, y: 7.2, w: 1.5, h: 0.7 },
      { asset: 'int_sack_grain', x: 2.5, y: 6.3, w: 0.7, h: 0.7 },
      { asset: 'int_sack_grain', x: 3.4, y: 6.3, w: 0.7, h: 0.7 },
      { asset: 'int_milk_pail',  x: 2.5, y: 7.2, w: 0.6, h: 0.6 },
      { asset: 'int_potted_plant_indoor', x: 3.8, y: 7.2, w: 0.5, h: 0.5 },
    ] },  // L-shape, door N
  chapel:     { name: 'Old Chapel',          emoji: '⛪', floor: FLOOR_STONE, tx: 62, ty: 76,
    parts: [{ dx: 1, dy: 0, w: 3, h: 3 }, { dx: 0, dy: 3, w: 5, h: 5 }],
    doors: [{ x1: 2, y1: 0, x2: 3, y2: 0 }],
    furniture: [
      { asset: 'int_altar', x: 1.5, y: 7.2, w: 2.0, h: 0.6 },
      { asset: 'int_pulpit', x: 1.8, y: 1.5, w: 1.0, h: 0.8 },
      { asset: 'int_pew_church', x: 0.3, y: 3.5, w: 2.0, h: 0.6 },
      { asset: 'int_pew_church', x: 2.7, y: 3.5, w: 2.0, h: 0.6 },
      { asset: 'int_pew_church', x: 0.3, y: 4.5, w: 2.0, h: 0.6 },
      { asset: 'int_pew_church', x: 2.7, y: 4.5, w: 2.0, h: 0.6 },
      { asset: 'int_pew_church', x: 0.3, y: 5.5, w: 2.0, h: 0.6 },
      { asset: 'int_pew_church', x: 2.7, y: 5.5, w: 2.0, h: 0.6 },
      { asset: 'int_potted_plant_indoor', x: 0.2, y: 7.2, w: 0.5, h: 0.5 },
      { asset: 'int_potted_plant_indoor', x: 4.3, y: 7.2, w: 0.5, h: 0.5 },
    ] },  // cross-ish, door N
};

// ── Town decorations — landscaping between buildings ───────────
// Absolute tile coordinates. Rendered by renderTownDecorations().
const TOWN_DECORATIONS = [
  // ~24 trees scattered across the dirt circle (~16 oaks, ~8 pines)
  // -- North --
  { asset: 'nat_tree_oak_1',  tx: 38, ty: 21, w: 2.5, h: 3.0 },   // between Co-Living & College
  { asset: 'nat_pine_1',      tx: 43, ty: 19, w: 2.0, h: 3.0 },   // north of College
  { asset: 'nat_tree_oak_3',  tx: 60, ty: 22, w: 2.5, h: 3.0 },   // cafe terrace
  // -- West --
  { asset: 'nat_pine_1',      tx: 24, ty: 31, w: 2.0, h: 3.0 },   // west meadow upper
  { asset: 'nat_tree_oak_2',  tx: 22, ty: 37, w: 2.5, h: 3.0 },   // west of Garden
  { asset: 'nat_tree_oak_1',  tx: 26, ty: 47, w: 2.5, h: 3.0 },   // below Garden
  { asset: 'nat_tree_oak_3',  tx: 17, ty: 52, w: 2.5, h: 3.0 },   // west of Uptown
  // -- Town square --
  { asset: 'nat_tree_oak_3',  tx: 40, ty: 30, w: 2.5, h: 3.0 },   // town square NW
  { asset: 'nat_tree_oak_1',  tx: 54, ty: 32, w: 2.5, h: 3.0 },   // town square NE
  { asset: 'nat_pine_1',      tx: 44, ty: 42, w: 2.0, h: 3.0 },   // town square S
  { asset: 'nat_tree_oak_2',  tx: 48, ty: 36, w: 2.5, h: 3.0 },   // town square center-S
  // -- East --
  { asset: 'nat_tree_oak_2',  tx: 68, ty: 44, w: 2.5, h: 3.0 },   // east alley
  { asset: 'nat_pine_1',      tx: 74, ty: 38, w: 2.0, h: 3.0 },   // east of Post Office
  { asset: 'nat_tree_oak_1',  tx: 72, ty: 56, w: 2.5, h: 3.0 },   // market row
  { asset: 'nat_tree_oak_3',  tx: 80, ty: 48, w: 2.5, h: 3.0 },   // east of Pub
  // -- South --
  { asset: 'nat_tree_oak_2',  tx: 46, ty: 59, w: 2.5, h: 3.0 },   // south common
  { asset: 'nat_pine_1',      tx: 36, ty: 62, w: 2.0, h: 3.0 },   // near park NW
  { asset: 'nat_tree_oak_1',  tx: 32, ty: 64, w: 2.5, h: 3.0 },   // near park W
  { asset: 'nat_pine_1',      tx: 50, ty: 72, w: 2.0, h: 3.0 },   // south passage
  { asset: 'nat_tree_oak_2',  tx: 56, ty: 74, w: 2.5, h: 3.0 },   // between Supply & Chapel
  { asset: 'nat_tree_oak_3',  tx: 44, ty: 74, w: 2.5, h: 3.0 },   // south of Farm
  // -- Garden path / interior --
  { asset: 'nat_tree_oak_1',  tx: 38, ty: 40, w: 2.5, h: 3.0 },   // garden path
  { asset: 'nat_pine_1',      tx: 36, ty: 50, w: 2.0, h: 3.0 },   // between Garden & Park
  { asset: 'nat_pine_1',      tx: 65, ty: 60, w: 2.0, h: 3.0 },   // east of Supply
];

// Obstacle zones for large decorations (blocked in collision grid)
const TOWN_OBSTACLES = [
  // Tree trunks (1x1 blocked at trunk base: tx+1,ty+2 for oaks; tx+1,ty+2 for pines)
  // -- North --
  { tx: 39, ty: 23, w: 1, h: 1 },   // oak between Co-Living & College
  { tx: 44, ty: 21, w: 1, h: 1 },   // pine north of College
  { tx: 61, ty: 24, w: 1, h: 1 },   // oak cafe terrace
  // -- West --
  { tx: 25, ty: 33, w: 1, h: 1 },   // pine west meadow upper
  { tx: 23, ty: 39, w: 1, h: 1 },   // oak west of Garden
  { tx: 27, ty: 49, w: 1, h: 1 },   // oak below Garden
  { tx: 18, ty: 54, w: 1, h: 1 },   // oak west of Uptown
  // -- Town square --
  { tx: 41, ty: 32, w: 1, h: 1 },   // oak town square NW
  { tx: 55, ty: 34, w: 1, h: 1 },   // oak town square NE
  { tx: 45, ty: 44, w: 1, h: 1 },   // pine town square S
  { tx: 49, ty: 38, w: 1, h: 1 },   // oak town square center-S
  // -- East --
  { tx: 69, ty: 46, w: 1, h: 1 },   // oak east alley
  { tx: 75, ty: 40, w: 1, h: 1 },   // pine east of Post Office
  { tx: 73, ty: 58, w: 1, h: 1 },   // oak market row
  { tx: 81, ty: 50, w: 1, h: 1 },   // oak east of Pub
  // -- South --
  { tx: 47, ty: 61, w: 1, h: 1 },   // oak south common
  { tx: 37, ty: 64, w: 1, h: 1 },   // pine near park NW
  { tx: 33, ty: 66, w: 1, h: 1 },   // oak near park W
  { tx: 51, ty: 74, w: 1, h: 1 },   // pine south passage
  { tx: 57, ty: 76, w: 1, h: 1 },   // oak between Supply & Chapel
  { tx: 45, ty: 76, w: 1, h: 1 },   // oak south of Farm
  // -- Garden path / interior --
  { tx: 39, ty: 42, w: 1, h: 1 },   // oak garden path
  { tx: 37, ty: 52, w: 1, h: 1 },   // pine between Garden & Park
  { tx: 66, ty: 62, w: 1, h: 1 },   // pine east of Supply
];

// Compute bounding box (iw, ih) from parts for each building
for (const b of Object.values(BUILDINGS)) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of b.parts) {
    minX = Math.min(minX, p.dx);
    minY = Math.min(minY, p.dy);
    maxX = Math.max(maxX, p.dx + p.w);
    maxY = Math.max(maxY, p.dy + p.h);
  }
  b.iw = maxX - minX;
  b.ih = maxY - minY;
  b._minDx = minX;
  b._minDy = minY;
}

// Precompute pixel centers for each building
for (const b of Object.values(BUILDINGS)) {
  b.cx = b.tx * TILE + TILE / 2;
  b.cy = b.ty * TILE + TILE / 2;
}

// ── Location helpers (used by agent-schedule.js) ──────────────
const LOCATIONS = {};
for (const [key, b] of Object.entries(BUILDINGS)) {
  LOCATIONS[key] = { name: b.name, emoji: b.emoji };
}
const LOC_KEYS = Object.keys(BUILDINGS);

// ── Home positions — 100 plots evenly spaced around circle perimeter ──
const HOME_POSITIONS = [];
for (let i = 0; i < 100; i++) {
  const angle = (i / 100) * 2 * Math.PI - Math.PI / 2; // start from top
  const tx = Math.round(CIRCLE_CENTER.tx + CIRCLE_RADIUS * Math.cos(angle));
  const ty = Math.round(CIRCLE_CENTER.ty + CIRCLE_RADIUS * Math.sin(angle));
  HOME_POSITIONS.push({ tx, ty, angle });
}

// ── Home position in pixels ────────────────────────────────
function getHomePosT(agentId) {
  const h = HOME_POSITIONS[agentId % HOME_POSITIONS.length];
  return { x: h.tx * TILE + TILE / 2, y: h.ty * TILE + TILE / 2 };
}

// ── House sprite library ────────────────────────────────────
const HOUSE_S = [
  'house_frontal_1','house_yellow','house_frontal_4','house_frontal_5',
  'house_frontal_7','house_frontal_8','house_frontal_9','house_frontal_10',
  'house_frontal_11','house_frontal_12','house_frontal_13'
];
const HOUSE_SE = [
  'house_blue','house_brown','house_frontal_3',
  'house_rear_1','house_rear_3','house_rear_4',
  'house_rear_v2_1','house_rear_v2_2','house_rear_v2_4',
  'house_east_3','house_east_4',
  'house_pink_flipX','house_red_flipX',
  'house_rear_v2_3_flipX','house_rear_2_flipX',
  'house_east_1_flipX','house_east_2_flipX'
];
const HOUSE_SW = [
  'house_pink','house_red','house_frontal_3_flipX',
  'house_rear_2','house_rear_v2_3','house_east_1','house_east_2',
  'house_blue_flipX','house_brown_flipX',
  'house_rear_1_flipX','house_rear_3_flipX','house_rear_4_flipX',
  'house_rear_v2_1_flipX','house_rear_v2_2_flipX','house_rear_v2_4_flipX',
  'house_east_3_flipX','house_east_4_flipX'
];

// Returns { folder: 'S'|'SE'|'SW', pool: string[] } based on angle (degrees from top, clockwise)
// deg: 0=top(12 o'clock), 90=right(3 o'clock), 180=bottom(6 o'clock), 270=left(9 o'clock)
function getHouseView(deg) {
  // S-facing at top (330-30) and bottom (150-210) — frontal view
  if (deg >= 330 || deg < 30 || (deg >= 150 && deg < 210)) {
    return { folder: 'S', pool: HOUSE_S };
  }
  // SW on the right side (30-150) — houses face left toward center
  if (deg >= 30 && deg < 150) {
    return { folder: 'SW', pool: HOUSE_SW };
  }
  // SE on the left side (210-330) — houses face right toward center
  return { folder: 'SE', pool: HOUSE_SE };
}

// Assign each home a deterministic house sprite
const HOME_HOUSES = [];
for (let i = 0; i < 100; i++) {
  const deg = (i / 100) * 360;
  const view = getHouseView(deg);
  const sprite = view.pool[i % view.pool.length];
  HOME_HOUSES.push({ folder: view.folder, sprite });
}

// ── Terrain grid generation ────────────────────────────────
function buildTerrainGrid() {
  const W = MAP_TILES_W, H = MAP_TILES_H;
  const grid = Array.from({ length: H }, () => new Array(W).fill(TERRAIN.GRASS));
  const cx = CIRCLE_CENTER.tx, cy = CIRCLE_CENTER.ty;
  const r2 = CIRCLE_RADIUS * CIRCLE_RADIUS;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const dx = x - cx, dy = y - cy;
      if (dx * dx + dy * dy <= r2) {
        grid[y][x] = TERRAIN.DIRT;
      }
    }
  }
  return grid;
}

// ── Collision grid generation ────────────────────────────────
// true = blocked, false = walkable
function buildCollisionGrid(terrainGrid) {
  const W = MAP_TILES_W, H = MAP_TILES_H;
  const coll = Array.from({ length: H }, () => new Array(W).fill(true));

  // Dirt circle is walkable
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (terrainGrid[y][x] === TERRAIN.DIRT) {
        coll[y][x] = false;
      }
    }
  }

  // Building interiors are BLOCKED for pathfinding.
  // BFS routes around buildings on dirt paths.
  // Agents smoothly enter/exit via the last/first tween segment.
  // (Some building tiles overlap with dirt circle — re-block them.)
  // Open-air spaces (parks, gardens) stay walkable.
  for (const [key, b] of Object.entries(BUILDINGS)) {
    if (b.open) continue;
    for (const p of b.parts) {
      const startX = b.tx + p.dx;
      const startY = b.ty + p.dy;
      for (let y = startY; y < startY + p.h; y++)
        for (let x = startX; x < startX + p.w; x++)
          if (x >= 0 && x < W && y >= 0 && y < H) coll[y][x] = true;
    }
  }

  // Home tiles are walkable (they're on the perimeter, may be on grass)
  for (const h of HOME_POSITIONS) {
    if (h.ty >= 0 && h.ty < H && h.tx >= 0 && h.tx < W) {
      coll[h.ty][h.tx] = false;
      // Also open 1-tile radius around home for easier pathfinding
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const nx = h.tx + dx, ny = h.ty + dy;
          if (nx >= 0 && nx < W && ny >= 0 && ny < H) coll[ny][nx] = false;
        }
    }
  }

  // Block specific zones within open buildings (e.g. pond water)
  for (const [key, b] of Object.entries(BUILDINGS)) {
    if (!b.blockedZones) continue;
    for (const z of b.blockedZones) {
      const sx = b.tx + z.dx, sy = b.ty + z.dy;
      for (let y = sy; y < sy + z.h; y++)
        for (let x = sx; x < sx + z.w; x++)
          if (x >= 0 && x < W && y >= 0 && y < H) coll[y][x] = true;
    }
  }

  // Block standalone town obstacles (trees, wells, boulders)
  for (const obs of TOWN_OBSTACLES) {
    for (let y = obs.ty; y < obs.ty + obs.h; y++)
      for (let x = obs.tx; x < obs.tx + obs.w; x++)
        if (x >= 0 && x < W && y >= 0 && y < H) coll[y][x] = true;
  }

  return coll;
}

// ── 3D Wall computation for buildings ────────────────────────
// Computes outer wall edges from a building's parts (open-plan interiors).
// Returns array of { cx, cy, side: 'N'|'S'|'E'|'W', type: 'outer' }
function computeBuildingWalls(b) {
  var source = b.rooms || b.parts;
  var cells = {};
  for (var pi = 0; pi < source.length; pi++) {
    var p = source[pi];
    for (var y = p.dy; y < p.dy + p.h; y++)
      for (var x = p.dx; x < p.dx + p.w; x++)
        cells[x + ',' + y] = true;
  }

  var edges = [];
  for (var key in cells) {
    var sp = key.split(','), cx = +sp[0], cy = +sp[1];
    if (!cells[cx + ',' + (cy - 1)]) edges.push({ cx: cx, cy: cy, side: 'N', type: 'outer' });
    if (!cells[cx + ',' + (cy + 1)]) edges.push({ cx: cx, cy: cy, side: 'S', type: 'outer' });
    if (!cells[(cx - 1) + ',' + cy]) edges.push({ cx: cx, cy: cy, side: 'W', type: 'outer' });
    if (!cells[(cx + 1) + ',' + cy]) edges.push({ cx: cx, cy: cy, side: 'E', type: 'outer' });
  }

  // Filter out door gaps
  var doors = b.doors || [];
  var result = [];
  for (var ei = 0; ei < edges.length; ei++) {
    var e = edges[ei], blocked = false;
    for (var di = 0; di < doors.length; di++) {
      var d = doors[di];
      // Horizontal door (y1 === y2): blocks S edges at y-1 or N edges at y
      if (d.y1 === d.y2) {
        if (e.side === 'S' && e.cy + 1 === d.y1 && e.cx >= d.x1 && e.cx < d.x2) { blocked = true; break; }
        if (e.side === 'N' && e.cy === d.y1 && e.cx >= d.x1 && e.cx < d.x2) { blocked = true; break; }
      }
      // Vertical door (x1 === x2): blocks E edges at x-1 or W edges at x
      if (d.x1 === d.x2) {
        if (e.side === 'E' && e.cx + 1 === d.x1 && e.cy >= d.y1 && e.cy < d.y2) { blocked = true; break; }
        if (e.side === 'W' && e.cx === d.x1 && e.cy >= d.y1 && e.cy < d.y2) { blocked = true; break; }
      }
    }
    if (!blocked) result.push(e);
  }
  return result;
}

// Precompute wall edges for all buildings
const BUILDING_WALLS = {};
for (const [key, b] of Object.entries(BUILDINGS)) {
  BUILDING_WALLS[key] = computeBuildingWalls(b);
}

// ── Wall-edge blocking set (for BFS pathfinding) ──────────────
// Stores "x1,y1>x2,y2" strings for each blocked edge transition.
// BFS checks this set before allowing movement between adjacent tiles.
const WALL_EDGES = new Set();
for (const [key, b] of Object.entries(BUILDINGS)) {
  if (!b.doors || b.doors.length === 0) continue;  // skip doorless buildings
  const walls = BUILDING_WALLS[key];
  for (let i = 0; i < walls.length; i++) {
    const w = walls[i];
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

// ── Door tile lookup (for pathfinding) ──────────────────────
// Returns the walkable tile just OUTSIDE a building's primary door.
function getBuildingDoorTile(buildingKey) {
  const b = BUILDINGS[buildingKey];
  if (!b || !b.doors || b.doors.length === 0) return null;
  const d = b.doors[0];  // use primary door

  if (d.y1 === d.y2) {
    // Horizontal door at relative y = d.y1
    const midX = Math.floor((d.x1 + d.x2) / 2);
    const absX = b.tx + midX;
    const absYbelow = b.ty + d.y1;      // tile at boundary row
    const absYabove = b.ty + d.y1 - 1;  // tile above boundary
    // Check if the tile below the boundary is inside the building
    let belowInside = false;
    for (const p of b.parts) {
      const sx = b.tx + p.dx, sy = b.ty + p.dy;
      if (absX >= sx && absX < sx + p.w && absYbelow >= sy && absYbelow < sy + p.h) {
        belowInside = true; break;
      }
    }
    // Exit tile is on the side that is NOT inside the building
    return belowInside ? [absX, absYabove] : [absX, absYbelow];
  } else {
    // Vertical door at relative x = d.x1
    const midY = Math.floor((d.y1 + d.y2) / 2);
    const absY = b.ty + midY;
    const absXright = b.tx + d.x1;      // tile at boundary column
    const absXleft  = b.tx + d.x1 - 1;  // tile left of boundary
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

// Precompute interior cell sets for each building (absolute tile coords)
// Used by interior BFS to route agents inside non-rectangular buildings.
const BUILDING_CELLS = {};
for (const [key, b] of Object.entries(BUILDINGS)) {
  const cells = new Set();
  const source = b.parts;
  for (const p of source) {
    for (let y = p.dy; y < p.dy + p.h; y++)
      for (let x = p.dx; x < p.dx + p.w; x++)
        cells.add((b.tx + x) + ',' + (b.ty + y));
  }
  BUILDING_CELLS[key] = cells;
}

// Remove blocked zone tiles from BUILDING_CELLS (e.g. pond water)
for (const [key, b] of Object.entries(BUILDINGS)) {
  if (!b.blockedZones || !BUILDING_CELLS[key]) continue;
  for (const z of b.blockedZones) {
    for (let y = z.dy; y < z.dy + z.h; y++)
      for (let x = z.dx; x < z.dx + z.w; x++)
        BUILDING_CELLS[key].delete((b.tx + x) + ',' + (b.ty + y));
  }
}

