// ============================================================
// GABM Viz — Config
// Change DATA_RUN to point at any model's run folder.
// ============================================================

const CONFIG = {
  // Path to data (relative to index.html, works locally + on GitHub Pages)
  // Change these three lines to switch between runs:
  DATA_RUN:    "../data/synthetic_test/run-1",
  DATA_PREFIX: "synthetic_test_",
  RUN_SUFFIX:  "_run-1",

  // World settings
  WORLD_NAME:  "Dewberry Hollow",
  MODEL_LABEL: "Synthetic  |  100 agents  |  50 days",

  // Animation
  DEFAULT_FPS: 3,        // timesteps per second at 1× speed
  SPEED_STEPS: [0.5, 1, 2, 4],

  // Canvas dimensions
  CANVAS_W: 900,
  CANVAS_H: 440,

  // Layout split (home zone | divider | grid zone)
  HOME_ZONE_W: 280,
  DIVIDER_W: 2,

  // Agent sprite size (pixels on canvas)
  AGENT_SIZE: 22,
  AGENT_PADDING: 6,

  // Colors
  COLORS: {
    BG:          "#0a0a0f",
    HOME_BG:     "#111827",
    GRID_BG:     "#0d1117",
    DIVIDER:     "#2d3748",
    HOUSE:       "#2d3748",
    HOUSE_ROOF:  "#4a5568",
    SUSCEPTIBLE: "#4ade80",   // green
    INFECTED:    "#f87171",   // red
    RECOVERED:   "#60a5fa",   // blue
    TEXT:        "#e2e8f0",
    TEXT_DIM:    "#64748b",
    GRID_LINE:   "#1a2035",
    CHART_S:     "#4ade80",
    CHART_I:     "#f87171",
    CHART_R:     "#60a5fa",
  },
};
