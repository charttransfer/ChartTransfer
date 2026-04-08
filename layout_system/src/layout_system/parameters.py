"""Global parameters for layout optimization."""

# Optimization resolution stages
OPT_RES_LIST = (256, 512)  # Coarse-to-fine: 512 is ~3.8x cheaper than 1000

# Augmented Lagrangian parameters
OUTER_ROUNDS = 5   # augmented-lagrangian outer updates per stage
INNER_STEPS = 30   # gradient steps per outer round
RHO_INIT = 1e-4    # initial penalty parameter for constraint enforcement
RHO_MULT = 5     # multiplier for penalty parameter

# Optimizer parameters
LEARNING_RATE = 0.01
SIZE_MIN = 20.0    # minimum element size in pixels

# Penalty weights
PEN_WEIGHT = 1000.0   # weight for penetration penalty
PEN_ETA_PX = 1.0   # eta parameter for penetration penalty

# Loss weights
W_SIMILARITY = 100   # weight for position/size similarity loss
W_READABILITY = 100  # weight for readability loss (size hierarchy)
W_ALIGNMENT_CONSISTENCY = 100.0  # weight for alignment consistency loss (hierarchical alignment)
W_ALIGNMENT_SIMILARITY = 100.0  # weight for alignment similarity loss (based on JSON constraint)
W_DATA_INK = 3  # weight for data ink loss (maximize union area, minimize white space)
W_VISUAL_BALANCE = 1.0  # weight for visual balance loss (centroid to center distance)

# Tau schedule for soft mask (controls boundary sharpness)
TAU_SCHEDULE = (2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2)

# Pictogram dilation radius
PICTOGRAM_DILATION_RADIUS = 5.0  # pixels

# Readability threshold
SIZE_RATIO_THRESHOLD = 1.5  # minimum ratio to consider size difference significant
# Rules: if ratio >= 1.5 or ratio <= 1/1.5 (0.67), the size difference is significant

# Min size constraints (default values)
MIN_WIDTH_DEFAULT = 30.0
MIN_HEIGHT_DEFAULT = 30.0

# Proximity ratio loss parameters
W_PROXIMITY = 10000.0  # Weight for proximity ratio loss (default disabled, can be enabled when needed)
W_FULLY_INSIDE = 10.0  # Weight for fully_overlap: non-chart inside chart no_grid, disjoint from not_overlap
PROXIMITY_EPSILON = 1e-6  # Small value to avoid division by zero

# Bilevel normalized loss weights (L_norm_i = L_i / L_i_initial, total = sum(w_i * L_norm_i))
BILEVEL_W_ALIGNMENT = 1.0
BILEVEL_W_BALANCE = 1.0
BILEVEL_W_INK = 1.0
BILEVEL_W_SIMILARITY = 1.0
BILEVEL_W_PROXIMITY = 1.0
BILEVEL_W_OVERLAP = 1.0

# Grid search initialization parameters
GRID_SEARCH_DOWNSCALE_FACTOR = 10
GRID_SEARCH_POSITION_STEPS = 10
GRID_SEARCH_SCALE_STEPS = 5
