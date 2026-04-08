"""Grid search optimizer for layout optimization (ablation baseline for SDF optimizer).

Uses iterative coordinate descent over a discrete (x, y, scale) grid,
evaluating the same loss functions as the SDF gradient-based optimizer.
"""

import os
import shutil
import time as _time
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch

import layout_system.parameters as params

from .core import (
    load_binary_mask_from_rgba,
    tight_bbox_ratio,
    binary_to_sdf_norm,
    dilate_mask,
    make_container_grid,
    sdf_to_softmask,
    area_sum,
)
from .bbox import (
    bbox_aspect_from_unconstrained,
    unconstrained_from_bbox,
)
from .losses import (
    compute_alignment_consistency_loss,
    compute_alignment_similarity_loss,
    compute_readability_loss,
    compute_proximity_ratio_loss,
    compute_visual_balance_loss,
    compute_position_size_similarity_loss,
    compute_overlap_loss_mask,
    compute_data_ink_loss_mask,
    compute_fully_inside_loss,
    _get_pair_overlap_type,
    _is_non_overlap_constraint,
)
from .visualization import (
    visualize_optimization_progress,
    save_composite_image,
)
from .utils import (
    check_all_non_overlap,
    normalize_reference_bboxes,
    init_bboxes_from_reference,
)


class _OverlapCache:
    """Pre-zoomed masks and non-overlap pairs for incremental overlap computation.

    Instead of calling compute_overlap_loss_mask (which zooms ALL masks every
    call), we zoom each mask once and cache the results.  When one element
    changes, we only re-zoom that element's mask and recompute only the
    pairs involving it.
    """

    def __init__(
        self,
        raw_masks: List[np.ndarray],
        bboxes: List[Tuple[float, float, float, float]],
        Wc: int, Hc: int,
        overlap_constraints,
    ):
        from scipy.ndimage import zoom as _zoom
        self._zoom = _zoom
        self.Wc = Wc
        self.Hc = Hc
        self.raw_masks = raw_masks
        self.num = len(raw_masks)

        # Pre-compute non-overlap pairs
        self.non_overlap_pairs: List[Tuple[int, int]] = []
        for i in range(self.num):
            for j in range(i + 1, self.num):
                pt = _get_pair_overlap_type(overlap_constraints, i, j)
                if _is_non_overlap_constraint(pt):
                    self.non_overlap_pairs.append((i, j))

        # Pre-zoom all masks
        self.int_bboxes: List[Tuple[int, int, int, int]] = []
        self.zoomed: List[np.ndarray] = []
        for k in range(self.num):
            zb, ib = self._zoom_mask(k, bboxes[k])
            self.zoomed.append(zb)
            self.int_bboxes.append(ib)

    def _zoom_mask(self, k: int, bbox):
        x, y, w, h = bbox
        xi, yi = int(float(x)), int(float(y))
        wi, hi = max(1, int(float(w))), max(1, int(float(h)))
        ib = (xi, yi, wi, hi)

        mask_k = self.raw_masks[k]
        m01 = (mask_k > 0.5).astype(np.float32) if mask_k.dtype != np.float32 else np.clip(mask_k, 0, 1)
        if m01.shape != (hi, wi):
            m01 = self._zoom(m01, (hi / m01.shape[0], wi / m01.shape[1]), order=1)
        return m01 > 0.5, ib

    def compute_full(self) -> float:
        total = 0.0
        for i, j in self.non_overlap_pairs:
            total += self._pair_overlap(i, j)
        return total

    def compute_incremental(self, changed_idx: int, new_bbox) -> float:
        """Temporarily swap one element's zoomed mask and recompute overlap."""
        old_zoomed = self.zoomed[changed_idx]
        old_ib = self.int_bboxes[changed_idx]

        new_zoomed, new_ib = self._zoom_mask(changed_idx, new_bbox)
        self.zoomed[changed_idx] = new_zoomed
        self.int_bboxes[changed_idx] = new_ib

        total = 0.0
        for i, j in self.non_overlap_pairs:
            total += self._pair_overlap(i, j)

        # Restore
        self.zoomed[changed_idx] = old_zoomed
        self.int_bboxes[changed_idx] = old_ib
        return total

    def update(self, changed_idx: int, new_bbox):
        """Permanently update an element's cached zoom."""
        new_zoomed, new_ib = self._zoom_mask(changed_idx, new_bbox)
        self.zoomed[changed_idx] = new_zoomed
        self.int_bboxes[changed_idx] = new_ib

    def _pair_overlap(self, i: int, j: int) -> float:
        x_i, y_i, w_i, h_i = self.int_bboxes[i]
        x_j, y_j, w_j, h_j = self.int_bboxes[j]

        y_si, y_ei = max(0, y_i), min(y_i + h_i, self.Hc)
        x_si, x_ei = max(0, x_i), min(x_i + w_i, self.Wc)
        y_sj, y_ej = max(0, y_j), min(y_j + h_j, self.Hc)
        x_sj, x_ej = max(0, x_j), min(x_j + w_j, self.Wc)

        iy0 = max(y_si, y_sj)
        iy1 = min(y_ei, y_ej)
        ix0 = max(x_si, x_sj)
        ix1 = min(x_ei, x_ej)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0

        crop_i = self.zoomed[i][iy0 - y_si:iy1 - y_si, ix0 - x_si:ix1 - x_si]
        crop_j = self.zoomed[j][iy0 - y_sj:iy1 - y_sj, ix0 - x_sj:ix1 - x_sj]
        return float(np.sum(crop_i & crop_j))


class _DataInkCache:
    """Pre-zoomed masks for incremental data ink (non-layer) computation.

    compute_data_ink_loss_mask zooms each mask and composites onto canvas.
    We cache the composite and per-element placements to avoid re-zooming
    unchanged elements.
    """

    def __init__(
        self,
        raw_masks: List[np.ndarray],
        bboxes: List[Tuple[float, float, float, float]],
        Wc: int, Hc: int,
    ):
        from scipy.ndimage import zoom as _zoom
        self._zoom = _zoom
        self.Wc = Wc
        self.Hc = Hc
        self.raw_masks = raw_masks
        self.num = len(raw_masks)
        self.bboxes = list(bboxes)

        # Pre-zoom each mask and build composite
        self._zoomed_placed: List[Optional[Tuple[np.ndarray, int, int, int, int]]] = []
        self.composite = np.zeros((Hc, Wc), dtype=np.float32)

        for k in range(self.num):
            placed = self._zoom_and_place(k, bboxes[k])
            self._zoomed_placed.append(placed)
            if placed is not None:
                crop, y_start, x_start, y_end, x_end = placed
                self.composite[y_start:y_end, x_start:x_end] = np.maximum(
                    self.composite[y_start:y_end, x_start:x_end], crop
                )

    def _zoom_and_place(self, k: int, bbox):
        x, y, w, h = bbox
        xi, yi = int(float(x)), int(float(y))
        wi, hi = max(1, int(float(w))), max(1, int(float(h)))

        mask_k = self.raw_masks[k]
        if mask_k.size == 0:
            return None
        m01 = (mask_k > 0.5).astype(np.float32) if mask_k.dtype != np.float32 else np.clip(mask_k, 0, 1)
        if m01.shape[0] != hi or m01.shape[1] != wi:
            m01 = self._zoom(m01, (hi / m01.shape[0], wi / m01.shape[1]), order=1)
        m01 = np.clip(m01, 0, 1)

        y_end = min(yi + hi, self.Hc)
        x_end = min(xi + wi, self.Wc)
        y_start = max(0, yi)
        x_start = max(0, xi)
        if y_end <= y_start or x_end <= x_start:
            return None
        crop = m01[:y_end - y_start, :x_end - x_start]
        return (crop, y_start, x_start, y_end, x_end)

    def compute_full(self) -> float:
        return -float(np.sum(self.composite > 0.5))

    def compute_incremental(self, changed_idx: int, new_bbox) -> float:
        """Recompute data ink with one element changed, without re-zooming others."""
        # Rebuild composite from scratch but only zoom the changed element
        comp = np.zeros((self.Hc, self.Wc), dtype=np.float32)
        for k in range(self.num):
            if k == changed_idx:
                placed = self._zoom_and_place(k, new_bbox)
            else:
                placed = self._zoomed_placed[k]
            if placed is not None:
                crop, y_start, x_start, y_end, x_end = placed
                comp[y_start:y_end, x_start:x_end] = np.maximum(
                    comp[y_start:y_end, x_start:x_end], crop
                )
        return -float(np.sum(comp > 0.5))

    def update(self, changed_idx: int, new_bbox):
        """Permanently update the cache for one element."""
        self.bboxes[changed_idx] = new_bbox
        self._zoomed_placed[changed_idx] = self._zoom_and_place(changed_idx, new_bbox)
        # Rebuild composite
        self.composite = np.zeros((self.Hc, self.Wc), dtype=np.float32)
        for placed in self._zoomed_placed:
            if placed is not None:
                crop, y_start, x_start, y_end, x_end = placed
                self.composite[y_start:y_end, x_start:x_end] = np.maximum(
                    self.composite[y_start:y_end, x_start:x_end], crop
                )


def _render_hard_mask(
    sdf_tensor: torch.Tensor,
    x: float, y: float, w: float, h: float,
    Hc: int, Wc: int,
    device: str,
) -> np.ndarray:
    """Render SDF to a binary mask placed on the canvas."""
    X_grid, Y_grid = make_container_grid(Hc, Wc, device=device)
    with torch.no_grad():
        m, _ = sdf_to_softmask(
            sdf_tensor,
            torch.tensor(x, device=device),
            torch.tensor(y, device=device),
            torch.tensor(w, device=device),
            torch.tensor(h, device=device),
            X_grid, Y_grid, tau_px=0.5,
        )
    return (m.squeeze().cpu().numpy() > 0.5).astype(np.float32)


def _compute_bbox_only_loss(
    bboxes: List[Tuple[float, float, float, float]],
    Wc: int, Hc: int,
    reference_bboxes,
    reference_parent_bbox,
    w_similarity: float,
    size_rules,
    w_readability: float,
    w_alignment_consistency: float,
    alignment_constraint,
    w_alignment_similarity: float,
    proximity_info,
    w_proximity: float,
    w_visual_balance: float,
    overlap_constraints,
    is_layer: bool,
    pen_pairs: List[Tuple[int, int]],
    device: str,
) -> float:
    """Compute losses that only depend on bboxes (no mask rendering).

    This is cheap and used as an early-exit filter in the grid search loop.
    """
    num_nodes = len(bboxes)
    bbox_tensors = [
        torch.tensor(list(b), dtype=torch.float32, device=device) for b in bboxes
    ]

    loss = 0.0

    if w_visual_balance > 0:
        bbox_tuples = [(torch.tensor(b[0], device=device),
                        torch.tensor(b[1], device=device),
                        torch.tensor(b[2], device=device),
                        torch.tensor(b[3], device=device)) for b in bboxes]
        L_balance = compute_visual_balance_loss(
            bbox_tuples, Wc, Hc, masks=None, device=device
        )
        loss += w_visual_balance * L_balance.item()

    if w_similarity > 0 and reference_bboxes and len(reference_bboxes) >= num_nodes:
        generated_parent_bbox = (0.0, 0.0, float(Wc), float(Hc))
        L_sim = compute_position_size_similarity_loss(
            reference_bboxes[:num_nodes],
            bbox_tensors,
            reference_parent_bbox,
            generated_parent_bbox,
            device=device,
        )
        loss += w_similarity * L_sim.item()

    if w_readability > 0 and size_rules:
        L_read = compute_readability_loss(
            size_rules, bbox_tensors,
            size_ratio_threshold=params.SIZE_RATIO_THRESHOLD,
            device=device,
        )
        loss += w_readability * L_read.item()

    if w_alignment_consistency > 0 and reference_bboxes and len(reference_bboxes) >= num_nodes:
        generated_parent_bbox_a = (0.0, 0.0, float(Wc), float(Hc))
        L_align_cons = compute_alignment_consistency_loss(
            reference_bboxes[:num_nodes],
            bbox_tensors,
            reference_parent_bbox,
            generated_parent_bbox_a,
            device=device,
        )
        loss += w_alignment_consistency * L_align_cons.item()

    if w_alignment_similarity > 0 and alignment_constraint:
        container_bbox_a = (0.0, 0.0, float(Wc), float(Hc))
        L_align_sim = compute_alignment_similarity_loss(
            bbox_tensors, container_bbox_a, alignment_constraint, device=device,
        )
        loss += w_alignment_similarity * L_align_sim.item()

    if w_proximity > 0 and proximity_info:
        container_bboxes_p = proximity_info.get("containers", [])
        child_bboxes_list_p = proximity_info.get("children", [])
        grandchild_bboxes_list_p = proximity_info.get("grandchildren", [])
        container_types_p = proximity_info.get("types", [])
        container_weights_p = proximity_info.get("weights", None)

        if len(container_bboxes_p) == 0:
            container_bbox_p = (0.0, 0.0, float(Wc), float(Hc))
            child_bboxes_p = list(bboxes)
            grandchild_bboxes_p = []
            container_type_p = container_types_p[0] if container_types_p else "row"
            L_prox = compute_proximity_ratio_loss(
                [container_bbox_p], [child_bboxes_p], [grandchild_bboxes_p],
                [container_type_p],
                container_weights=[1.0] if container_weights_p is None else container_weights_p,
                epsilon=params.PROXIMITY_EPSILON, device=device,
            )
        else:
            if len(child_bboxes_list_p) > 0 and len(child_bboxes_list_p[0]) == num_nodes:
                updated = [list(bboxes)] + child_bboxes_list_p[1:]
            else:
                updated = child_bboxes_list_p
            L_prox = compute_proximity_ratio_loss(
                container_bboxes_p, updated, grandchild_bboxes_list_p,
                container_types_p, container_weights_p,
                epsilon=params.PROXIMITY_EPSILON, device=device,
            )
        loss += w_proximity * L_prox.item()

    min_gap_px = 20.0
    pen_weight = params.PEN_WEIGHT
    if pen_weight > 0 and not is_layer and pen_pairs:
        L_pen = 0.0
        for i, j in pen_pairs:
            x_i, y_i, w_i, h_i = bboxes[i]
            x_j, y_j, w_j, h_j = bboxes[j]
            gap_x = max(x_j - (x_i + w_i), x_i - (x_j + w_j))
            gap_y = max(y_j - (y_i + h_i), y_i - (y_j + h_j))
            if gap_x >= 0 and gap_y >= 0:
                gap = (gap_x ** 2 + gap_y ** 2) ** 0.5
            elif gap_x < 0 and gap_y < 0:
                gap = max(gap_x, gap_y)
            else:
                gap = max(gap_x, gap_y)
            L_pen += max(0.0, min_gap_px - gap)
        loss += pen_weight * L_pen

    return loss


def _compute_mask_loss(
    bboxes: List[Tuple[float, float, float, float]],
    raw_masks: List[np.ndarray],
    Wc: int, Hc: int,
    w_overlap: float,
    overlap_constraints,
    w_data_ink: float,
    is_layer: bool,
    rendered_mask_areas: Optional[List[float]] = None,
) -> float:
    """Compute losses that depend on masks (overlap, data_ink).

    Legacy path for initial loss computation (full recompute).
    """
    num_nodes = len(bboxes)
    loss = 0.0

    if w_overlap > 0 and num_nodes >= 2:
        L_overlap = compute_overlap_loss_mask(
            raw_masks, bboxes, Wc, Hc, overlap_constraints
        )
        loss += w_overlap * L_overlap

    if w_data_ink != 0:
        if is_layer and rendered_mask_areas is not None:
            L_data_ink = -sum(rendered_mask_areas)
        else:
            L_data_ink = compute_data_ink_loss_mask(raw_masks, bboxes, Wc, Hc)
        loss += w_data_ink * L_data_ink

    return loss


def _compute_mask_loss_incremental(
    changed_idx: int,
    new_bbox: Tuple[float, float, float, float],
    w_overlap: float,
    w_data_ink: float,
    is_layer: bool,
    overlap_cache: Optional[_OverlapCache],
    data_ink_cache: Optional[_DataInkCache],
    rendered_mask_areas: Optional[List[float]] = None,
    changed_rendered_area: Optional[float] = None,
) -> float:
    """Incremental mask loss: only re-zoom the changed element."""
    loss = 0.0

    if w_overlap > 0 and overlap_cache is not None:
        L_overlap = overlap_cache.compute_incremental(changed_idx, new_bbox)
        loss += w_overlap * L_overlap

    if w_data_ink != 0:
        if is_layer and rendered_mask_areas is not None and changed_rendered_area is not None:
            total = sum(rendered_mask_areas) - rendered_mask_areas[changed_idx] + changed_rendered_area
            L_data_ink = -total
        elif data_ink_cache is not None:
            L_data_ink = data_ink_cache.compute_incremental(changed_idx, new_bbox)
        else:
            L_data_ink = 0.0
        loss += w_data_ink * L_data_ink

    return loss


def _compute_total_loss(
    bboxes: List[Tuple[float, float, float, float]],
    masks_np: List[np.ndarray],
    raw_masks: List[np.ndarray],
    Wc: int, Hc: int,
    reference_bboxes,
    reference_parent_bbox,
    w_similarity: float,
    size_rules,
    w_readability: float,
    w_alignment_consistency: float,
    alignment_constraint,
    w_alignment_similarity: float,
    proximity_info,
    w_proximity: float,
    w_data_ink: float,
    w_visual_balance: float,
    w_overlap: float,
    overlap_constraints,
    device: str,
) -> float:
    """Compute total weighted loss (legacy interface for initial loss)."""
    num_nodes = len(bboxes)

    is_layer = False
    if proximity_info:
        ct = proximity_info.get("types", [])
        if ct:
            is_layer = (ct[0] == "layer")

    pen_pairs = []
    if num_nodes >= 2:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                pt = _get_pair_overlap_type(overlap_constraints, i, j)
                if pt != "fully_overlap":
                    pen_pairs.append((i, j))

    bbox_loss = _compute_bbox_only_loss(
        bboxes, Wc, Hc,
        reference_bboxes, reference_parent_bbox,
        w_similarity, size_rules, w_readability,
        w_alignment_consistency, alignment_constraint, w_alignment_similarity,
        proximity_info, w_proximity, w_visual_balance,
        overlap_constraints, is_layer, pen_pairs, device,
    )

    rendered_mask_areas = None
    if is_layer and masks_np:
        rendered_mask_areas = [float(np.sum(m > 0.5)) for m in masks_np]

    mask_loss = _compute_mask_loss(
        bboxes, raw_masks, Wc, Hc,
        w_overlap, overlap_constraints, w_data_ink,
        is_layer, rendered_mask_areas,
    )

    return bbox_loss + mask_loss


def optimize(
    png_list,
    original_png_list=None,
    Wc=1000, Hc=1000,
    opt_res_list=params.OPT_RES_LIST,
    outer_rounds=params.OUTER_ROUNDS,
    inner_steps=params.INNER_STEPS,
    tau_schedule=params.TAU_SCHEDULE,
    rho_init=params.RHO_INIT,
    rho_mult=params.RHO_MULT,
    lr=params.LEARNING_RATE,
    size_min=params.SIZE_MIN,
    min_sizes=None,
    pen_weight=params.PEN_WEIGHT,
    pen_eta_px=params.PEN_ETA_PX,
    reference_bboxes=None,
    reference_parent_bbox=None,
    w_similarity=params.W_SIMILARITY,
    size_rules=None,
    w_readability=params.W_READABILITY,
    w_alignment_consistency=params.W_ALIGNMENT_CONSISTENCY,
    alignment_constraint=None,
    w_alignment_similarity=params.W_ALIGNMENT_SIMILARITY,
    proximity_info=None,
    w_proximity=params.W_PROXIMITY,
    w_data_ink=params.W_DATA_INK,
    w_visual_balance=params.W_VISUAL_BALANCE,
    min_gap_px=20.0,
    overlap_constraints=None,
    chart_dual_masks=None,
    fully_overlap_pairs=None,
    w_fully_inside=params.W_FULLY_INSIDE,
    enable_grid_search_init=True,
    grid_search_config=None,
    initial_only=False,
    # Grid search specific parameters
    num_rounds=3,
    position_grid_size=15,
    scale_steps=8,
    device=None,
    save_prefix=None,
    output_dir=None,
    debug=False,
):
    """Grid search layout optimizer (ablation baseline for SDF optimizer).

    Same interface as sdf/optimizer.py optimize(), but uses iterative
    coordinate descent on a discrete grid instead of gradient-based
    optimization.  All loss functions are shared with the SDF optimizer.

    Returns:
        List of final bounding boxes [(x, y, w, h), ...]
    """
    _t_total_start = _time.time()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if debug:
        print("[GridSearchOptimizer] Device:", device)

    if png_list is None or len(png_list) < 1:
        raise ValueError("png_list must be provided with at least one image path")

    num_nodes = len(png_list)
    if original_png_list is None:
        original_png_list = png_list

    dilation_radii = [15.0] * num_nodes

    if reference_bboxes is None:
        reference_bboxes = []
    if reference_parent_bbox is None:
        reference_parent_bbox = (0.0, 0.0, float(Wc), float(Hc))

    reference_bboxes, reference_parent_bbox = normalize_reference_bboxes(
        reference_bboxes, reference_parent_bbox
    )

    if overlap_constraints is None:
        overlap_constraints = []
    if chart_dual_masks is None:
        chart_dual_masks = {}
    if fully_overlap_pairs is None:
        fully_overlap_pairs = []
    if size_rules is None:
        size_rules = []

    if min_sizes is None:
        min_sizes = [(params.MIN_WIDTH_DEFAULT, params.MIN_HEIGHT_DEFAULT)] * num_nodes
    elif len(min_sizes) < num_nodes:
        default_size = (params.MIN_WIDTH_DEFAULT, params.MIN_HEIGHT_DEFAULT)
        min_sizes = min_sizes + [default_size] * (num_nodes - len(min_sizes))
    elif len(min_sizes) > num_nodes:
        min_sizes = min_sizes[:num_nodes]

    # Visualization folder
    viz_folder = None
    if debug:
        if save_prefix:
            viz_folder = f"{save_prefix}_gs_visualization"
        else:
            viz_folder = "gs_optimization_visualization"
        if output_dir is not None:
            viz_folder = os.path.join(output_dir, os.path.basename(viz_folder))
        if os.path.exists(viz_folder):
            shutil.rmtree(viz_folder)
        os.makedirs(viz_folder, exist_ok=True)

    # ---- Load masks and compute SDF templates ----
    _t_mask = _time.time()
    masks = []
    ratios = []
    for i, png_path in enumerate(png_list):
        if png_path is None:
            raise ValueError(f"Image path at index {i} is None")
        mask = load_binary_mask_from_rgba(png_path)
        mask = dilate_mask(mask, dilation_radii[i])
        masks.append(mask)
        r, _ = tight_bbox_ratio(mask)
        ratios.append(r)
    print(f"[GridSearch timer] mask loading + dilation: {_time.time() - _t_mask:.3f}s")

    _t_sdf = _time.time()
    sdf_tensors = []
    for mask in masks:
        sdf_norm = binary_to_sdf_norm(mask, pad=16)
        sdf_t = torch.from_numpy(sdf_norm)[None, None].to(device)
        sdf_tensors.append(sdf_t)
    print(f"[GridSearch timer] SDF template precompute: {_time.time() - _t_sdf:.3f}s")

    # ---- Initialize from grid search init or reference bboxes ----
    _t_init = _time.time()
    current_bboxes: List[Tuple[float, float, float, float]] = []

    # Grid search init is only enabled when num_nodes < 3 and no pair has an
    # overlap constraint — mirrors SDF optimizer gating via the shared helper.
    _use_gs_init = (
        enable_grid_search_init
        and num_nodes < 3
        and check_all_non_overlap(overlap_constraints, num_nodes)
    )
    if not _use_gs_init and enable_grid_search_init and debug:
        print(
            "[Initialization] Grid search init disabled: "
            "num_nodes >= 3 or overlap constraint present"
        )

    if _use_gs_init:
        from .grid_search_init import grid_search_initialization
        gs_cfg = grid_search_config or {}
        init_params, Wc, Hc = grid_search_initialization(
            masks=masks, sdf_tensors=sdf_tensors, ratios=ratios,
            Wc=Wc, Hc=Hc, min_sizes=min_sizes,
            reference_bboxes=reference_bboxes,
            downscale_factor=gs_cfg.get("downscale_factor", 10),
            position_steps=gs_cfg.get("position_steps", 10),
            scale_steps=gs_cfg.get("scale_steps", 5),
            device=device, size_min=size_min, debug=debug, viz_folder=viz_folder,
        )
        for i in range(num_nodes):
            tx_init, ty_init, ts_init = init_params[i]
            tx_t = torch.tensor(tx_init, device=device)
            ty_t = torch.tensor(ty_init, device=device)
            ts_t = torch.tensor(ts_init, device=device)
            x, y, w, h = bbox_aspect_from_unconstrained(
                tx_t, ty_t, ts_t, Wc, Hc, ratios[i],
                min_width=min_sizes[i][0], min_height=min_sizes[i][1],
                size_min=size_min,
            )
            current_bboxes.append((x.item(), y.item(), w.item(), h.item()))
    elif reference_bboxes and len(reference_bboxes) >= num_nodes:
        current_bboxes = init_bboxes_from_reference(
            reference_bboxes, num_nodes, min_sizes, ratios, Wc, Hc, size_min, device
        )
    else:
        for i in range(num_nodes):
            mw, mh = min_sizes[i]
            current_bboxes.append((0.0, 0.0, float(mw), float(mh)))
    print(f"[GridSearch timer] initialization: {_time.time() - _t_init:.3f}s")

    if initial_only:
        if debug:
            print("[Initialization] Returning initialization-only bboxes")
        return list(current_bboxes)

    if debug:
        print(f"[GridSearchOptimizer] Initial bboxes:")
        for i, bb in enumerate(current_bboxes):
            print(f"  Node {i}: ({bb[0]:.1f}, {bb[1]:.1f}, {bb[2]:.1f}, {bb[3]:.1f})")

    # ---- Helper: render masks for a bbox config ----
    def _render_masks_for_bboxes(bbox_list):
        rendered = []
        for i, (bx, by, bw, bh) in enumerate(bbox_list):
            m = _render_hard_mask(sdf_tensors[i], bx, by, bw, bh, Hc, Wc, device)
            rendered.append(m)
        return rendered

    # ---- Helper: bbox from (x, y, w) with aspect ratio ----
    def _make_bbox(elem_idx, x, y, w):
        r = ratios[elem_idx]
        h = w / r
        x = float(np.clip(x, 0, max(0, Wc - w)))
        y = float(np.clip(y, 0, max(0, Hc - h)))
        return (x, y, float(w), float(h))

    # Overlap weight (same magnitude as SDF augmented Lagrangian converged penalty)
    w_overlap = 10000.0

    # Pre-compute constants for the inner loop
    _is_layer = False
    if proximity_info:
        _ct = proximity_info.get("types", [])
        if _ct:
            _is_layer = (_ct[0] == "layer")

    _pen_pairs: List[Tuple[int, int]] = []
    if num_nodes >= 2:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                pt = _get_pair_overlap_type(overlap_constraints, i, j)
                if pt != "fully_overlap":
                    _pen_pairs.append((i, j))

    # ---- Iterative coordinate descent ----
    _t_opt = _time.time()
    best_loss = float('inf')
    best_bboxes = list(current_bboxes)

    # Compute initial loss (full recompute)
    init_masks = _render_masks_for_bboxes(current_bboxes)
    best_loss = _compute_total_loss(
        current_bboxes, init_masks, masks, Wc, Hc,
        reference_bboxes, reference_parent_bbox,
        w_similarity, size_rules, w_readability,
        w_alignment_consistency, alignment_constraint, w_alignment_similarity,
        proximity_info, w_proximity, w_data_ink, w_visual_balance,
        w_overlap, overlap_constraints, device,
    )

    # Build caches for incremental mask-dependent loss computation
    _t_cache = _time.time()
    _overlap_cache = None
    if w_overlap > 0 and num_nodes >= 2:
        _overlap_cache = _OverlapCache(masks, current_bboxes, Wc, Hc, overlap_constraints)

    _data_ink_cache = None
    if w_data_ink != 0 and not _is_layer:
        _data_ink_cache = _DataInkCache(masks, current_bboxes, Wc, Hc)

    if _is_layer:
        _rendered_mask_areas = [float(np.sum(m > 0.5)) for m in init_masks]
    else:
        _rendered_mask_areas = None

    print(f"[GridSearch] initial loss={best_loss:.4f}, "
          f"rounds={num_rounds}, elems={num_nodes}, "
          f"scales={scale_steps}, positions={position_grid_size}x{position_grid_size}, "
          f"is_layer={_is_layer}, cache_build={_time.time() - _t_cache:.3f}s")
    _trial_count = 0
    _skip_count = 0

    for round_idx in range(num_rounds):
        _t_round = _time.time()
        improved_this_round = False

        for elem_idx in range(num_nodes):
            r = ratios[elem_idx]
            mw_min, mh_min = min_sizes[elem_idx]

            w_min_search = max(float(mw_min), float(r) * float(mh_min))
            w_max_search = min(float(Wc), float(r) * float(Hc))
            if w_min_search >= w_max_search:
                w_min_search = max(1.0, 0.5 * w_max_search)

            cur_x, cur_y, cur_w, cur_h = current_bboxes[elem_idx]

            scale_candidates = np.linspace(w_min_search, w_max_search, scale_steps)
            scale_candidates = np.unique(np.append(scale_candidates, cur_w))

            best_elem_loss = best_loss
            best_elem_bbox = current_bboxes[elem_idx]
            _elem_trials = 0
            _elem_skips = 0
            _t_elem = _time.time()

            for w_cand in scale_candidates:
                h_cand = w_cand / r
                if h_cand > Hc or w_cand > Wc:
                    continue

                x_max = max(0.0, Wc - w_cand)
                y_max = max(0.0, Hc - h_cand)

                x_candidates = np.linspace(0, x_max, position_grid_size)
                y_candidates = np.linspace(0, y_max, position_grid_size)
                x_candidates = np.unique(np.append(x_candidates, np.clip(cur_x, 0, x_max)))
                y_candidates = np.unique(np.append(y_candidates, np.clip(cur_y, 0, y_max)))

                for x_cand in x_candidates:
                    for y_cand in y_candidates:
                        candidate_bbox = _make_bbox(elem_idx, x_cand, y_cand, w_cand)

                        trial_bboxes = list(current_bboxes)
                        trial_bboxes[elem_idx] = candidate_bbox

                        # Phase 1: cheap bbox-only losses as early-exit filter
                        bbox_loss = _compute_bbox_only_loss(
                            trial_bboxes, Wc, Hc,
                            reference_bboxes, reference_parent_bbox,
                            w_similarity, size_rules, w_readability,
                            w_alignment_consistency, alignment_constraint,
                            w_alignment_similarity,
                            proximity_info, w_proximity, w_visual_balance,
                            overlap_constraints, _is_layer, _pen_pairs, device,
                        )
                        _elem_trials += 1

                        if bbox_loss >= best_elem_loss:
                            _elem_skips += 1
                            continue

                        # Phase 2: incremental mask-dependent losses
                        changed_area = None
                        if _is_layer and _rendered_mask_areas is not None:
                            changed_mask = _render_hard_mask(
                                sdf_tensors[elem_idx],
                                candidate_bbox[0], candidate_bbox[1],
                                candidate_bbox[2], candidate_bbox[3],
                                Hc, Wc, device,
                            )
                            changed_area = float(np.sum(changed_mask > 0.5))

                        mask_loss = _compute_mask_loss_incremental(
                            elem_idx, candidate_bbox,
                            w_overlap, w_data_ink, _is_layer,
                            _overlap_cache, _data_ink_cache,
                            _rendered_mask_areas, changed_area,
                        )

                        trial_loss = bbox_loss + mask_loss
                        if trial_loss < best_elem_loss:
                            best_elem_loss = trial_loss
                            best_elem_bbox = candidate_bbox

            _trial_count += _elem_trials
            _skip_count += _elem_skips
            _elem_elapsed = _time.time() - _t_elem
            _improved_tag = ""
            if best_elem_loss < best_loss:
                current_bboxes[elem_idx] = best_elem_bbox
                best_loss = best_elem_loss
                best_bboxes = list(current_bboxes)
                improved_this_round = True
                _improved_tag = " *improved*"
                # Update caches with the winning bbox
                if _overlap_cache is not None:
                    _overlap_cache.update(elem_idx, best_elem_bbox)
                if _data_ink_cache is not None:
                    _data_ink_cache.update(elem_idx, best_elem_bbox)
                if _is_layer and _rendered_mask_areas is not None:
                    m = _render_hard_mask(
                        sdf_tensors[elem_idx],
                        best_elem_bbox[0], best_elem_bbox[1],
                        best_elem_bbox[2], best_elem_bbox[3],
                        Hc, Wc, device,
                    )
                    _rendered_mask_areas[elem_idx] = float(np.sum(m > 0.5))
            print(f"  [GridSearch R{round_idx} E{elem_idx}/{num_nodes}] "
                  f"trials={_elem_trials}, skipped={_elem_skips}, loss={best_loss:.4f}, "
                  f"time={_elem_elapsed:.2f}s{_improved_tag}")

        elapsed_round = _time.time() - _t_round
        _skip_rate = (_skip_count / _trial_count * 100) if _trial_count > 0 else 0
        print(f"[GridSearch] Round {round_idx}/{num_rounds} done: "
              f"loss={best_loss:.4f}, improved={improved_this_round}, "
              f"time={elapsed_round:.2f}s, "
              f"trials={_trial_count}, skip_rate={_skip_rate:.1f}%")

        # Visualize after each round
        if debug and viz_folder:
            with torch.no_grad():
                X_viz, Y_viz = make_container_grid(Hc, Wc, device=device)
                viz_softmasks = []
                viz_bboxes_list = []
                for i, (bx, by, bw, bh) in enumerate(current_bboxes):
                    m, _ = sdf_to_softmask(
                        sdf_tensors[i],
                        torch.tensor(bx, device=device),
                        torch.tensor(by, device=device),
                        torch.tensor(bw, device=device),
                        torch.tensor(bh, device=device),
                        X_viz, Y_viz, tau_px=0.5,
                    )
                    viz_softmasks.append(m)
                    viz_bboxes_list.append((bx, by, bw, bh))
                loss_info = {'total': best_loss}
                save_path = os.path.join(viz_folder, f"round_{round_idx:02d}.png")
                visualize_optimization_progress(
                    viz_softmasks, viz_bboxes_list, Wc, Hc,
                    epoch=round_idx, loss_info=loss_info,
                    save_path=save_path, debug=debug,
                )

        if not improved_this_round:
            print(f"[GridSearch] Converged at round {round_idx} (no improvement)")
            break

    _total_elapsed = _time.time() - _t_opt
    final_bboxes = list(best_bboxes)
    _final_skip_rate = (_skip_count / _trial_count * 100) if _trial_count > 0 else 0
    print(f"[GridSearch] Done: final_loss={best_loss:.4f}, "
          f"total_trials={_trial_count}, skipped={_skip_count} ({_final_skip_rate:.1f}%), "
          f"elapsed={_total_elapsed:.2f}s")

    if debug:
        print("\n=== Grid Search Final Results ===")
        for i, bbox in enumerate(final_bboxes):
            print(f"bbox{i + 1} (x,y,w,h) = {bbox}")

    # Save composite image
    if debug and viz_folder:
        save_png_list = []
        for i in range(num_nodes):
            orig_png = original_png_list[i] if i < len(original_png_list) and original_png_list[i] is not None else png_list[i]
            save_png_list.append(orig_png)
        if all(p is not None for p in save_png_list):
            composite_path = os.path.join(viz_folder, "composite_result.png")
            save_composite_image(
                png_list=save_png_list, bbox_list=final_bboxes,
                Wc=Wc, Hc=Hc, save_path=composite_path, debug=debug,
            )

    print(f"[GridSearch timer] TOTAL optimize: {_time.time() - _t_total_start:.3f}s")
    return final_bboxes
