"""Main optimization function for SDF-based layout optimization."""

import os
import shutil
import time as _time
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F

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
    compute_fully_inside_loss,
    _get_pair_overlap_type,
)
from .visualization import (
    visualize_optimization_progress,
    visualize_final_result,
    save_composite_image,
)
from .utils import (
    check_all_non_overlap,
    normalize_reference_bboxes,
    unconstrained_from_reference_bboxes,
)


def optimize(
    png_list,  # List of image paths for N nodes
    original_png_list=None,  # List of original image paths for saving composite
    Wc=1000, Hc=1000,
    opt_res_list=params.OPT_RES_LIST,  # optimize on these resolutions
    outer_rounds=params.OUTER_ROUNDS,  # augmented-lagrangian outer updates per stage
    inner_steps=params.INNER_STEPS,    # gradient steps per outer round
    tau_schedule=params.TAU_SCHEDULE,
    rho_init=params.RHO_INIT,          # initial penalty parameter
    rho_mult=params.RHO_MULT,          # penalty multiplier
    lr=params.LEARNING_RATE,
    size_min=params.SIZE_MIN,         # Legacy parameter for backward compatibility
    min_sizes=None,                    # List of (min_width, min_height) tuples for each element
    pen_weight=params.PEN_WEIGHT,      # weight for penetration penalty
    pen_eta_px=params.PEN_ETA_PX,
    reference_bboxes=None,             # List of reference element bboxes (x, y, w, h) from Example layout
    reference_parent_bbox=None,        # Reference parent container bbox (x, y, w, h)
    w_similarity=params.W_SIMILARITY,  # weight for position/size similarity loss
    size_rules=None,                   # List of tuples (source_idx, target_idx) for size hierarchy rules
    w_readability=params.W_READABILITY, # weight for readability loss
    w_alignment_consistency=params.W_ALIGNMENT_CONSISTENCY,  # weight for alignment consistency loss
    alignment_constraint=None,          # Dictionary with alignment constraint from JSON (direction, value)
    w_alignment_similarity=params.W_ALIGNMENT_SIMILARITY,  # weight for alignment similarity loss
    proximity_info=None,               # Dictionary with container hierarchy info for proximity ratio calculation
    w_proximity=params.W_PROXIMITY,    # weight for proximity ratio loss
    w_data_ink=params.W_DATA_INK,      # weight for data ink loss (maximize union area)
    w_visual_balance=params.W_VISUAL_BALANCE,  # weight for visual balance loss
    min_gap_px=20.0,                    # minimum gap between elements in pixels
    overlap_constraints=None,
    chart_dual_masks=None,
    fully_overlap_pairs=None,
    w_fully_inside=params.W_FULLY_INSIDE,
    enable_grid_search_init=True,
    grid_search_config=None,
    initial_only=False,
    device=None,
    save_prefix=None,
    output_dir=None,
    debug=False,
):
    """Main SDF-based layout optimization function.
    
    Args:
        png_list: List of image paths for N nodes
        original_png_list: List of original image paths for saving composite
        Wc, Hc: Container width and height
        opt_res_list: Optimize on these resolutions
        outer_rounds: Augmented-lagrangian outer updates per stage
        inner_steps: Gradient steps per outer round
        tau_schedule: Schedule for tau parameter
        rho_init: Initial penalty parameter
        rho_mult: Penalty multiplier
        lr: Learning rate
        size_min: Legacy minimum size parameter
        min_sizes: List of (min_width, min_height) tuples for each element
        pen_weight: Weight for penetration penalty
        pen_eta_px: Eta parameter for penetration penalty
        reference_bboxes: List of reference element bboxes from Example layout
        reference_parent_bbox: Reference parent container bbox
        w_similarity: Weight for position/size similarity loss
        size_rules: List of tuples (source_idx, target_idx) for size hierarchy rules
        w_readability: Weight for readability loss
        w_alignment_consistency: Weight for alignment consistency loss
        alignment_constraint: Dictionary with alignment constraint from JSON
        w_alignment_similarity: Weight for alignment similarity loss
        proximity_info: Dictionary with container hierarchy info
        w_proximity: Weight for proximity ratio loss
        w_data_ink: Weight for data ink loss
        w_visual_balance: Weight for visual balance loss
        min_gap_px: Minimum gap between elements in pixels
        overlap_constraints: Overlap constraints from JSON
        chart_dual_masks: Dictionary of chart dual masks (no_grid, not_overlap)
        fully_overlap_pairs: List of (non_chart_idx, chart_idx) pairs
        w_fully_inside: Weight for fully inside loss
        enable_grid_search_init: Enable grid search initialization
        grid_search_config: Grid search configuration dictionary
        initial_only: Return strategy-consistent initialization bboxes without
            running optimization loops
        device: PyTorch device
        save_prefix: Prefix for saving result images
        debug: Enable debug mode (visualization and detailed logging)
    
    Returns:
        List of final bounding boxes [(x, y, w, h), ...]
    """
    _t_total_start = _time.time()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if debug:
        print("Device:", device)

    # Validate png_list
    if png_list is None or len(png_list) < 1:
        raise ValueError("png_list must be provided with at least one image path")
    
    num_nodes = len(png_list)
    
    # Handle original_png_list
    if original_png_list is None:
        original_png_list = png_list
    
    # Set dilation_radii to 100 for all elements
    dilation_radii = [15.0] * num_nodes
    
    if debug:
        print(f"Number of nodes: {num_nodes}")
        print(f"Dilation radii: {dilation_radii}")

    # Validate reference bboxes
    if reference_bboxes is None:
        reference_bboxes = []
    if reference_parent_bbox is None:
        reference_parent_bbox = (0.0, 0.0, float(Wc), float(Hc))
    
    reference_bboxes, reference_parent_bbox = normalize_reference_bboxes(
        reference_bboxes, reference_parent_bbox
    )
    if debug and reference_bboxes:
        print(f"Normalized reference bboxes ({len(reference_bboxes)} items)")
    
    if overlap_constraints is None:
        overlap_constraints = []
    if chart_dual_masks is None:
        chart_dual_masks = {}
    if fully_overlap_pairs is None:
        fully_overlap_pairs = []

    # Validate size rules
    if size_rules is None:
        size_rules = []
    
    # Validate and set up min_sizes
    if min_sizes is None:
        # Use default values for all elements
        min_sizes = [(params.MIN_WIDTH_DEFAULT, params.MIN_HEIGHT_DEFAULT)] * num_nodes
    elif len(min_sizes) < num_nodes:
        # Extend with default values
        default_size = (params.MIN_WIDTH_DEFAULT, params.MIN_HEIGHT_DEFAULT)
        min_sizes = min_sizes + [default_size] * (num_nodes - len(min_sizes))
    elif len(min_sizes) > num_nodes:
        # Truncate to num_nodes
        min_sizes = min_sizes[:num_nodes]
    
    if debug:
        print(f"Reference bboxes: {len(reference_bboxes)} elements")
        print(f"Reference parent bbox: {reference_parent_bbox}")
        print(f"Size rules: {len(size_rules)} rules")
        print(f"Similarity weight: {w_similarity}, Readability weight: {w_readability}")
        print(f"Min sizes: {min_sizes}")

    # Create visualization folder for this optimization run (only in debug mode)
    viz_folder = None
    if debug:
        if save_prefix:
            viz_folder = f"{save_prefix}_visualization"
        else:
            viz_folder = "optimization_visualization"
        
        # Directory structure: output_dir/save_prefix_visualization
        if output_dir is not None:
            viz_folder = os.path.join(output_dir, os.path.basename(viz_folder))
        # Clear existing folder contents if it exists
        if os.path.exists(viz_folder):
            shutil.rmtree(viz_folder)
        
        os.makedirs(viz_folder, exist_ok=True)
        if debug:
            print(f"Visualization folder created: {viz_folder}")

    # load masks for all nodes
    _t_mask = _time.time()
    masks = []
    ratios = []
    for i, png_path in enumerate(png_list):
        if png_path is None:
            raise ValueError(f"Image path at index {i} is None")
        mask = load_binary_mask_from_rgba(png_path)
        
        # Apply dilation with radius 100 to all elements
        mask = dilate_mask(mask, dilation_radii[i])
        if debug:
            print(f"Applied dilation to node {i} mask with radius {dilation_radii[i]:.1f} pixels")
        masks.append(mask)
        r, _ = tight_bbox_ratio(mask)
        ratios.append(r)
        if debug:
            print(f"Node {i} aspect ratio (tight alpha bbox): r={r:.4f}")
    print(f"[SDF timer] mask loading + dilation: {_time.time() - _t_mask:.3f}s")

    # precompute SDF templates for all nodes
    _t_sdf = _time.time()
    sdf_norms = []
    sdf_tensors = []
    for i, mask in enumerate(masks):
        sdf_norm = binary_to_sdf_norm(mask, pad=16)
        sdf_norms.append(sdf_norm)
        sdf_t = torch.from_numpy(sdf_norm)[None, None].to(device)
        sdf_tensors.append(sdf_t)
    print(f"[SDF timer] SDF template precompute: {_time.time() - _t_sdf:.3f}s")

    _t_dual = _time.time()
    sdf_no_grid_tensors = {}
    sdf_not_overlap_tensors = {}
    for chart_idx, paths in chart_dual_masks.items():
        no_grid_path = paths.get("no_grid_path")
        not_overlap_path = paths.get("not_overlap_path")
        if no_grid_path and os.path.exists(no_grid_path):
            mask_ng = load_binary_mask_from_rgba(no_grid_path)
            mask_ng = dilate_mask(mask_ng, dilation_radii[chart_idx] if chart_idx < len(dilation_radii) else 15.0)
            sdf_ng = binary_to_sdf_norm(mask_ng, pad=16)
            sdf_no_grid_tensors[chart_idx] = torch.from_numpy(sdf_ng)[None, None].to(device)
        if not_overlap_path and os.path.exists(not_overlap_path):
            mask_no = load_binary_mask_from_rgba(not_overlap_path)
            mask_no = dilate_mask(mask_no, dilation_radii[chart_idx] if chart_idx < len(dilation_radii) else 15.0)
            sdf_no = binary_to_sdf_norm(mask_no, pad=16)
            sdf_not_overlap_tensors[chart_idx] = torch.from_numpy(sdf_no)[None, None].to(device)
    print(f"[SDF timer] chart dual masks: {_time.time() - _t_dual:.3f}s")

    if grid_search_config is None:
        grid_search_config = {}
    if debug:
        print(f"enable grid search init: {enable_grid_search_init}")
    use_grid_search = enable_grid_search_init
    if use_grid_search:
        if num_nodes >= 3:
            use_grid_search = False
        else:
            all_non_overlap = check_all_non_overlap(overlap_constraints, num_nodes)
            if not all_non_overlap:
                if debug:
                    print("[Initialization] Grid search disabled: not all pairs are non-overlap")
                use_grid_search = False
    
    _t_init = _time.time()
    opt_params = []
    if use_grid_search:
        if debug:
            print("[Initialization] Using grid search initialization")
        
        from .grid_search_init import grid_search_initialization
        
        downscale_factor = grid_search_config.get("downscale_factor", 10)
        position_steps = grid_search_config.get("position_steps", 10)
        scale_steps = grid_search_config.get("scale_steps", 5)
        
        init_params, Wc, Hc = grid_search_initialization(
            masks=masks,
            sdf_tensors=sdf_tensors,
            ratios=ratios,
            Wc=Wc,
            Hc=Hc,
            min_sizes=min_sizes,
            reference_bboxes=reference_bboxes,
            downscale_factor=downscale_factor,
            position_steps=position_steps,
            scale_steps=scale_steps,
            device=device,
            size_min=size_min,
            debug=debug,
            viz_folder=viz_folder,
        )
        
        for i in range(num_nodes):
            tx_init, ty_init, ts_init = init_params[i]
            tx = torch.nn.Parameter(torch.tensor(tx_init, device=device))
            ty = torch.nn.Parameter(torch.tensor(ty_init, device=device))
            ts = torch.nn.Parameter(torch.tensor(ts_init, device=device))
            opt_params.extend([tx, ty, ts])
    
    elif reference_bboxes and len(reference_bboxes) >= num_nodes:
        if debug:
            print("[Initialization] Using reference bboxes")
        for i, (tx_init, ty_init, ts_init) in enumerate(
            unconstrained_from_reference_bboxes(
                reference_bboxes, num_nodes, min_sizes, ratios,
                Wc, Hc, size_min, debug=debug,
            )
        ):
            if debug:
                print(f"Node {i}: aspect ratio {ratios[i]:.4f}")
            tx = torch.nn.Parameter(torch.tensor(tx_init, device=device))
            ty = torch.nn.Parameter(torch.tensor(ty_init, device=device))
            ts = torch.nn.Parameter(torch.tensor(ts_init, device=device))
            opt_params.extend([tx, ty, ts])
    
    else:
        if debug:
            print("[Initialization] Using zeros (default)")
        
        for i in range(num_nodes):
            tx_init, ty_init, ts_init = 0.0, 0.0, 0.0
            tx = torch.nn.Parameter(torch.tensor(tx_init, device=device))
            ty = torch.nn.Parameter(torch.tensor(ty_init, device=device))
            ts = torch.nn.Parameter(torch.tensor(ts_init, device=device))
            opt_params.extend([tx, ty, ts])
    
    print(f"[SDF timer] initialization (grid_search={use_grid_search}): {_time.time() - _t_init:.3f}s")

    def _materialize_bboxes_from_params():
        bboxes = []
        for i in range(num_nodes):
            tx_idx = i * 3
            ty_idx = i * 3 + 1
            ts_idx = i * 3 + 2
            tx = opt_params[tx_idx]
            ty = opt_params[ty_idx]
            ts = opt_params[ts_idx]

            min_width, min_height = min_sizes[i]
            x, y, w, h = bbox_aspect_from_unconstrained(
                tx, ty, ts, Wc, Hc, ratios[i],
                min_width=min_width, min_height=min_height,
                size_min=size_min,
            )
            bboxes.append((x.item(), y.item(), w.item(), h.item()))
        return bboxes

    if initial_only:
        if debug:
            print("[Initialization] Returning initialization-only bboxes")
        return _materialize_bboxes_from_params()

    opt = torch.optim.Adam(opt_params, lr=lr)

    # augmented lagrangian multipliers for constraint g = A_inter = 0
    lam = torch.tensor(0.0, device=device)
    rho = torch.tensor(rho_init, device=device)

    tau_list = list(tau_schedule)
    if len(tau_list) < outer_rounds:
        tau_list += [tau_list[-1]] * (outer_rounds - len(tau_list))

    # Pre-compute pair overlap types to avoid repeated lookups in inner loop
    _non_overlap_pairs = []
    _pen_pairs = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            pair_type = _get_pair_overlap_type(overlap_constraints, i, j)
            if pair_type not in ("fully_overlap", "partially_overlap"):
                _non_overlap_pairs.append((i, j))
            if pair_type != "fully_overlap":
                _pen_pairs.append((i, j))

    # Pre-compute is_layer flag (constant across iterations)
    _is_layer = False
    if proximity_info:
        _container_types = proximity_info.get("types", [])
        if _container_types:
            _is_layer = (_container_types[0] == "layer")

    # Distribute outer_rounds across resolution stages.
    # Low-res stages serve as warm-up with fewer rounds; the final stage gets the rest.
    num_stages = len(opt_res_list)
    if num_stages == 1:
        _rounds_per_stage = [outer_rounds]
    else:
        warmup_rounds = max(1, outer_rounds // 4)
        _rounds_per_stage = [warmup_rounds] * (num_stages - 1)
        _rounds_per_stage.append(outer_rounds - warmup_rounds * (num_stages - 1))

    # staged optimization over resolutions
    for stage_idx, stage_res in enumerate(opt_res_list):
        _t_stage = _time.time()
        Hs = Ws = int(stage_res)
        X, Y = make_container_grid(Hs, Ws, device=device)
        stage_outer_rounds = _rounds_per_stage[stage_idx]
        if debug:
            print(f"\n=== Stage optimize at {Ws}x{Hs} (container {Wc}x{Hc}), "
                  f"outer_rounds={stage_outer_rounds} ===")

        # Visualize initial state before optimization (only for first stage and only in debug mode)
        if stage_idx == 0 and debug:
            with torch.no_grad():
                # Use full resolution for visualization
                X_init, Y_init = make_container_grid(Hc, Wc, device=device)
                initial_bboxes = []
                initial_softmasks = []
                for i in range(num_nodes):
                    tx_idx = i * 3
                    ty_idx = i * 3 + 1
                    ts_idx = i * 3 + 2
                    tx = opt_params[tx_idx]
                    ty = opt_params[ty_idx]
                    ts = opt_params[ts_idx]
                    
                    min_width, min_height = min_sizes[i]
                    x, y, w, h = bbox_aspect_from_unconstrained(
                        tx, ty, ts, Wc, Hc, ratios[i],
                        min_width=min_width, min_height=min_height,
                        size_min=size_min
                    )
                    if debug:
                        print(f"Node {i} initial bbox: ({x.item():.1f}, {y.item():.1f}, {w.item():.1f}, {h.item():.1f})")
                    initial_bboxes.append((x.item(), y.item(), w.item(), h.item()))
                    
                    # Use smaller tau for initial visualization to show actual mask shape
                    m, _ = sdf_to_softmask(sdf_tensors[i], x, y, w, h, X_init, Y_init, tau_px=0.5)
                    initial_softmasks.append(m)
                
                # Compute initial loss values for display
                initial_union = torch.ones_like(initial_softmasks[0])
                for m in initial_softmasks:
                    initial_union = initial_union * (1.0 - m)
                initial_union = 1.0 - initial_union
                initial_A_union = initial_union.sum().item()  # Full resolution, da=1
                
                initial_inter = torch.zeros_like(initial_softmasks[0])
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        initial_inter = initial_inter + initial_softmasks[i] * initial_softmasks[j]
                initial_A_inter = initial_inter.sum().item()  # Full resolution, da=1
                
                # Compute visual balance loss for initial state (using bboxes)
                initial_bboxes_tensors = []
                for i in range(num_nodes):
                    tx_idx = i * 3
                    ty_idx = i * 3 + 1
                    ts_idx = i * 3 + 2
                    tx = opt_params[tx_idx]
                    ty = opt_params[ty_idx]
                    ts = opt_params[ts_idx]
                    
                    min_width, min_height = min_sizes[i]
                    x, y, w, h = bbox_aspect_from_unconstrained(
                        tx, ty, ts, Wc, Hc, ratios[i],
                        min_width=min_width, min_height=min_height,
                        size_min=size_min
                    )
                    initial_bboxes_tensors.append((x, y, w, h))
                
                L_visual_balance_init = compute_visual_balance_loss(
                    initial_bboxes_tensors, Wc, Hc, masks=initial_softmasks, device=device
                )
                
                initial_loss_info = {
                    'A_union': initial_A_union,
                    'A_inter': initial_A_inter,
                    'visual_balance': w_visual_balance * L_visual_balance_init.item(),
                }
                
                initial_save_path = os.path.join(viz_folder, f"initial_stage{stage_idx}.png")
                
                visualize_optimization_progress(
                    initial_softmasks, initial_bboxes, Wc, Hc,
                    epoch=-1, loss_info=initial_loss_info,
                    save_path=initial_save_path, debug=debug
                )

        _acc_softmask = 0.0
        _acc_union_inter = 0.0
        _acc_losses = 0.0
        _acc_backward = 0.0
        _acc_al_update = 0.0
        _acc_debug_viz = 0.0
        _total_inner_iters = 0

        for k in range(stage_outer_rounds):
            tau_px = float(tau_list[min(k, len(tau_list)-1)])
            align_factor = (k + 1) / stage_outer_rounds
            w_align_cons_eff = w_alignment_consistency * align_factor
            w_align_sim_eff = w_alignment_similarity * align_factor
            per_loss_grads = None

            for t in range(inner_steps):
                _total_inner_iters += 1
                opt.zero_grad(set_to_none=True)

                # Compute bboxes for all nodes
                _t_sm = _time.time()
                bboxes = []
                softmasks = []
                distances = []
                for i in range(num_nodes):
                    tx_idx = i * 3
                    ty_idx = i * 3 + 1
                    ts_idx = i * 3 + 2
                    tx = opt_params[tx_idx]
                    ty = opt_params[ty_idx]
                    ts = opt_params[ts_idx]
                    
                    min_width, min_height = min_sizes[i]
                    x, y, w, h = bbox_aspect_from_unconstrained(
                        tx, ty, ts, Wc, Hc, ratios[i],
                        min_width=min_width, min_height=min_height,
                        size_min=size_min
                    )
                    bboxes.append((x, y, w, h))
                    
                    m, d_px = sdf_to_softmask(sdf_tensors[i], x, y, w, h, X, Y, tau_px=tau_px)
                    softmasks.append(m)
                    distances.append(d_px)

                _acc_softmask += _time.time() - _t_sm

                # Compute union: 1 - product of (1 - mask_i)
                _t_ui = _time.time()
                union = torch.ones_like(softmasks[0])
                for m in softmasks:
                    union = union * (1.0 - m)
                union = 1.0 - union

                # Compute intersection: sum of pairwise intersections (exclude fully_overlap/partially_overlap pairs)
                inter = torch.zeros_like(softmasks[0])
                for i, j in _non_overlap_pairs:
                    inter = inter + softmasks[i] * softmasks[j]

                A_union = area_sum(union, Wc, Hc)
                A_inter = area_sum(inter, Wc, Hc)
                _acc_union_inter += _time.time() - _t_ui

                _t_loss = _time.time()

                if _is_layer:
                    A_sum = sum(area_sum(m, Wc, Hc) for m in softmasks)
                    L_data_ink = -A_sum
                    if debug and t == 0 and k == 0:
                        print(f"[data_ink] using A_sum (is_layer=True), A_sum={A_sum.item():.2f}")
                else:
                    L_data_ink = -A_union
                    if debug and t == 0 and k == 0:
                        print(f"[data_ink] using A_union (is_layer=False), A_union={A_union.item():.2f}")

                L_visual_balance = compute_visual_balance_loss(
                    bboxes, Wc, Hc, masks=softmasks, device=device
                )

                # Build stacked bbox tensors once and reuse for all losses
                gen_bbox_tensors = [torch.stack([bbox[0], bbox[1], bbox[2], bbox[3]]) for bbox in bboxes]
                gen_parent_bbox = (0.0, 0.0, float(Wc), float(Hc))

                # Penetration loss (skip for layer containers)
                L_pen = torch.tensor(0.0, device=device)
                if not _is_layer:
                    for i, j in _pen_pairs:
                        x_i, y_i, w_i, h_i = bboxes[i]
                        x_j, y_j, w_j, h_j = bboxes[j]
                        gap_x = torch.max(x_j - (x_i + w_i), x_i - (x_j + w_j))
                        gap_y = torch.max(y_j - (y_i + h_i), y_i - (y_j + h_j))
                        if gap_x >= 0 and gap_y >= 0:
                            gap = torch.sqrt(gap_x * gap_x + gap_y * gap_y)
                        elif gap_x < 0 and gap_y < 0:
                            gap = torch.max(gap_x, gap_y)
                        else:
                            gap = torch.max(gap_x, gap_y)
                        L_pen = L_pen + F.relu(min_gap_px - gap)

                # Position/Size similarity loss
                L_similarity = torch.tensor(0.0, device=device)
                if reference_bboxes and len(reference_bboxes) >= num_nodes:
                    L_similarity = compute_position_size_similarity_loss(
                        reference_bboxes[:num_nodes],
                        gen_bbox_tensors,
                        reference_parent_bbox,
                        gen_parent_bbox,
                        device=device
                    )

                # Readability loss (size hierarchy consistency)
                L_readability = torch.tensor(0.0, device=device)
                if size_rules and len(size_rules) > 0:
                    L_readability = compute_readability_loss(
                        size_rules,
                        gen_bbox_tensors,
                        size_ratio_threshold=params.SIZE_RATIO_THRESHOLD,
                        device=device
                    )

                # Alignment consistency loss (hierarchical alignment)
                L_alignment_consistency = torch.tensor(0.0, device=device)
                if reference_bboxes and len(reference_bboxes) >= num_nodes:
                    L_alignment_consistency = compute_alignment_consistency_loss(
                        reference_bboxes[:num_nodes],
                        gen_bbox_tensors,
                        reference_parent_bbox,
                        gen_parent_bbox,
                        device=device
                    )
                
                # Alignment similarity loss (based on JSON constraint)
                L_alignment_similarity = torch.tensor(0.0, device=device)
                if alignment_constraint and w_alignment_similarity > 0:
                    L_alignment_similarity = compute_alignment_similarity_loss(
                        gen_bbox_tensors,
                        gen_parent_bbox,
                        alignment_constraint,
                        device=device
                    )

                # Proximity ratio loss
                L_proximity = torch.tensor(0.0, device=device)
                if proximity_info and w_proximity > 0:
                    container_bboxes = proximity_info.get("containers", [])
                    child_bboxes_list = proximity_info.get("children", [])
                    grandchild_bboxes_list = proximity_info.get("grandchildren", [])
                    container_types = proximity_info.get("types", [])
                    container_weights = proximity_info.get("weights", None)
                    
                    if len(container_bboxes) == 0:
                        container_bbox = gen_parent_bbox
                        grandchild_bboxes = []
                        container_type = container_types[0] if container_types else "row"
                        
                        L_proximity = compute_proximity_ratio_loss(
                            [container_bbox],
                            [gen_bbox_tensors],
                            [grandchild_bboxes],
                            [container_type],
                            container_weights=[1.0] if container_weights is None else container_weights,
                            epsilon=params.PROXIMITY_EPSILON,
                            device=device
                        )
                    else:
                        if len(child_bboxes_list) > 0 and len(child_bboxes_list[0]) == num_nodes:
                            updated_child_bboxes_list = [gen_bbox_tensors] + child_bboxes_list[1:]
                        else:
                            updated_child_bboxes_list = child_bboxes_list
                        
                        L_proximity = compute_proximity_ratio_loss(
                            container_bboxes,
                            updated_child_bboxes_list,
                            grandchild_bboxes_list,
                            container_types,
                            container_weights,
                            epsilon=params.PROXIMITY_EPSILON,
                            device=device
                        )

                L_fully_inside = torch.tensor(0.0, device=device)
                if fully_overlap_pairs and w_fully_inside > 0 and sdf_no_grid_tensors and sdf_not_overlap_tensors:
                    chart_no_grid_soft = {}
                    chart_not_overlap_soft = {}
                    for chart_idx in set(c for _, c in fully_overlap_pairs):
                        if chart_idx not in sdf_no_grid_tensors or chart_idx not in sdf_not_overlap_tensors:
                            continue
                        x, y, w, h = bboxes[chart_idx]
                        m_ng, _ = sdf_to_softmask(sdf_no_grid_tensors[chart_idx], x, y, w, h, X, Y, tau_px=tau_px)
                        m_no, _ = sdf_to_softmask(sdf_not_overlap_tensors[chart_idx], x, y, w, h, X, Y, tau_px=tau_px)
                        chart_no_grid_soft[chart_idx] = m_ng
                        chart_not_overlap_soft[chart_idx] = m_no
                    L_fully_inside = compute_fully_inside_loss(
                        softmasks, chart_no_grid_soft, chart_not_overlap_soft,
                        fully_overlap_pairs, Wc, Hc, device=device
                    )

                g = A_inter
                # # Data ink loss: maximize union area (minimize white space)
                # # Negative because we want to maximize A_union (minimize -A_union)
                # L_data_ink = -A_union
                
                # Scale pen_weight by rho to prevent L_pen from being overwhelmed when rho is large
                # When rho is large, AL constraint dominates, so we need to scale pen_weight accordingly
                pen_weight_scaled = pen_weight * (1.0 + rho.item() / 1e4)
                
                loss = (lam * g + 0.5 * rho * g * g + pen_weight_scaled * L_pen +
                        w_similarity * L_similarity + w_readability * L_readability +
                        w_align_cons_eff * L_alignment_consistency + w_align_sim_eff * L_alignment_similarity +
                        w_proximity * L_proximity + w_data_ink * L_data_ink + w_visual_balance * L_visual_balance +
                        w_fully_inside * L_fully_inside)

                _acc_losses += _time.time() - _t_loss

                _t_bw = _time.time()
                per_loss_grads = None
                if debug and t == inner_steps - 1:
                    loss_terms = [
                        ("AL", lam * g + 0.5 * rho * g * g),
                        ("pen", pen_weight_scaled * L_pen),
                        ("sim", w_similarity * L_similarity),
                        ("read", w_readability * L_readability),
                        ("align_cons", w_align_cons_eff * L_alignment_consistency),
                        ("align_sim", w_align_sim_eff * L_alignment_similarity),
                        ("prox", w_proximity * L_proximity),
                        ("data_ink", w_data_ink * L_data_ink),
                        ("balance", w_visual_balance * L_visual_balance),
                        ("fully_inside", w_fully_inside * L_fully_inside),
                    ]
                    per_loss_grads = {}
                    n_params = len(opt_params)
                    for name, term in loss_terms:
                        if term.requires_grad and term.grad_fn is not None:
                            grads = torch.autograd.grad(term, opt_params, retain_graph=True, allow_unused=True)
                            per_loss_grads[name] = [g.item() if g is not None else 0.0 for g in grads]
                        else:
                            per_loss_grads[name] = [0.0] * n_params

                loss.backward()
                opt.step()
                _acc_backward += _time.time() - _t_bw

            # outer AL update
            _t_al = _time.time()
            with torch.no_grad():
                # Compute bboxes for logging and AL update
                bboxes_log = []
                for i in range(num_nodes):
                    tx_idx = i * 3
                    ty_idx = i * 3 + 1
                    ts_idx = i * 3 + 2
                    tx = opt_params[tx_idx]
                    ty = opt_params[ty_idx]
                    ts = opt_params[ts_idx]
                    
                    min_width, min_height = min_sizes[i]
                    x, y, w, h = bbox_aspect_from_unconstrained(
                        tx, ty, ts, Wc, Hc, ratios[i],
                        min_width=min_width, min_height=min_height,
                        size_min=size_min
                    )
                    bboxes_log.append((x, y, w, h))
                
                # Compute A_inter for AL update (always needed)
                softmasks_log = []
                for i in range(num_nodes):
                    x, y, w, h = bboxes_log[i]
                    m, d_px = sdf_to_softmask(sdf_tensors[i], x, y, w, h, X, Y, tau_px=tau_px)
                    softmasks_log.append(m)
                
                inter_log = torch.zeros_like(softmasks_log[0])
                for i, j in _non_overlap_pairs:
                    inter_log = inter_log + softmasks_log[i] * softmasks_log[j]
                
                A_inter = area_sum(inter_log, Wc, Hc)
                
                # Update Lagrangian multiplier
                lam = lam + rho * A_inter
                rho = rho * rho_mult
                
                # Recompute canvas size to tightly fit current bboxes
                min_x = min(bbox[0].item() for bbox in bboxes_log)
                min_y = min(bbox[1].item() for bbox in bboxes_log)
                max_x = max((bbox[0] + bbox[2]).item() for bbox in bboxes_log)
                max_y = max((bbox[1] + bbox[3]).item() for bbox in bboxes_log)
                
                # Add some padding
                padding = 50
                new_Wc = int(max_x - min_x + 2 * padding)
                new_Hc = int(max_y - min_y + 2 * padding)
                
                # Update container size and grid if changed significantly
                if abs(new_Wc - Wc) > 0.1 * Wc or abs(new_Hc - Hc) > 0.1 * Hc:
                    Wc, Hc = new_Wc, new_Hc
                    X, Y = make_container_grid(Hs, Ws, device=device)
                    if debug:
                        print(f"[Epoch {k}] Updated canvas size to {Wc}x{Hc}")
                
                # Detailed logging and visualization (only in debug mode)
                if debug:
                    # Compute all loss components for detailed logging
                    distances_log = []
                    for i in range(num_nodes):
                        x, y, w, h = bboxes_log[i]
                        m, d_px = sdf_to_softmask(sdf_tensors[i], x, y, w, h, X, Y, tau_px=tau_px)
                        distances_log.append(d_px)
                    
                    union_log = torch.ones_like(softmasks_log[0])
                    for m in softmasks_log:
                        union_log = union_log * (1.0 - m)
                    union_log = 1.0 - union_log
                    A_union = area_sum(union_log, Wc, Hc)

                    A_union = area_sum(union_log, Wc, Hc)

                    L_pen_val = torch.tensor(0.0, device=device)
                    if not _is_layer:
                        for i, j in _pen_pairs:
                            x_i, y_i, w_i, h_i = bboxes_log[i]
                            x_j, y_j, w_j, h_j = bboxes_log[j]
                            gap_x = torch.max(x_j - (x_i + w_i), x_i - (x_j + w_j))
                            gap_y = torch.max(y_j - (y_i + h_i), y_i - (y_j + h_j))
                            if gap_x >= 0 and gap_y >= 0:
                                gap = torch.sqrt(gap_x * gap_x + gap_y * gap_y)
                            elif gap_x < 0 and gap_y < 0:
                                gap = torch.max(gap_x, gap_y)
                            else:
                                gap = torch.max(gap_x, gap_y)
                            L_pen_val = L_pen_val + F.relu(min_gap_px - gap)

                    gen_bbox_tensors_log = [torch.stack([bbox[0], bbox[1], bbox[2], bbox[3]]) for bbox in bboxes_log]
                    gen_parent_bbox_log = (0.0, 0.0, float(Wc), float(Hc))

                    L_similarity_val = torch.tensor(0.0, device=device)
                    if reference_bboxes and len(reference_bboxes) >= num_nodes:
                        L_similarity_val = compute_position_size_similarity_loss(
                            reference_bboxes[:num_nodes],
                            gen_bbox_tensors_log,
                            reference_parent_bbox,
                            gen_parent_bbox_log,
                            device=device
                        )

                    L_readability_val = torch.tensor(0.0, device=device)
                    if size_rules and len(size_rules) > 0:
                        L_readability_val = compute_readability_loss(
                            size_rules,
                            gen_bbox_tensors_log,
                            size_ratio_threshold=params.SIZE_RATIO_THRESHOLD,
                            device=device
                        )

                    L_alignment_consistency_val = torch.tensor(0.0, device=device)
                    if reference_bboxes and len(reference_bboxes) >= num_nodes:
                        L_alignment_consistency_val = compute_alignment_consistency_loss(
                            reference_bboxes[:num_nodes],
                            gen_bbox_tensors_log,
                            reference_parent_bbox,
                            gen_parent_bbox_log,
                            device=device
                        )
                    
                    L_alignment_similarity_val = torch.tensor(0.0, device=device)
                    if alignment_constraint and w_alignment_similarity > 0:
                        L_alignment_similarity_val = compute_alignment_similarity_loss(
                            gen_bbox_tensors_log,
                            gen_parent_bbox_log,
                            alignment_constraint,
                            device=device
                        )

                    # Recompute proximity ratio loss for logging
                    L_proximity_val = torch.tensor(0.0, device=device)
                    if proximity_info and w_proximity > 0:
                        container_bboxes = proximity_info.get("containers", [])
                        child_bboxes_list = proximity_info.get("children", [])
                        grandchild_bboxes_list = proximity_info.get("grandchildren", [])
                        container_types = proximity_info.get("types", [])
                        container_weights = proximity_info.get("weights", None)
                        
                        if len(container_bboxes) == 0:
                            container_bbox = (0.0, 0.0, float(Wc), float(Hc))
                            child_bboxes = [(bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()) for bbox in bboxes_log]
                            grandchild_bboxes = []
                            container_type = container_types[0] if container_types else "row"  # Default to row
                            
                            L_proximity_val = compute_proximity_ratio_loss(
                                [container_bbox],
                                [child_bboxes],
                                [grandchild_bboxes],
                                [container_type],
                                container_weights=[1.0] if container_weights is None else container_weights,
                                epsilon=params.PROXIMITY_EPSILON,
                                device=device
                            )
                        else:
                            generated_bboxes_proximity_log = [(bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()) for bbox in bboxes_log]
                            if len(child_bboxes_list) > 0 and len(child_bboxes_list[0]) == num_nodes:
                                updated_child_bboxes_list_log = [generated_bboxes_proximity_log] + child_bboxes_list[1:]
                            else:
                                updated_child_bboxes_list_log = child_bboxes_list
                            
                            L_proximity_val = compute_proximity_ratio_loss(
                                container_bboxes,
                                updated_child_bboxes_list_log,
                                grandchild_bboxes_list,
                                container_types,
                                container_weights,
                                epsilon=params.PROXIMITY_EPSILON,
                                device=device
                            )
                    
                    if _is_layer:
                        A_sum_log = sum(area_sum(m, Wc, Hc) for m in softmasks_log)
                        L_data_ink_val = -A_sum_log
                        data_ink_src = "A_sum"
                    else:
                        L_data_ink_val = -A_union
                        data_ink_src = "A_union"

                    L_visual_balance_val = compute_visual_balance_loss(
                        bboxes_log, Wc, Hc, masks=softmasks_log, device=device
                    )
                    
                    # Compute total loss for visualization
                    g_val = A_inter
                    total_loss_val = (lam.item() * g_val.item() + 0.5 * rho.item() * g_val.item() * g_val.item() +
                                     pen_weight * L_pen_val.item() +
                                     w_similarity * L_similarity_val.item() +
                                     w_readability * L_readability_val.item() +
                                     w_align_cons_eff * L_alignment_consistency_val.item() +
                                     w_align_sim_eff * L_alignment_similarity_val.item() +
                                     w_proximity * L_proximity_val.item() +
                                     w_data_ink * L_data_ink_val.item() +
                                     w_visual_balance * L_visual_balance_val.item())
                    
                    print(f"[outer {k:02d}] tau={tau_px:.3f}  data_ink_src={data_ink_src}  A_union={A_union.item():.2f}  A_inter={A_inter.item():.6f}  "
                          f"L_pen={L_pen_val.item():.4f}  L_sim={w_similarity*L_similarity_val.item():.4f}  "
                          f"L_read={w_readability*L_readability_val.item():.4f}  "
                          f"L_align_cons={w_align_cons_eff*L_alignment_consistency_val.item():.4f}  "
                          f"L_align_sim={w_align_sim_eff*L_alignment_similarity_val.item():.4f}  "
                          f"L_prox={w_proximity*L_proximity_val.item():.4f}  "
                          f"L_data_ink={w_data_ink*L_data_ink_val.item():.4f}  "
                          f"L_balance={w_visual_balance*L_visual_balance_val.item():.4f}  "
                          f"lam={lam.item():.3e}  rho={rho.item():.3e}")
                    if per_loss_grads is not None:
                        print("[grad decomposition] per-node (tx, ty, ts) from each loss:")
                        for name, grads in per_loss_grads.items():
                            parts = []
                            for i in range(num_nodes):
                                tx_g, ty_g, ts_g = grads[i*3], grads[i*3+1], grads[i*3+2]
                                parts.append(f"N{i}:({tx_g:+.2e},{ty_g:+.2e},{ts_g:+.2e})")
                            print(f"  {name}: {' '.join(parts)}")

                    # Visualize optimization progress at end of each outer epoch
                    # Use full resolution for visualization
                    X_viz, Y_viz = make_container_grid(Hc, Wc, device=device)
                    epoch_bboxes = []
                    epoch_softmasks = []
                    for i, bbox_log in enumerate(bboxes_log):
                        x, y, w, h = bbox_log
                        epoch_bboxes.append((x.item(), y.item(), w.item(), h.item()))
                        m_viz, _ = sdf_to_softmask(sdf_tensors[i], x, y, w, h, X_viz, Y_viz, tau_px=tau_px)
                        epoch_softmasks.append(m_viz)
                    
                    # Collect gradients
                    epoch_gradients = []
                    for i in range(num_nodes):
                        tx_idx = i * 3
                        ty_idx = i * 3 + 1
                        ts_idx = i * 3 + 2
                        grads = {
                            'tx': opt_params[tx_idx].grad.item() if opt_params[tx_idx].grad is not None else 0.0,
                            'ty': opt_params[ty_idx].grad.item() if opt_params[ty_idx].grad is not None else 0.0,
                            'ts': opt_params[ts_idx].grad.item() if opt_params[ts_idx].grad is not None else 0.0,
                        }
                        epoch_gradients.append(grads)
                    
                    epoch_loss_info = {
                        'total': total_loss_val,
                        'A_union': A_union.item(),
                        'A_inter': A_inter.item(),
                        'pen': L_pen_val.item(),
                        'similarity': w_similarity * L_similarity_val.item(),
                        'readability': w_readability * L_readability_val.item(),
                        'alignment': w_align_cons_eff * L_alignment_consistency_val.item() + w_align_sim_eff * L_alignment_similarity_val.item(),
                        'alignment_consistency': w_align_cons_eff * L_alignment_consistency_val.item(),
                        'alignment_similarity': w_align_sim_eff * L_alignment_similarity_val.item(),
                        'proximity': w_proximity * L_proximity_val.item(),
                        'data_ink': w_data_ink * L_data_ink_val.item(),
                    }
                    
                    epoch_save_path = os.path.join(viz_folder, f"stage{stage_idx}_epoch{k:02d}.png")
                    
                    visualize_optimization_progress(
                        epoch_softmasks, epoch_bboxes, Wc, Hc,
                        epoch=k, loss_info=epoch_loss_info,
                        save_path=epoch_save_path,
                        gradients=epoch_gradients, debug=debug
                    )
                else:
                    pass
                _acc_al_update += _time.time() - _t_al

        _t_stage_elapsed = _time.time() - _t_stage
        print(f"[SDF timer] stage {stage_idx} (res={stage_res}) total: {_t_stage_elapsed:.3f}s  "
              f"iters={_total_inner_iters}")
        print(f"[SDF timer]   softmask:      {_acc_softmask:.3f}s")
        print(f"[SDF timer]   union/inter:   {_acc_union_inter:.3f}s")
        print(f"[SDF timer]   losses:        {_acc_losses:.3f}s")
        print(f"[SDF timer]   backward+step: {_acc_backward:.3f}s")
        print(f"[SDF timer]   AL update+viz: {_acc_al_update:.3f}s")

    # final bbox (continuous)
    _t_final = _time.time()
    final_bboxes = []
    with torch.no_grad():
        for i in range(num_nodes):
            tx_idx = i * 3
            ty_idx = i * 3 + 1
            ts_idx = i * 3 + 2
            tx = opt_params[tx_idx]
            ty = opt_params[ty_idx]
            ts = opt_params[ts_idx]
            
            min_width, min_height = min_sizes[i]
            x, y, w, h = bbox_aspect_from_unconstrained(
                tx, ty, ts, Wc, Hc, ratios[i],
                min_width=min_width, min_height=min_height,
                size_min=size_min
            )
            final_bboxes.append((x.item(), y.item(), w.item(), h.item()))

    # hard evaluation at full 1000x1000: overlap using (SDF<0) AND
    with torch.no_grad():
        Xf, Yf = make_container_grid(Hc, Wc, device=device)
        # use a small tau for union display (not needed for hard overlap)
        softmasks_f = []
        distances_f = []
        for i in range(num_nodes):
            x, y, w, h = final_bboxes[i]
            m, d_px = sdf_to_softmask(sdf_tensors[i], 
                                     torch.tensor(x, device=device), 
                                     torch.tensor(y, device=device),
                                     torch.tensor(w, device=device), 
                                     torch.tensor(h, device=device), 
                                     Xf, Yf, tau_px=0.2)
            softmasks_f.append(m)
            distances_f.append(d_px)

        # Compute union
        union_f = torch.ones_like(softmasks_f[0])
        for m in softmasks_f:
            union_f = union_f * (1.0 - m)
        union_f = 1.0 - union_f
        A_union_f = union_f.sum().item()  # da=1 at full res
        
        # hard inside test: d_px < 0 for all pairs
        hard_overlap = torch.zeros_like(softmasks_f[0])
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                overlap_ij = ((distances_f[i] < 0.0) & (distances_f[j] < 0.0)).float()
                hard_overlap = hard_overlap + overlap_ij
        A_overlap_hard = hard_overlap.sum().item()

    if debug:
        print("\n=== Final Results ===")
        for i, bbox in enumerate(final_bboxes):
            print(f"bbox{i+1} (x,y,w,h) = {bbox}")
        print(f"Union area (approx, {Wc}x{Hc}) = {A_union_f:.2f}  -> ratio {A_union_f/(Wc*Hc):.4f}")
        print(f"Hard overlap area (SDF<0)       = {A_overlap_hard:.0f} pixels")

    # Save visualization and composite images (only in debug mode)
    if debug:
        # Determine save paths based on prefix (save to visualization folder)
        if save_prefix:
            final_result_path = os.path.join(viz_folder, f"{save_prefix}_final_result.png")
            composite_result_path = os.path.join(viz_folder, f"{save_prefix}_composite_result.png")
        else:
            final_result_path = os.path.join(viz_folder, "final_result.png")
            composite_result_path = os.path.join(viz_folder, "composite_result.png")
        
        # Visualize final result (only for 2 nodes, skip for N nodes)
        # TODO: Extend visualize_final_result to support N nodes
        if num_nodes == 2:
            mask1_orig = load_binary_mask_from_rgba(png_list[0])
            mask2_orig = load_binary_mask_from_rgba(png_list[1])
            if mask1_orig is not None and mask2_orig is not None:
                visualize_final_result(softmasks_f[0], softmasks_f[1], distances_f[0], distances_f[1], 
                                      final_bboxes[0], final_bboxes[1], 
                                      mask1_orig, mask2_orig, Wc, Hc, 
                                      save_path=final_result_path, debug=debug)
        
        # Save composite image with original images
        # Use original paths if provided, otherwise use the paths passed to optimize
        save_png_list = []
        for i in range(num_nodes):
            orig_png = original_png_list[i] if i < len(original_png_list) and original_png_list[i] is not None else png_list[i]
            save_png_list.append(orig_png)
        
        if all(png is not None for png in save_png_list):
            save_composite_image(png_list=save_png_list, bbox_list=final_bboxes, Wc=Wc, Hc=Hc, 
                               save_path=composite_result_path, debug=debug)
        else:
            if debug:
                print(f"Warning: Skipping composite image save - some image paths are None")

    print(f"[SDF timer] final eval + viz: {_time.time() - _t_final:.3f}s")
    print(f"[SDF timer] TOTAL optimize: {_time.time() - _t_total_start:.3f}s")
    return final_bboxes

