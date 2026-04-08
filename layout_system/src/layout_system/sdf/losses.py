"""Loss functions for SDF-based layout optimization."""

from typing import List, Tuple, Optional, Union
import numpy as np
import torch


# -----------------------------
# Alignment Consistency Loss (Hierarchical Alignment)
# -----------------------------
def compute_alignment_consistency_loss(
    reference_bboxes: List[Tuple[float, float, float, float]],
    generated_bboxes: List[torch.Tensor],
    reference_parent_bbox: Tuple[float, float, float, float],
    generated_parent_bbox: Tuple[float, float, float, float],
    device: str = "cpu"
) -> torch.Tensor:
    if len(generated_bboxes) < 2:
        return torch.tensor(0.0, device=device)
    
    bbox_tensor = torch.stack([bbox for bbox in generated_bboxes])
    
    x = bbox_tensor[:, 0]
    y = bbox_tensor[:, 1]
    w = bbox_tensor[:, 2]
    h = bbox_tensor[:, 3]
    
    left_coords = x
    right_coords = x + w
    center_x_coords = x + w / 2.0
    top_coords = y
    bottom_coords = y + h
    center_y_coords = y + h / 2.0
    
    horizontal_coords = torch.stack([
        left_coords,
        right_coords,
        center_x_coords
    ])
    
    vertical_coords = torch.stack([
        top_coords,
        bottom_coords,
        center_y_coords
    ])
    
    sigma_horizontal = torch.std(horizontal_coords, dim=1)
    sigma_vertical = torch.std(vertical_coords, dim=1)
    
    penalty = torch.min(sigma_horizontal) + torch.min(sigma_vertical)
    
    return penalty


# -----------------------------
# Alignment Similarity Loss (Based on JSON Constraint)
# -----------------------------
def compute_alignment_similarity_loss(
    generated_bboxes: List[torch.Tensor],
    container_bbox: Tuple[float, float, float, float],
    alignment_constraint: dict,
    device: str = "cpu"
) -> torch.Tensor:
    """Compute alignment similarity loss based on JSON alignment constraint.
    
    For row/column layouts, JSON specifies which axis to align elements along,
    rather than taking the minimum loss across all alignment possibilities.
    
    Args:
        generated_bboxes: List of generated element bboxes as torch tensors (x, y, w, h)
        container_bbox: Container bbox (x, y, w, h)
        alignment_constraint: Dictionary with "direction" and "value" keys
            - direction: "horizontal" or "vertical"
            - value: "left", "center", "right" (for horizontal) or "top", "center", "bottom" (for vertical)
        device: Device for tensor operations
    
    Returns:
        Alignment loss tensor
    """
    if len(generated_bboxes) < 2:
        return torch.tensor(0.0, device=device)
    
    if not alignment_constraint:
        return torch.tensor(0.0, device=device)
    
    direction = alignment_constraint.get("direction", "horizontal")
    value = alignment_constraint.get("value", "center")
    
    _, _, container_w, container_h = container_bbox
    
    L_alignment = torch.tensor(0.0, device=device)
    
    if direction == "horizontal":
        if value == "left":
            left_coords = []
            for bbox in generated_bboxes:
                x = bbox[0]
                left_coords.append(x)
            left_tensor = torch.stack(left_coords)
            L_alignment = torch.std(left_tensor) ** 2
        elif value == "center":
            center_x_coords = []
            for bbox in generated_bboxes:
                x, w = bbox[0], bbox[2]
                center_x = x + w / 2.0
                target_center = container_w / 2.0
                center_x_coords.append(center_x - target_center)
            center_x_tensor = torch.stack(center_x_coords)
            L_alignment = torch.mean(center_x_tensor ** 2)
        elif value == "right":
            right_coords = []
            for bbox in generated_bboxes:
                x, w = bbox[0], bbox[2]
                right_x = x + w
                target_right = container_w
                right_coords.append(right_x - target_right)
            right_tensor = torch.stack(right_coords)
            L_alignment = torch.mean(right_tensor ** 2)
    
    elif direction == "vertical":
        if value == "top":
            top_coords = []
            for bbox in generated_bboxes:
                y = bbox[1]
                top_coords.append(y)
            top_tensor = torch.stack(top_coords)
            L_alignment = torch.std(top_tensor) ** 2
        elif value == "center":
            center_y_coords = []
            for bbox in generated_bboxes:
                y, h = bbox[1], bbox[3]
                center_y = y + h / 2.0
                target_center = container_h / 2.0
                center_y_coords.append(center_y - target_center)
            center_y_tensor = torch.stack(center_y_coords)
            L_alignment = torch.mean(center_y_tensor ** 2)
        elif value == "bottom":
            bottom_coords = []
            for bbox in generated_bboxes:
                y, h = bbox[1], bbox[3]
                bottom_y = y + h
                target_bottom = container_h
                bottom_coords.append(bottom_y - target_bottom)
            bottom_tensor = torch.stack(bottom_coords)
            L_alignment = torch.mean(bottom_tensor ** 2)
    
    return L_alignment


# -----------------------------
# Readability Loss (Global Size Consistency)
# -----------------------------
def compute_readability_loss(
    size_rules: List[Tuple[int, int]],
    generated_bboxes: List[torch.Tensor],
    size_ratio_threshold: float = 1.5,
    device: str = "cpu"
) -> torch.Tensor:
    """Compute readability loss based on element size hierarchy rules.
    
    Args:
        size_rules: List of tuples (source_idx, target_idx) indicating that 
                   element at source_idx should be larger than element at target_idx
        generated_bboxes: List of bbox tensors (x, y, w, h) for generated elements
        size_ratio_threshold: Minimum ratio threshold (default from params.SIZE_RATIO_THRESHOLD)
        device: Device for tensor operations
    
    Returns:
        Sum of size difference penalties for violated rules
    """
    if not size_rules or len(size_rules) == 0:
        return torch.tensor(0.0, device=device)
    
    # Calculate size (area) for each element: Size = width × height
    sizes = []
    for bbox in generated_bboxes:
        w, h = bbox[2], bbox[3]  # width and height
        size = w * h
        sizes.append(size)
    
    # For each rule (source_idx, target_idx): source should be >= target * size_ratio_threshold
    # Penalty = max(0, target * size_ratio_threshold - source)
    total_loss = torch.tensor(0.0, device=device)
    for source_idx, target_idx in size_rules:
        if source_idx < len(sizes) and target_idx < len(sizes):
            size_source = sizes[source_idx]
            size_target = sizes[target_idx]
            # Required minimum size for source: target * size_ratio_threshold
            min_size_source = size_target * size_ratio_threshold
            # Penalty if source is smaller than required
            penalty = torch.clamp(min_size_source - size_source, min=0.0)
            total_loss = total_loss + penalty
    
    return total_loss


# -----------------------------
# Proximity Ratio Loss
# -----------------------------
def compute_proximity_ratio_loss(
    container_bboxes: List[Tuple[float, float, float, float]],
    child_bboxes_list: List[List],  # Can be List[Tuple] or List[Tensor]
    grandchild_bboxes_list: List[List[List[Tuple[float, float, float, float]]]],
    container_types: List[str],
    container_weights: Optional[List[float]] = None,
    epsilon: float = 1e-6,
    device: str = "cpu"
) -> torch.Tensor:
    """Compute proximity ratio loss for layout hierarchy clarity.
    
    Simplified for bottom-up layout: only considers current layer's children spacing
    and children's children spacing.
    
    Proximity Ratio measures whether inner-group distance < outer-group distance.
    Score_P = Gap_external / (Gap_internal + epsilon)
    Higher score (>1.0, ideally >1.5) indicates clear grouping.
    
    Args:
        container_bboxes: List of container bboxes (x, y, w, h)
        child_bboxes_list: For each container, list of child bboxes (x, y, w, h)
        grandchild_bboxes_list: For each container, list of lists of grandchildren bboxes
                                (children of each child)
        container_types: List of container types ("column", "row", or "layer")
        container_weights: Optional weights for weighted average (default: equal weights)
        epsilon: Small value to avoid division by zero
        device: Device for tensor operations
    
    Returns:
        Proximity loss: negative of weighted average score (to minimize)
    """
    if len(container_bboxes) == 0:
        return torch.tensor(0.0, device=device)
    
    scores = []
    weights = []

    for i, container_bbox in enumerate(container_bboxes):
        container_type = container_types[i] if i < len(container_types) else "layer"
        
        child_bboxes = child_bboxes_list[i] if i < len(child_bboxes_list) else []
        grandchild_bboxes_list_for_container = grandchild_bboxes_list[i] if i < len(grandchild_bboxes_list) else []
        
        # Ensure grandchild_bboxes_list_for_container is a list of lists (one list per child)
        if not isinstance(grandchild_bboxes_list_for_container, list):
            grandchild_bboxes_list_for_container = []
        elif len(grandchild_bboxes_list_for_container) > 0:
            # Check if it's a flat list (first element is a bbox tuple) or nested list
            first_elem = grandchild_bboxes_list_for_container[0]
            if isinstance(first_elem, (tuple, list)) and len(first_elem) == 4 and isinstance(first_elem[0], (int, float)):
                # Flat list: wrap it as a single child's grandchildren list
                grandchild_bboxes_list_for_container = [grandchild_bboxes_list_for_container]
        
        if len(child_bboxes) < 2:
            # Need at least 2 children to calculate internal gap
            continue
        
        # Step A: Determine direction based on container type
        # column: vertical arrangement, row: horizontal arrangement
        is_horizontal = (container_type == "row")
        
        # Step B: Calculate Gap_internal
        # For each child, calculate gap between its children (grandchildren), then take the maximum
        gap_internal_max = 0.0
        if len(grandchild_bboxes_list_for_container) > 0:
            for child_idx, grandchild_list in enumerate(grandchild_bboxes_list_for_container):
                if grandchild_list is None or not isinstance(grandchild_list, list):
                    continue
                
                # Collect valid grandchildren for this child
                grandchildren = []
                for gc in grandchild_list:
                    if gc is None:
                        continue
                    if isinstance(gc, (tuple, list)) and len(gc) >= 4:
                        grandchildren.append(tuple(gc[:4]))
                    elif isinstance(gc, dict):
                        grandchildren.append((
                            gc.get("x", 0), gc.get("y", 0),
                            gc.get("width", gc.get("w", 0)),
                            gc.get("height", gc.get("h", 0))
                        ))
                
                if len(grandchildren) < 2:
                    continue
                
                # Sort grandchildren by the same direction as container
                if is_horizontal:
                    sorted_gc = sorted(grandchildren, key=lambda b: b[0])
                else:
                    sorted_gc = sorted(grandchildren, key=lambda b: b[1])
                
                # Calculate gaps between adjacent grandchildren within this child
                for j in range(len(sorted_gc) - 1):
                    bbox1 = sorted_gc[j]
                    bbox2 = sorted_gc[j + 1]
                    
                    if is_horizontal:
                        gap = bbox2[0] - (bbox1[0] + bbox1[2])
                    else:
                        gap = bbox2[1] - (bbox1[1] + bbox1[3])
                    
                    gap_internal_max = max(gap_internal_max, max(0.0, gap))
        
        if gap_internal_max == 0.0:
            # No valid internal gaps found, skip this container
            continue
        
        gap_internal = gap_internal_max
        
        # Step C: Calculate Gap_external (spacing between children)
        # Convert child_bboxes to tensors if needed for gradient computation
        child_bboxes_tensors = []
        for bbox in child_bboxes:
            if isinstance(bbox, torch.Tensor):
                child_bboxes_tensors.append(bbox)
            elif isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
                # If bbox is tuple/list, create tensor but note: this breaks gradient chain
                # Ideally, bboxes should be passed as tensors from the caller
                child_bboxes_tensors.append(torch.tensor([bbox[0], bbox[1], bbox[2], bbox[3]], device=device, dtype=torch.float32))
            else:
                continue
        
        if len(child_bboxes_tensors) < 2:
            # Not enough children: skip or use fallback
            _, _, w_container, h_container = container_bbox
            gap_external_tensor = torch.tensor(max(w_container, h_container) * 0.1, device=device, dtype=torch.float32)
        else:
            # Sort children by x (horizontal) or y (vertical)
            if is_horizontal:
                x_coords = torch.stack([bbox[0] for bbox in child_bboxes_tensors])
                sorted_indices = torch.argsort(x_coords)
            else:
                y_coords = torch.stack([bbox[1] for bbox in child_bboxes_tensors])
                sorted_indices = torch.argsort(y_coords)
            
            sorted_children_tensors = [child_bboxes_tensors[i] for i in sorted_indices]
            
            gap_external_list = []
            for j in range(len(sorted_children_tensors) - 1):
                bbox1 = sorted_children_tensors[j]
                bbox2 = sorted_children_tensors[j + 1]
                
                if is_horizontal:
                    # Horizontal: gap = x2 - (x1 + w1)
                    gap = bbox2[0] - (bbox1[0] + bbox1[2])
                else:
                    # Vertical: gap = y2 - (y1 + h1)
                    gap = bbox2[1] - (bbox1[1] + bbox1[3])
                
                gap_external_list.append(gap)
            
            if len(gap_external_list) > 0:
                gap_external_tensor = torch.stack(gap_external_list).mean()
            else:
                _, _, w_container, h_container = container_bbox
                gap_external_tensor = torch.tensor(max(w_container, h_container) * 0.1, device=device, dtype=torch.float32)
        
        gap_internal_tensor = torch.tensor(gap_internal, device=device, dtype=torch.float32)
        
        # Step D: Calculate Score_P
        # Use gap_external directly (can be negative), but scale it for score calculation
        # When gap_external is negative, we still want gradient, so use a smooth function
        # Score = gap_external / (internal_gap + epsilon)
        # This allows negative scores, which will be penalized in the loss
        score_p = gap_external_tensor / (gap_internal_tensor + epsilon)
        
        scores.append(score_p)
        
        # Weight by container area (larger containers have more influence)
        _, _, w_container, h_container = container_bbox
        weight = w_container * h_container
        weights.append(weight)
    
    if len(scores) == 0:
        return torch.tensor(0.0, device=device)
    
    # Stack scores to preserve gradients
    scores_tensor = torch.stack(scores)
    
    # Normalize weights
    if container_weights is not None and len(container_weights) == len(weights):
        weights = container_weights
    
    weights_tensor = torch.tensor(weights, device=device)
    if weights_tensor.sum() > 0:
        weights_tensor = weights_tensor / weights_tensor.sum()
    else:
        weights_tensor = torch.ones_like(weights_tensor) / len(weights_tensor)
    
    # Weighted average score
    weighted_score = (scores_tensor * weights_tensor).sum()
    
    # One-sided penalty: only penalize when Score_P < 1.5 (grouping not clear enough).
    # Score_P >= 1.5 means external gap is sufficiently larger than internal gap → loss = 0.
    target_score = 1.5
    loss = torch.clamp(target_score - weighted_score, min=0.0) ** 2
    
    return loss


# -----------------------------
# Overlap Loss (mask-based)
# -----------------------------
def _get_pair_overlap_type(overlap_constraints, i: int, j: int) -> Optional[str]:
    if not overlap_constraints:
        return None
    if isinstance(overlap_constraints, dict):
        return overlap_constraints.get("overlap_type") or overlap_constraints.get("type")
    if isinstance(overlap_constraints, list):
        for c in overlap_constraints:
            si, ti = c.get("source_index"), c.get("target_index")
            if (si, ti) == (i, j) or (si, ti) == (j, i):
                return c.get("type") or c.get("overlap_type")
        return None
    return None


def _is_non_overlap_constraint(overlap_type: Optional[str]) -> bool:
    if overlap_type is None:
        return True
    non_overlap_types = ("non_overlap", "disjoint", "non-overlap")
    return overlap_type.lower() in non_overlap_types if isinstance(overlap_type, str) else False


def compute_fully_inside_loss(
    softmasks: List[torch.Tensor],
    chart_no_grid_softmasks: dict,
    chart_not_overlap_softmasks: dict,
    fully_overlap_pairs: List[Tuple[int, int]],
    Wc: int,
    Hc: int,
    device: str = "cpu",
) -> torch.Tensor:
    if not fully_overlap_pairs or not softmasks:
        return torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    for non_chart_idx, chart_idx in fully_overlap_pairs:
        if non_chart_idx >= len(softmasks) or chart_idx >= len(softmasks):
            continue
        no_grid = chart_no_grid_softmasks.get(chart_idx)
        not_overlap = chart_not_overlap_softmasks.get(chart_idx)
        if no_grid is None or not_overlap is None:
            continue
        nc = softmasks[non_chart_idx]
        area_nc = nc.sum()
        area_inside = (nc * no_grid).sum()
        area_avoid = (nc * not_overlap).sum()
        L_inside = area_nc - area_inside
        L_avoid = area_avoid
        total = total + L_inside + L_avoid
    return total


def compute_overlap_loss_mask(
    masks: List[np.ndarray],
    bboxes: List[Tuple[float, float, float, float]],
    Wc: int,
    Hc: int,
    overlap_constraints=None,
) -> float:
    if not masks or not bboxes or len(masks) != len(bboxes) or len(masks) < 2:
        return 0.0

    from scipy.ndimage import zoom

    num = len(masks)

    # ------------------------------------------------------------------
    # Optimization 1: Pre-zoom each mask to its bbox size exactly once.
    #
    # Original code zoomed mask_i inside the (i, j) pair loop, causing
    # O(N²) zoom calls (each element appeared in N-1 pairs).
    # By caching the result here we reduce zoom calls to O(N).
    # Masks are stored as bool arrays to make the later bitwise AND cheap.
    # ------------------------------------------------------------------
    int_bboxes: List[Tuple[int, int, int, int]] = []
    zoomed_bool_masks: List[np.ndarray] = []

    for k in range(num):
        x_k, y_k, w_k, h_k = bboxes[k]
        x_k = int(float(x_k))
        y_k = int(float(y_k))
        w_k = max(1, int(float(w_k)))
        h_k = max(1, int(float(h_k)))
        int_bboxes.append((x_k, y_k, w_k, h_k))

        mask_k = masks[k]
        mask_01 = (
            (mask_k > 0.5).astype(np.float32)
            if mask_k.dtype != np.float32
            else np.clip(mask_k, 0, 1)
        )
        if mask_01.shape != (h_k, w_k):
            zoom_y = h_k / mask_01.shape[0]
            zoom_x = w_k / mask_01.shape[1]
            mask_01 = zoom(mask_01, (zoom_y, zoom_x), order=1)

        # Convert to bool once; avoids repeated >0.5 comparisons in pair loop
        zoomed_bool_masks.append(mask_01 > 0.5)

    # ------------------------------------------------------------------
    # Pair loop: evaluate overlap only for non-overlap constraint pairs
    # ------------------------------------------------------------------
    total_loss = 0.0

    for i in range(num):
        for j in range(i + 1, num):
            pair_type = _get_pair_overlap_type(overlap_constraints, i, j)
            if not _is_non_overlap_constraint(pair_type):
                continue

            x_i, y_i, w_i, h_i = int_bboxes[i]
            x_j, y_j, w_j, h_j = int_bboxes[j]

            # Canvas-clipped extents of each element's placed region
            y_start_i, y_end_i = max(0, y_i), min(y_i + h_i, Hc)
            x_start_i, x_end_i = max(0, x_i), min(x_i + w_i, Wc)
            y_start_j, y_end_j = max(0, y_j), min(y_j + h_j, Hc)
            x_start_j, x_end_j = max(0, x_j), min(x_j + w_j, Wc)

            # ----------------------------------------------------------
            # Optimization 2: AABB intersection pre-check (O(1)).
            #
            # If the canvas-clipped bounding boxes don't overlap at all,
            # the pixel-level overlap must be zero — skip every mask op.
            # For well-separated layouts this eliminates most pair work.
            # ----------------------------------------------------------
            inter_y0 = max(y_start_i, y_start_j)
            inter_y1 = min(y_end_i, y_end_j)
            inter_x0 = max(x_start_i, x_start_j)
            inter_x1 = min(x_end_i, x_end_j)

            if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
                continue  # bboxes don't intersect on canvas → overlap = 0

            # ----------------------------------------------------------
            # Optimization 3: Crop to intersection region directly.
            #
            # Original code allocated two full Hc×Wc canvas arrays (~8 MB
            # combined) just to extract the overlapping region.  Instead,
            # we map the intersection canvas coords back to each element's
            # zoomed mask and slice the tiny overlap patch directly.
            #
            # Coordinate mapping:
            #   canvas column cx ∈ [x_start_k, x_end_k)
            #   → mask column  cx - x_start_k
            # (same for rows)  This holds for both positive and negative
            # origin coordinates because x_start_k = max(0, x_k).
            # ----------------------------------------------------------
            crop_i = zoomed_bool_masks[i][
                inter_y0 - y_start_i : inter_y1 - y_start_i,
                inter_x0 - x_start_i : inter_x1 - x_start_i,
            ]
            crop_j = zoomed_bool_masks[j][
                inter_y0 - y_start_j : inter_y1 - y_start_j,
                inter_x0 - x_start_j : inter_x1 - x_start_j,
            ]

            overlap_area = float(np.sum(crop_i & crop_j))
            total_loss += overlap_area

    return total_loss


# -----------------------------
# Visual Balance Loss
# -----------------------------
def _mass_centroid_from_mask_local(mask: np.ndarray, x: float, y: float, w: float, h: float) -> Tuple[float, float, float]:
    from scipy.ndimage import zoom
    h_int, w_int = max(1, int(h)), max(1, int(w))
    mask_01 = (mask > 0.5).astype(np.float32) if mask.dtype != np.float32 else np.clip(mask, 0, 1)
    if mask_01.shape != (h_int, w_int):
        zoom_y, zoom_x = h_int / mask_01.shape[0], w_int / mask_01.shape[1]
        mask_01 = zoom(mask_01, (zoom_y, zoom_x), order=1)
    mask_01 = np.clip(mask_01, 0, 1)
    mass = float(np.sum(mask_01))
    if mass < 1e-8:
        return 0.0, x + w / 2.0, y + h / 2.0
    jj = np.arange(w_int, dtype=np.float32)
    ii = np.arange(h_int, dtype=np.float32)
    mean_j = float(np.sum(mask_01 * jj[np.newaxis, :]) / mass)
    mean_i = float(np.sum(mask_01 * ii[:, np.newaxis]) / mass)
    centroid_x = x + mean_j
    centroid_y = y + mean_i
    return mass, centroid_x, centroid_y


def _mass_centroid_from_mask_container(mask_tensor: torch.Tensor, Wc: int, Hc: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = mask_tensor.squeeze()
    Hs, Ws = m.shape[0], m.shape[1]
    mass = m.sum()
    if mass < 1e-8:
        return mass, torch.tensor(Wc / 2.0, device=mask_tensor.device, dtype=mask_tensor.dtype), torch.tensor(Hc / 2.0, device=mask_tensor.device, dtype=mask_tensor.dtype)
    px_cont = (torch.arange(Ws, device=m.device, dtype=m.dtype) + 0.5) * Wc / float(Ws)
    py_cont = (torch.arange(Hs, device=m.device, dtype=m.dtype) + 0.5) * Hc / float(Hs)
    centroid_x = (m * px_cont.unsqueeze(0)).sum() / mass
    centroid_y = (m * py_cont.unsqueeze(1)).sum() / mass
    return mass, centroid_x, centroid_y


def compute_visual_balance_loss(
    bboxes: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    W_container: int,
    H_container: int,
    masks: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
    device: str = "cpu"
) -> torch.Tensor:
    if len(bboxes) == 0:
        return torch.tensor(0.0, device=device)
    
    total_mass = torch.tensor(0.0, device=device)
    weighted_x = torch.tensor(0.0, device=device)
    weighted_y = torch.tensor(0.0, device=device)
    
    use_masks = masks is not None and len(masks) == len(bboxes)
    
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        if use_masks:
            m = masks[i]
            if isinstance(m, torch.Tensor):
                mass, cx, cy = _mass_centroid_from_mask_container(m, W_container, H_container)
            else:
                x_f = x.item() if hasattr(x, 'item') else float(x)
                y_f = y.item() if hasattr(y, 'item') else float(y)
                w_f = w.item() if hasattr(w, 'item') else float(w)
                h_f = h.item() if hasattr(h, 'item') else float(h)
                mass, cx_f, cy_f = _mass_centroid_from_mask_local(m, x_f, y_f, w_f, h_f)
                mass = torch.tensor(mass, device=device, dtype=torch.float32)
                cx = torch.tensor(cx_f, device=device, dtype=torch.float32)
                cy = torch.tensor(cy_f, device=device, dtype=torch.float32)
        else:
            mass = w * h
            cx = x + w / 2.0
            cy = y + h / 2.0
        total_mass = total_mass + mass
        weighted_x = weighted_x + mass * cx
        weighted_y = weighted_y + mass * cy
    
    if total_mass < 1e-8:
        return torch.tensor(0.0, device=device)
    
    overall_centroid_x = weighted_x / total_mass
    overall_centroid_y = weighted_y / total_mass
    center_x = torch.tensor(W_container / 2.0, device=device, dtype=torch.float32)
    center_y = torch.tensor(H_container / 2.0, device=device, dtype=torch.float32)
    loss = (overall_centroid_x - center_x) ** 2 + (overall_centroid_y - center_y) ** 2
    return loss


# -----------------------------
# Position/Size Similarity Loss
# -----------------------------
def compute_position_size_similarity_loss(
    reference_bboxes: List[Tuple[float, float, float, float]],
    generated_bboxes: List[torch.Tensor],  # List of (x, y, w, h) tensors
    reference_parent_bbox: Tuple[float, float, float, float],
    generated_parent_bbox: Tuple[float, float, float, float],
    device: str = "cpu"
) -> torch.Tensor:
    """Compute position/size similarity loss between reference and generated layouts.
    
    Args:
        reference_bboxes: List of reference element bboxes (x, y, w, h) from Example layout
        generated_bboxes: List of generated element bboxes as torch tensors (x, y, w, h)
        reference_parent_bbox: Parent container bbox (x, y, w, h) for reference layout
        generated_parent_bbox: Parent container bbox (x, y, w, h) for generated layout
        device: Device for tensor operations
    
    Returns:
        Total similarity loss (L2 distance sum over all elements)
    """
    if len(reference_bboxes) != len(generated_bboxes):
        raise ValueError(f"Mismatch in number of elements: {len(reference_bboxes)} vs {len(generated_bboxes)}")
    
    # Extract parent container coordinates
    x_PE, y_PE, w_PE, h_PE = reference_parent_bbox
    x_PG, y_PG, w_PG, h_PG = generated_parent_bbox
    
    # Convert reference parent to tensors
    x_PE_t = torch.tensor(x_PE, device=device, dtype=torch.float32)
    y_PE_t = torch.tensor(y_PE, device=device, dtype=torch.float32)
    w_PE_t = torch.tensor(w_PE, device=device, dtype=torch.float32)
    h_PE_t = torch.tensor(h_PE, device=device, dtype=torch.float32)
    
    # Convert generated parent to tensors
    x_PG_t = torch.tensor(x_PG, device=device, dtype=torch.float32)
    y_PG_t = torch.tensor(y_PG, device=device, dtype=torch.float32)
    w_PG_t = torch.tensor(w_PG, device=device, dtype=torch.float32)
    h_PG_t = torch.tensor(h_PG, device=device, dtype=torch.float32)
    
    total_loss = torch.tensor(0.0, device=device)
    
    for ref_bbox, gen_bbox in zip(reference_bboxes, generated_bboxes):
        # Extract element coordinates
        x_E, y_E, w_E, h_E = ref_bbox
        
        # Convert reference element to tensors
        x_E_t = torch.tensor(x_E, device=device, dtype=torch.float32)
        y_E_t = torch.tensor(y_E, device=device, dtype=torch.float32)
        w_E_t = torch.tensor(w_E, device=device, dtype=torch.float32)
        h_E_t = torch.tensor(h_E, device=device, dtype=torch.float32)
        
        # Generated bbox is already a tensor (x, y, w, h)
        x_G_t, y_G_t, w_G_t, h_G_t = gen_bbox
        
        # Compute relative attribute vector V_E for reference element
        V_E = torch.stack([
            (x_E_t - x_PE_t) / w_PE_t,  # relative x position
            (y_E_t - y_PE_t) / h_PE_t,  # relative y position
            w_E_t / w_PE_t,              # relative width
            h_E_t / h_PE_t               # relative height
        ])
        
        # Compute relative attribute vector V_G for generated element
        V_G = torch.stack([
            (x_G_t - x_PG_t) / w_PG_t,  # relative x position
            (y_G_t - y_PG_t) / h_PG_t,  # relative y position
            w_G_t / w_PG_t,              # relative width
            h_G_t / h_PG_t               # relative height
        ])
        
        # Compute L2 distance: D = ||V_E - V_G||_2
        diff = V_E - V_G
        distance = torch.norm(diff, p=2)
        
        total_loss = total_loss + distance
    
    return total_loss


def compute_contrast_loss(
    ref_bboxes: List[Tuple[float, float, float, float]],
    gen_bboxes: List[Tuple[float, float, float, float]],
    priorities: Optional[List[float]] = None,
    alpha: float = 1.67,
) -> float:
    """Contrast loss enforcing visual hierarchy (paper Eq. 1).

    L_contrast = Σ_{(i,j): p_i>p_j} max(0, α·s_j − s_i)

    When *priorities* is None, priority is inferred from reference element
    areas (larger reference area => higher priority).  *alpha* defaults to
    1.67 following the Modular Scale convention.
    """
    n = len(gen_bboxes)
    if n < 2 or len(ref_bboxes) != n:
        return 0.0

    if priorities is None:
        priorities = [float(b[2]) * float(b[3]) for b in ref_bboxes]

    gen_sizes = [float(b[2]) * float(b[3]) for b in gen_bboxes]

    total = 0.0
    for i in range(n):
        for j in range(n):
            if priorities[i] > priorities[j]:
                penalty = max(0.0, alpha * gen_sizes[j] - gen_sizes[i])
                total += penalty
    return total


def compute_repetition_loss(
    gen_bboxes: List[Tuple[float, float, float, float]],
    element_types: List[str],
    epsilon: float = 1e-5,
) -> float:
    """Repetition loss enforcing size consistency within same-type groups (paper Eq. 2).

    L_repetition = Σ_k σ_k / (μ_k + ε)

    G_k groups elements sharing the same type under a common parent.
    """
    n = len(gen_bboxes)
    if n < 2 or len(element_types) != n:
        return 0.0

    groups: dict = {}
    for i, etype in enumerate(element_types):
        groups.setdefault(etype, []).append(i)

    total = 0.0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        sizes = np.array(
            [float(gen_bboxes[idx][2]) * float(gen_bboxes[idx][3]) for idx in indices]
        )
        mu = float(np.mean(sizes))
        sigma = float(np.std(sizes))
        total += sigma / (mu + epsilon)
    return total


def compute_data_ink_loss_mask(
    masks: List[np.ndarray],
    bboxes: List[Tuple[float, float, float, float]],
    Wc: int,
    Hc: int,
) -> float:
    """Compute data ink loss from masks placed at bbox positions. Loss = -union_area (minimize = maximize area)."""
    if not masks or not bboxes or len(masks) != len(bboxes):
        return 0.0
    from scipy.ndimage import zoom
    composite = np.zeros((Hc, Wc), dtype=np.float32)
    for mask, bbox in zip(masks, bboxes):
        x, y, w, h = bbox
        x_int, y_int = int(float(x)), int(float(y))
        w_int, h_int = max(1, int(float(w))), max(1, int(float(h)))
        if mask.size == 0:
            continue
        mask_01 = (mask > 0.5).astype(np.float32) if mask.dtype != np.float32 else np.clip(mask, 0, 1)
        if mask_01.shape[0] != h_int or mask_01.shape[1] != w_int:
            zoom_y = h_int / mask_01.shape[0]
            zoom_x = w_int / mask_01.shape[1]
            mask_resized = zoom(mask_01, (zoom_y, zoom_x), order=1)
        else:
            mask_resized = mask_01
        mask_resized = np.clip(mask_resized, 0, 1)
        y_end = min(y_int + h_int, Hc)
        x_end = min(x_int + w_int, Wc)
        y_start = max(0, y_int)
        x_start = max(0, x_int)
        if y_end > y_start and x_end > x_start:
            crop = mask_resized[:y_end - y_start, :x_end - x_start]
            composite[y_start:y_end, x_start:x_end] = np.maximum(
                composite[y_start:y_end, x_start:x_end], crop
            )
    union_area = float(np.sum(composite > 0.5))
    return -union_area

