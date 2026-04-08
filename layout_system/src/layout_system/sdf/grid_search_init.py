"""Grid search initialization for SDF-based layout optimization."""

import os
import time as _time
from typing import List, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .core import make_container_grid, sdf_to_softmask
from .bbox import unconstrained_from_bbox


def _downsample_mask(mask: np.ndarray, grid_size: int) -> np.ndarray:
    """Vectorized downsample: if any pixel in block is 1, downsampled pixel is 1."""
    h, w = mask.shape
    dh = h // grid_size
    dw = w // grid_size
    cropped = mask[:dh * grid_size, :dw * grid_size]
    reshaped = cropped.reshape(dh, grid_size, dw, grid_size)
    return (reshaped.max(axis=(1, 3)) > 0.5).astype(np.uint8)


def _render_element_mask(
    sdf_tensor: torch.Tensor,
    x: float, y: float, w: float, h: float,
    H: int, W: int,
    device: str,
) -> np.ndarray:
    """Render SDF to binary mask at given bbox (full resolution)."""
    X_grid, Y_grid = make_container_grid(H, W, device=device)
    m, _ = sdf_to_softmask(
        sdf_tensor,
        torch.tensor(x, device=device),
        torch.tensor(y, device=device),
        torch.tensor(w, device=device),
        torch.tensor(h, device=device),
        X_grid, Y_grid, tau_px=0.5
    )
    mask_np = m.squeeze().cpu().numpy()
    return (mask_np > 0.5).astype(np.uint8)


def _render_stamp_lowres(
    sdf_tensor: torch.Tensor,
    w: float, h: float,
    downscale_factor: int,
    device: str,
) -> np.ndarray:
    """Render SDF mask at origin (0,0) directly at downsampled resolution.

    Instead of rendering at full resolution and downsampling, create a small grid
    whose pixel centers map to the original coordinate space, then sample the SDF.

    Returns:
        Binary stamp of shape (ceil(h/ds), ceil(w/ds)).
    """
    stamp_w = max(1, int(np.ceil(w / downscale_factor)))
    stamp_h = max(1, int(np.ceil(h / downscale_factor)))
    xs = (torch.arange(stamp_w, device=device, dtype=torch.float32) + 0.5) * downscale_factor
    ys = (torch.arange(stamp_h, device=device, dtype=torch.float32) + 0.5) * downscale_factor
    Y_s, X_s = torch.meshgrid(ys, xs, indexing="ij")
    with torch.no_grad():
        m, _ = sdf_to_softmask(
            sdf_tensor,
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(float(w), device=device),
            torch.tensor(float(h), device=device),
            X_s, Y_s, tau_px=0.5,
        )
    return (m.cpu().numpy().reshape(stamp_h, stamp_w) > 0.5).astype(np.uint8)


def _check_positions(stamp, downsampled_avoid, padding_ds, ds_h, ds_w,
                     ref_cx_ds, ref_cy_ds):
    """Slide stamp over avoid mask and find the best non-overlapping position.

    Returns (found, best_pos, best_dist) where best_pos = (x_ds, y_ds).
    Position preference: closest to (ref_cx_ds, ref_cy_ds).
    """
    sh, sw = stamp.shape
    y_start = padding_ds
    y_end = ds_h - sh - padding_ds + 1
    x_start = padding_ds
    x_end = ds_w - sw - padding_ds + 1

    if y_end <= y_start or x_end <= x_start:
        return False, None, float('inf')

    best_pos = None
    best_dist = float('inf')

    for y_ds in range(y_start, y_end):
        for x_ds in range(x_start, x_end):
            region = downsampled_avoid[y_ds:y_ds + sh, x_ds:x_ds + sw]
            if np.any(region & stamp):
                continue
            cx_ds = x_ds + sw * 0.5
            cy_ds = y_ds + sh * 0.5
            dist = (cx_ds - ref_cx_ds) ** 2 + (cy_ds - ref_cy_ds) ** 2
            if dist < best_dist:
                best_dist = dist
                best_pos = (x_ds, y_ds)

    found = best_pos is not None
    return found, best_pos, best_dist


def grid_search_initialization(
    masks: List[np.ndarray],
    sdf_tensors: List[torch.Tensor],
    ratios: List[float],
    Wc: int,
    Hc: int,
    min_sizes: List[Tuple[float, float]],
    reference_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    downscale_factor: int = 5,
    position_steps: int = 15,
    scale_steps: int = 5,
    device: str = "cpu",
    size_min: Optional[float] = None,
    debug: bool = False,
    viz_folder: Optional[str] = None,
) -> List[Tuple[float, float, float]]:
    """Grid search to find initial (tx, ty, ts) parameters that maximize ink with no overlap.

    Optimized version: binary search on scale + sliding window overlap check in
    downsampled space.  Each scale renders the SDF stamp only once, then checks
    all positions by slicing the downsampled avoid mask — no per-position SDF
    rendering.
    """
    _t_gs_start = _time.time()
    num_nodes = len(masks)

    if num_nodes == 0:
        return []

    if debug:
        print(f"[Grid Search] Starting greedy init for {num_nodes} nodes")
        print(f"[Grid Search] Container: {Wc}x{Hc}, downscale: {downscale_factor}x")

    # --- sort elements by area (largest first) ---
    element_areas = []
    for i in range(num_nodes):
        if reference_bboxes and i < len(reference_bboxes):
            ref_bbox = reference_bboxes[i]
            if isinstance(ref_bbox, (tuple, list)) and len(ref_bbox) >= 4:
                ref_w, ref_h = ref_bbox[2], ref_bbox[3]
            elif isinstance(ref_bbox, dict):
                ref_w = ref_bbox.get("width", ref_bbox.get("w", 100))
                ref_h = ref_bbox.get("height", ref_bbox.get("h", 100))
            else:
                ref_w, ref_h = 100, 100
            area = ref_w * ref_h
        else:
            min_width, min_height = min_sizes[i]
            area = min_width * min_height
        element_areas.append((area, i))

    element_areas.sort(reverse=True)
    sorted_indices = [idx for _, idx in element_areas]

    # --- place largest element to fill canvas ---
    largest_idx = sorted_indices[0]
    if reference_bboxes and largest_idx < len(reference_bboxes):
        ref_bbox = reference_bboxes[largest_idx]
        if isinstance(ref_bbox, (tuple, list)) and len(ref_bbox) >= 4:
            largest_w, largest_h = ref_bbox[2], ref_bbox[3]
        elif isinstance(ref_bbox, dict):
            largest_w = ref_bbox.get("width", ref_bbox.get("w", Wc))
            largest_h = ref_bbox.get("height", ref_bbox.get("h", Hc))
        else:
            largest_w, largest_h = float(Wc), float(Hc)
    else:
        r = ratios[largest_idx]
        min_w, min_h = min_sizes[largest_idx]
        largest_w = min(float(Wc), float(r) * float(Hc))
        largest_h = largest_w / r
        largest_w = max(min_w, largest_w)
        largest_h = max(min_h, largest_h)

    Wc, Hc = int(largest_w), int(largest_h)
    ds_h = max(1, Hc // downscale_factor)
    ds_w = max(1, Wc // downscale_factor)

    if debug:
        print(f"[Grid Search] Canvas = largest element: {Wc}x{Hc}")
        print(f"[Grid Search] Downsampled canvas: {ds_w}x{ds_h}")
        print(f"[Grid Search] Placement order: {sorted_indices}")

    placed_bboxes = {}
    placed_bboxes[largest_idx] = (0.0, 0.0, float(Wc), float(Hc))

    # Build initial downsampled avoid mask directly at low resolution
    downsampled_avoid = _render_stamp_lowres(
        sdf_tensors[largest_idx], float(Wc), float(Hc), downscale_factor, device
    )
    # Ensure shape matches (ds_h, ds_w)
    if downsampled_avoid.shape != (ds_h, ds_w):
        tmp = np.zeros((ds_h, ds_w), dtype=np.uint8)
        sh = min(downsampled_avoid.shape[0], ds_h)
        sw = min(downsampled_avoid.shape[1], ds_w)
        tmp[:sh, :sw] = downsampled_avoid[:sh, :sw]
        downsampled_avoid = tmp

    if debug:
        print(f"[Grid Search] Placed largest element {largest_idx} at (0, 0, {Wc}, {Hc}) - no search")

    # --- greedy placement for remaining elements ---
    for placement_step, elem_idx in enumerate(sorted_indices):
        if elem_idx == largest_idx:
            continue

        _t_elem = _time.time()
        r = ratios[elem_idx]
        min_width, min_height = min_sizes[elem_idx]

        if debug:
            print(f"[Grid Search] Placing element {elem_idx} (step {placement_step + 1}/{num_nodes})")

        # extract reference bbox
        ref_bbox = None
        if reference_bboxes and elem_idx < len(reference_bboxes):
            rb = reference_bboxes[elem_idx]
            if isinstance(rb, (tuple, list)) and len(rb) >= 4:
                ref_bbox = rb
            elif isinstance(rb, dict):
                ref_bbox = (
                    rb.get("x", 0), rb.get("y", 0),
                    rb.get("width", rb.get("w", 100)),
                    rb.get("height", rb.get("h", 100)),
                )

        # determine w search range
        if ref_bbox is not None:
            wmin = float(min_width)
            wmax = min(float(Wc), float(ref_bbox[2]) * 1.5)
        else:
            wmax = min(float(Wc), float(r) * float(Hc))
            wmin = max(float(min_width), float(r) * float(min_height))
        if wmin >= wmax:
            wmin = max(1.0, 0.5 * wmax)

        padding = max(1, int(min(Wc, Hc) * 0.02))
        padding_ds = max(1, padding // downscale_factor)

        # reference center in downsampled coords (tiebreaker)
        if ref_bbox is not None:
            ref_cx_ds = (ref_bbox[0] + ref_bbox[2] * 0.5) / downscale_factor
            ref_cy_ds = (ref_bbox[1] + ref_bbox[3] * 0.5) / downscale_factor
        else:
            ref_cx_ds = ds_w * 0.5
            ref_cy_ds = ds_h * 0.5

        best_config = None
        best_stamp_area = -1
        best_dist = float('inf')

        # --- binary search on width ---
        lo_w = wmin
        hi_w = wmax
        bs_iters = 0

        while hi_w - lo_w > max(1.0, (wmax - wmin) * 0.03):
            mid_w = (lo_w + hi_w) * 0.5
            mid_h = mid_w / r
            bs_iters += 1

            if mid_h < min_height or mid_h > Hc or mid_w > Wc:
                hi_w = mid_w
                continue

            stamp = _render_stamp_lowres(
                sdf_tensors[elem_idx], mid_w, mid_h, downscale_factor, device
            )

            found, pos, dist = _check_positions(
                stamp, downsampled_avoid, padding_ds, ds_h, ds_w,
                ref_cx_ds, ref_cy_ds,
            )

            if found:
                lo_w = mid_w
                stamp_area = int(np.sum(stamp))
                if stamp_area > best_stamp_area or (stamp_area == best_stamp_area and dist < best_dist):
                    best_stamp_area = stamp_area
                    best_dist = dist
                    bx = pos[0] * downscale_factor
                    by = pos[1] * downscale_factor
                    best_config = (float(bx), float(by), float(mid_w), float(mid_h))
            else:
                hi_w = mid_w

        if debug:
            print(f"[Grid Search]   binary search: {bs_iters} iters, w range [{wmin:.1f}, {wmax:.1f}]")

        if best_config is None:
            if debug:
                print(f"[Grid Search]   No valid config for element {elem_idx}, using fallback")
            best_config = (0.0, 0.0, min_width, min_width / r)

        placed_bboxes[elem_idx] = best_config

        # update downsampled avoid mask
        placed_stamp = _render_stamp_lowres(
            sdf_tensors[elem_idx], best_config[2], best_config[3],
            downscale_factor, device,
        )
        bx_ds = int(best_config[0] / downscale_factor)
        by_ds = int(best_config[1] / downscale_factor)
        psh, psw = placed_stamp.shape
        y_end_p = min(by_ds + psh, ds_h)
        x_end_p = min(bx_ds + psw, ds_w)
        downsampled_avoid[by_ds:y_end_p, bx_ds:x_end_p] = np.maximum(
            downsampled_avoid[by_ds:y_end_p, bx_ds:x_end_p],
            placed_stamp[:y_end_p - by_ds, :x_end_p - bx_ds],
        )

        if debug:
            print(f"[Grid Search]   Placed bbox: ({best_config[0]:.1f}, {best_config[1]:.1f}, "
                  f"{best_config[2]:.1f}, {best_config[3]:.1f})")
        print(f"[Grid Search timer] element {elem_idx} total: {_time.time() - _t_elem:.3f}s")

    print(f"[Grid Search timer] TOTAL grid_search_initialization: {_time.time() - _t_gs_start:.3f}s")

    # convert placed bboxes to unconstrained params
    result = []
    for i in range(num_nodes):
        x, y, w, h = placed_bboxes[i]
        tx, ty, ts = unconstrained_from_bbox(
            x, y, w, h, Wc, Hc, ratios[i],
            min_width=min_sizes[i][0], min_height=min_sizes[i][1],
            size_min=size_min,
        )
        result.append((tx, ty, ts))

    if debug and viz_folder:
        _save_grid_search_visualization(
            placed_bboxes, sdf_tensors, ratios, Wc, Hc, min_sizes,
            device, viz_folder, debug,
        )

    return result, Wc, Hc


def _save_grid_search_visualization(
    placed_bboxes: dict,
    sdf_tensors: List[torch.Tensor],
    ratios: List[float],
    Wc: int,
    Hc: int,
    min_sizes: List[Tuple[float, float]],
    device: str,
    viz_folder: str,
    debug: bool,
):
    """Save visualization of grid search result."""
    num_nodes = len(placed_bboxes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    bboxes = []
    softmasks_viz = []
    for i in range(num_nodes):
        x, y, w, h = placed_bboxes[i]
        m = _render_element_mask(sdf_tensors[i], x, y, w, h, Hc, Wc, device)
        softmasks_viz.append(m.astype(np.float32))
        bboxes.append((x, y, w, h))

    union = np.zeros((Hc, Wc), dtype=np.float32)
    for m in softmasks_viz:
        union = np.maximum(union, m)
    ink_ratio = np.sum(union > 0.5) / (Wc * Hc)

    ax1.imshow(union, cmap='gray', origin='upper', extent=[0, Wc, Hc, 0])
    ax1.set_title(f'Grid Search Result: Union (Ink Ratio: {ink_ratio:.4f})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)

    for i, (x, y, w, h) in enumerate(bboxes):
        rect = patches.Rectangle((x, y + h), w, -h, linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x + w/2, y + h/2, f'{i}', color='red', fontsize=14,
                fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    colors = plt.cm.tab10(np.linspace(0, 1, num_nodes))
    composite = np.zeros((Hc, Wc, 3))
    for i, m in enumerate(softmasks_viz):
        for c in range(3):
            composite[:, :, c] += m * colors[i][c]
    composite = np.clip(composite, 0, 1)

    ax2.imshow(composite, origin='upper', extent=[0, Wc, Hc, 0])
    ax2.set_title('Grid Search Result: Colored Masks', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)

    for i, (x, y, w, h) in enumerate(bboxes):
        rect = patches.Rectangle((x, y + h), w, -h, linewidth=2, edgecolor='white', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x + w/2, y + h/2, f'{i}', color='white', fontsize=14,
                fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    legend_elements = [patches.Patch(facecolor=colors[i], edgecolor='white', label=f'Element {i}')
                      for i in range(num_nodes)]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(viz_folder, 'grid_search_init.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    if debug:
        print(f"[Grid Search] Visualization saved to: {save_path}")
        print(f"[Grid Search] Final ink ratio: {ink_ratio:.4f}")
        for i, (x, y, w, h) in enumerate(bboxes):
            print(f"  Element {i}: ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")
