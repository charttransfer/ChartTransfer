"""BBox parameterization functions for optimization."""

import numpy as np
import torch


def bbox_aspect_from_unconstrained(tx, ty, ts, W, H, r, min_width=None, min_height=None, size_min=None):
    """Convert unconstrained parameters to bbox with fixed aspect ratio.
    
    This function maps unconstrained optimization parameters (tx, ty, ts) to
    a valid bounding box that:
    - Maintains the aspect ratio r = w/h
    - Stays within the container bounds (W, H)
    - Respects minimum size constraints
    
    Args:
        tx, ty, ts: Unconstrained parameters (will be mapped via sigmoid)
        W, H: Container width and height
        r: Aspect ratio (w/h)
        min_width: Minimum width constraint (if None, uses size_min or default)
        min_height: Minimum height constraint (if None, uses size_min or default)
        size_min: Legacy parameter for backward compatibility (maps to both min_width and min_height)
    
    Returns:
        (x, y, w, h) bounding box
    """
    # Backward compatibility: if size_min is provided, use it for both
    if size_min is not None:
        if min_width is None:
            min_width = size_min
        if min_height is None:
            min_height = size_min
    
    # Use defaults if not provided
    if min_width is None:
        min_width = 30.0  # Default from parameters
    if min_height is None:
        min_height = 30.0  # Default from parameters
    
    # w must satisfy w<=W and h=w/r<=H => w <= rH
    wmax = min(float(W), float(r) * float(H))

    # Enforce constraints:
    # w >= min_width
    # h = w/r >= min_height => w >= r * min_height
    # Combined: w >= max(min_width, r * min_height)
    wmin = max(float(min_width), float(r) * float(min_height))
    if wmin >= wmax:
        # infeasible min size; fall back
        wmin = max(1.0, 0.5 * wmax)

    w = wmin + (wmax - wmin) * torch.sigmoid(ts)
    h = w / r

    x = (W - w) * torch.sigmoid(tx)
    y = (H - h) * torch.sigmoid(ty)
    return x, y, w, h


def unconstrained_from_bbox(x, y, w, h, W, H, r, min_width=None, min_height=None, size_min=None, debug: bool = False):
    """Convert bbox to unconstrained parameters (inverse of bbox_aspect_from_unconstrained).
    
    This function is used to initialize optimization parameters from a reference bbox.
    It adjusts the bbox to match the material's aspect ratio while keeping the area
    approximately constant.
    
    Args:
        x, y, w, h: Bounding box (x, y, w, h)
        W, H: Container width and height
        r: Aspect ratio (w/h) - material's aspect ratio (used to adjust w/h)
        min_width: Minimum width constraint
        min_height: Minimum height constraint
        size_min: Legacy parameter for backward compatibility
    
    Returns:
        (tx, ty, ts) unconstrained parameters
    """
    # Backward compatibility: if size_min is provided, use it for both
    if size_min is not None:
        if min_width is None:
            min_width = size_min
        if min_height is None:
            min_height = size_min
    
    # Use defaults if not provided
    if min_width is None:
        min_width = 30.0
    if min_height is None:
        min_height = 30.0
    
    ref_cx = x + w / 2
    ref_cy = y + h / 2

    # Adjust w and h to match material's aspect ratio r
    # If JSON bbox has aspect ratio r_json = w/h, but material has r = w_material/h_material
    # We need to adjust: use material's aspect ratio, keep area approximately constant
    r_json = w / h if h > 0 else 1.0

    if abs(r_json - r) > 1e-6:
        # Aspect ratios don't match, adjust to material's aspect ratio
        # Strategy: keep area approximately constant, adjust to match r
        area = w * h
        if r > 0:
            # w_new * h_new = area, w_new / h_new = r => h_new^2 * r = area => h_new = sqrt(area/r)
            h_adjusted = np.sqrt(area / r)
            w_adjusted = h_adjusted * r
        else:
            h_adjusted = h
            w_adjusted = w
        
        # Ensure adjusted dimensions are within bounds
        w_adjusted = max(min_width, min(w_adjusted, W))
        h_adjusted = w_adjusted / r if r > 0 else h_adjusted
        h_adjusted = max(min_height, min(h_adjusted, H))
        if r > 0:
            w_adjusted = h_adjusted * r
        
        w = float(w_adjusted)
        h = float(h_adjusted)
        if debug:
            print(f"  Adjusted bbox from ({w/h if h>0 else 0:.4f}) to aspect ratio {r:.4f}: w={w:.1f}, h={h:.1f}")

    x = np.clip(ref_cx - w / 2, 0.0, max(0.0, float(W) - w))
    y = np.clip(ref_cy - h / 2, 0.0, max(0.0, float(H) - h))

    # Compute wmin and wmax (same as in bbox_aspect_from_unconstrained)
    wmax = min(float(W), float(r) * float(H))
    wmin = max(float(min_width), float(r) * float(min_height))
    if wmin >= wmax:
        wmin = max(1.0, 0.5 * wmax)
    
    # Invert: w = wmin + (wmax - wmin) * sigmoid(ts)
    # => sigmoid(ts) = (w - wmin) / (wmax - wmin)
    # => ts = logit((w - wmin) / (wmax - wmin))
    if wmax > wmin:
        sigmoid_ts = (w - wmin) / (wmax - wmin)
        sigmoid_ts = np.clip(sigmoid_ts, 1e-6, 1.0 - 1e-6)  # Avoid log(0)
        ts = np.log(sigmoid_ts / (1.0 - sigmoid_ts))
    else:
        ts = 0.0
    
    # Invert: x = (W - w) * sigmoid(tx)
    # => sigmoid(tx) = x / (W - w)
    # => tx = logit(x / (W - w))
    if W > w:
        sigmoid_tx = x / (W - w)
        sigmoid_tx = np.clip(sigmoid_tx, 1e-6, 1.0 - 1e-6)
        tx = np.log(sigmoid_tx / (1.0 - sigmoid_tx))
    else:
        tx = 0.0
    
    # Invert: y = (H - h) * sigmoid(ty)
    # => sigmoid(ty) = y / (H - h)
    # => ty = logit(y / (H - h))
    if H > h:
        sigmoid_ty = y / (H - h)
        sigmoid_ty = np.clip(sigmoid_ty, 1e-6, 1.0 - 1e-6)
        ty = np.log(sigmoid_ty / (1.0 - sigmoid_ty))
    else:
        ty = 0.0
    
    return tx, ty, ts

