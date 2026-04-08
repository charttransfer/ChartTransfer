"""SDF core functions: IO, mask processing, and SDF computation."""

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_dilation

import torch
import torch.nn.functional as F


# -----------------------------
# IO: RGBA PNG -> binary mask from alpha
# -----------------------------
def load_binary_mask_from_rgba(png_path: str, alpha_thresh: int = 1) -> np.ndarray:
    """Load a binary mask from an RGBA PNG file using the alpha channel.
    
    Args:
        png_path: Path to the PNG file
        alpha_thresh: Threshold for alpha channel (pixels with alpha > thresh are 1)
    
    Returns:
        Binary mask as float32 array (0 or 1)
    """
    img = Image.open(png_path).convert("RGBA")
    a = np.array(img, dtype=np.uint8)[..., 3]
    mask = (a > alpha_thresh).astype(np.float32)
    return mask


def tight_bbox_ratio(mask01: np.ndarray):
    """Calculate the aspect ratio of the tight bounding box of a mask.
    
    Args:
        mask01: Binary mask (0 or 1)
    
    Returns:
        Tuple of (aspect_ratio, (x0, y0, w, h))
    """
    ys, xs = np.where(mask01 > 0.5)
    if len(xs) == 0:
        raise ValueError("Mask is empty.")
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    w0 = x1 - x0
    h0 = y1 - y0
    r = w0 / float(h0)
    return r, (x0, y0, w0, h0)


# -----------------------------
# Binary mask -> SDF (inside negative, outside positive), normalized to local coord units
# local coords are [-1,1]x[-1,1]
# -----------------------------
def binary_to_sdf_norm(mask01: np.ndarray, pad: int = 16) -> np.ndarray:
    """Convert a binary mask to a normalized signed distance field.
    
    The SDF is normalized so that distances are in local coordinate units
    where the mask spans [-1, 1] in both dimensions.
    
    Args:
        mask01: Binary mask (0 or 1)
        pad: Padding to add around the mask
    
    Returns:
        Normalized SDF (positive outside, negative inside)
    """
    mask = (mask01 > 0.5)
    H0, W0 = mask.shape

    # pad with outside False to ensure border is outside
    mask_p = np.pad(mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=False)

    dist_out = distance_transform_edt(~mask_p)   # outside: dist to nearest inside
    dist_in = distance_transform_edt(mask_p)     # inside: dist to nearest outside
    sdf_px = dist_out - dist_in                  # outside +, inside -

    # Ht, Wt = sdf_px.shape
    # scale = (min(Ht, Wt) / 2.0)                  # px per local-unit (since [-1,1] spans ~min/2)
    sdf_px = sdf_px[pad:pad+H0, pad:pad+W0]
    scale = (min(H0, W0) / 2.0)                  # px per local-unit (since [-1,1] spans ~min/2)
    sdf_norm = (sdf_px / scale).astype(np.float32)
    return sdf_norm


# -----------------------------
# Morphological dilation for mask expansion
# -----------------------------
def dilate_mask(mask01: np.ndarray, radius: float) -> np.ndarray:
    """Dilate a binary mask using morphological dilation with a circular structuring element.
    
    Args:
        mask01: Binary mask (0 or 1)
        radius: Dilation radius in pixels
    
    Returns:
        Dilated binary mask
    """
    if radius <= 0:
        return mask01
    
    # Create circular structuring element
    # For radius r, we need a (2*r+1) x (2*r+1) square, then mask it to a circle
    r_int = int(np.ceil(radius))
    size = 2 * r_int + 1
    y, x = np.ogrid[-r_int:r_int+1, -r_int:r_int+1]
    mask_circle = (x*x + y*y <= radius*radius)
    
    # Perform dilation
    dilated = binary_dilation(mask01, structure=mask_circle.astype(np.uint8))
    
    return dilated.astype(np.float32)


# -----------------------------
# Container sampling grid (pixel centers) for a chosen optimization resolution
# -----------------------------
def make_container_grid(H: int, W: int, device):
    """Create a grid of pixel center coordinates for a container.
    
    Args:
        H: Container height
        W: Container width
        device: PyTorch device
    
    Returns:
        Tuple of (X, Y) coordinate grids, each of shape [H, W]
    """
    ys = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return X, Y  # [H,W]


# -----------------------------
# Differentiable rasterization: sample SDF -> soft mask
# tau_px controls boundary softness in container pixel units
# -----------------------------
def sdf_to_softmask(
    sdf_norm_t: torch.Tensor,  # [1,1,Ht,Wt]
    x, y, w, h,                # bbox in container pixels
    X, Y,                      # grid in container pixels [Hs,Ws]
    tau_px: float = 1.5,
):
    """Convert a normalized SDF to a soft mask at given bbox location.
    
    Args:
        sdf_norm_t: Normalized SDF tensor [1, 1, Ht, Wt]
        x, y, w, h: Bounding box in container pixels
        X, Y: Grid coordinates in container pixels [Hs, Ws]
        tau_px: Softness parameter in pixels
    
    Returns:
        Tuple of (soft_mask, distance_in_pixels)
    """
    cx = x + 0.5 * w
    cy = y + 0.5 * h

    # local coords (u,v) in [-1,1] within bbox
    u = (X - cx) / (0.5 * w)
    v = (Y - cy) / (0.5 * h)
    grid = torch.stack([u, v], dim=-1).unsqueeze(0)  # [1,Hs,Ws,2]

    d_norm = F.grid_sample(
        sdf_norm_t, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False
    )  # [1,1,Hs,Ws]

    # convert local distance -> approx container pixel distance
    d_px = d_norm * (0.5 * torch.minimum(w, h))
    m = torch.sigmoid(-d_px / tau_px)

    # Hard clip outside bbox
    inside_bbox = (u.abs() <= 1.0) & (v.abs() <= 1.0)
    m = m * inside_bbox.float().unsqueeze(0).unsqueeze(0)
    
    return m, d_px


def area_sum(mask01: torch.Tensor, W_container: int, H_container: int):
    """Calculate the area of a mask scaled to container coordinates.
    
    Args:
        mask01: Mask tensor [1, 1, Hs, Ws]
        W_container: Container width
        H_container: Container height
    
    Returns:
        Scaled area sum
    """
    _, _, Hs, Ws = mask01.shape
    da = (W_container / float(Ws)) * (H_container / float(Hs))
    return mask01.sum() * da

