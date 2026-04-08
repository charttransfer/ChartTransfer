"""Composite utilities for combining node results."""

import numpy as np
from typing import List, Tuple
from PIL import Image


def composite_nodes(nodes: List[dict], bboxes: List[Tuple[float, float, float, float]],
                    container_bbox: Tuple[float, float, float, float], debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Composite multiple nodes into a single mask.
    
    Args:
        nodes: List of node dictionaries with "mask" key
        bboxes: List of (x, y, w, h) bounding boxes for each node
        container_bbox: Container bounding box (x, y, w, h)
    
    Returns:
        Tuple of (composite_mask, composite_sdf)
        - composite_mask: Combined mask (H, W)
        - composite_sdf: Combined SDF (H, W) - simplified version
    """
    # Handle container_bbox format
    if isinstance(container_bbox, (tuple, list)) and len(container_bbox) == 4:
        cx, cy, cw, ch = container_bbox
    elif isinstance(container_bbox, dict):
        cx = container_bbox.get("x", 0)
        cy = container_bbox.get("y", 0)
        cw = container_bbox.get("width", container_bbox.get("w", 0))
        ch = container_bbox.get("height", container_bbox.get("h", 0))
    else:
        raise ValueError(f"Invalid container_bbox format: {container_bbox}")
    
    cw_int = int(float(cw))
    ch_int = int(float(ch))
    
    # Initialize composite mask
    composite_mask = np.zeros((ch_int, cw_int), dtype=np.float32)
    
    # Place each node's mask at its bbox position
    for i, (node, bbox) in enumerate(zip(nodes, bboxes)):
        mask = node.get("mask")
        if mask is None:
            continue
        
        # Handle different bbox formats
        if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            x, y, w, h = bbox
        elif isinstance(bbox, dict):
            x = bbox.get("x", 0)
            y = bbox.get("y", 0)
            w = bbox.get("width", bbox.get("w", 0))
            h = bbox.get("height", bbox.get("h", 0))
        else:
            if debug:
                print(f"Warning: Invalid bbox format at index {i}: {bbox} (type: {type(bbox)})")
            continue
        
        x_int = int(float(x))
        y_int = int(float(y))
        w_int = int(float(w))
        h_int = int(float(h))
        
        # Resize mask to bbox size if needed
        if mask.shape != (h_int, w_int):
            from scipy.ndimage import zoom
            zoom_y = h_int / mask.shape[0]
            zoom_x = w_int / mask.shape[1]
            mask_resized = zoom(mask, (zoom_y, zoom_x), order=1)
        else:
            mask_resized = mask
        
        # Clip to valid range
        mask_resized = np.clip(mask_resized, 0, 1)
        
        # Place mask in composite
        y_end = min(y_int + h_int, ch_int)
        x_end = min(x_int + w_int, cw_int)
        y_start = max(0, y_int)
        x_start = max(0, x_int)
        
        if y_end > y_start and x_end > x_start:
            mask_crop = mask_resized[:y_end-y_start, :x_end-x_start]
            composite_mask[y_start:y_end, x_start:x_end] = np.maximum(
                composite_mask[y_start:y_end, x_start:x_end],
                mask_crop
            )
    
    # Generate simplified SDF from composite mask
    # This is a simplified version - full SDF generation would use distance_transform_edt
    composite_sdf = composite_mask - 0.5  # Simple approximation
    
    return composite_mask, composite_sdf

