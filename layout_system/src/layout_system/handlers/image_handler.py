"""Image node handler."""

import os
import numpy as np
from PIL import Image
from typing import Tuple, Optional
from .base import NodeHandler
from layout_system.utils.placeholder import create_placeholder_rectangle


def load_binary_mask_from_rgba(png_path: str, alpha_thresh: int = 1, debug: bool = False) -> np.ndarray:
    actual_path = get_no_grid_version(png_path, debug=debug)
    
    img = Image.open(actual_path).convert("RGBA")
    a = np.array(img, dtype=np.uint8)[..., 3]
    mask = (a > alpha_thresh).astype(np.float32)
    return mask


def get_no_grid_version(image_path: str, debug: bool = False) -> str:
    """
    Get the no_grid version path of an image; returns the original path if it doesn't exist.
    
    When computing masks for chart images, the version with grid lines removed is preferred
    to more accurately determine the chart's actual content area.
    
    Args:
        image_path: Original image path (e.g., variation_xxx.png)
    
    Returns:
        Path to the no_grid version (if it exists) or the original path
    """
    if not image_path or not os.path.exists(image_path):
        return image_path
    
    # Construct the no_grid version path
    base_name, ext = os.path.splitext(image_path)
    no_grid_path = f"{base_name}_no_grid{ext}"
    
    # Return the no_grid version if it exists; otherwise return the original path
    if os.path.exists(no_grid_path):
        if debug:
            print(f"[ImageHandler] Using no_grid version for mask computation: {os.path.basename(no_grid_path)}")
        return no_grid_path
    else:
        return image_path


class ImageNodeHandler(NodeHandler):
    """Handler for image and chart nodes."""
    
    def can_handle(self, node_type: str) -> bool:
        return node_type in ["image", "chart"]
    
    def load(self, node: dict, base_dir: Optional[str] = None, debug: bool = False) -> Tuple[np.ndarray, dict]:
        """Load image node data.
        
        Args:
            node: Node dictionary with "image_path" key
            base_dir: Base directory for resolving relative paths
        
        Returns:
            Tuple of (mask, metadata)
        """
        image_path = node.get("image_path")
        if not image_path:
            # No image_path, create placeholder from bbox
            bbox = node.get("bbox", {})
            width = bbox.get("width", 100)
            height = bbox.get("height", 100)
            _, mask = create_placeholder_rectangle(width, height)
            return mask, {"placeholder": True, "width": width, "height": height}
        
        # Resolve path
        if base_dir and not os.path.isabs(image_path):
            full_path = os.path.join(base_dir, image_path)
        else:
            full_path = image_path
        
        mask = load_binary_mask_from_rgba(full_path, debug=debug)
        
        metadata = {
            "image_path": image_path,
            "full_path": full_path,
            "shape": mask.shape,
        }
        
        return mask, metadata
    
    def create_placeholder(self, width: float, height: float, 
                          metadata: Optional[dict] = None) -> Tuple[Image.Image, np.ndarray]:
        """Create placeholder for image node."""
        return create_placeholder_rectangle(width, height)

