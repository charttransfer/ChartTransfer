"""Text node handler."""

import os
import numpy as np
from PIL import Image
from typing import Tuple, Optional
from .base import NodeHandler
from layout_system.utils.placeholder import create_placeholder_rectangle


def load_binary_mask_from_rgba(png_path: str, alpha_thresh: int = 1) -> np.ndarray:
    """Load binary mask from RGBA PNG."""
    img = Image.open(png_path).convert("RGBA")
    a = np.array(img, dtype=np.uint8)[..., 3]
    mask = (a > alpha_thresh).astype(np.float32)
    return mask


class TextNodeHandler(NodeHandler):
    """Handler for text nodes."""
    
    def can_handle(self, node_type: str) -> bool:
        return node_type == "text"
    
    def load(self, node: dict, base_dir: Optional[str] = None, debug: bool = False) -> Tuple[np.ndarray, dict]:
        """Load text node data.
        
        Args:
            node: Node dictionary with bbox and optionally image_path
            base_dir: Base directory for resolving image paths
        
        Returns:
            Tuple of (mask, metadata)
        """
        # Check if text node has image_path (e.g., rendered text image)
        image_path = node.get("image_path")
        if image_path:
            # Resolve full path
            if base_dir:
                full_path = os.path.join(base_dir, image_path)
            else:
                full_path = image_path
            
            # Load actual mask from image if file exists
            if os.path.exists(full_path):
                mask = load_binary_mask_from_rgba(full_path)
                metadata = {
                    "image_path": image_path,
                    "placeholder": False,
                    "content": node.get("content", ""),
                    "type": "text",
                }
                return mask, metadata
        
        # Fallback: use bbox to create placeholder
        bbox = node.get("bbox", {})
        width = bbox.get("width", 100)
        height = bbox.get("height", 100)
        
        # Create placeholder rectangle
        _, mask = create_placeholder_rectangle(width, height)
        
        metadata = {
            "placeholder": True,
            "width": width,
            "height": height,
            "content": node.get("content", ""),
            "type": "text",
        }
        
        return mask, metadata
    
    def create_placeholder(self, width: float, height: float, 
                          metadata: Optional[dict] = None) -> Tuple[Image.Image, np.ndarray]:
        """Create placeholder for text node."""
        return create_placeholder_rectangle(width, height)

