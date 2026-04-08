"""
Element loader.

Loads elements from RGBA PNG images, determines element dimensions from
image size, and generates masks from the alpha channel.
"""

import os
import numpy as np
from PIL import Image
from typing import Optional, Tuple

from .utils.nodes import LeafNode, NodeType


def resize_image_with_aspect_ratio(img: Image.Image, 
                                   target_width: int, 
                                   target_height: int, debug: bool = False) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Resize an image while preserving aspect ratio, aligning to the shorter side,
    without adding transparent padding.
    
    Example: original 1024x1024, target 500x1000, result 500x500 (preserves 1:1 ratio, aligns to shorter side).
    
    Args:
        img: PIL Image object (RGBA format)
        target_width: Target width
        target_height: Target height
    
    Returns:
        (resized_image, actual_size): The resized image and its actual dimensions
    """
    original_width, original_height = img.size
    original_aspect = original_width / original_height
    if debug:
        print("original_aspect: ", original_aspect)
        print("target_width: ", target_width, "target_height: ", target_height)
    # Find the shorter side of the target dimensions and use it as the baseline
    if target_width <= target_height:
        # Target width is the shorter side; scale based on width
        new_width = target_width
        new_height = int(original_height * (target_width / original_width))
    else:
        # Target height is the shorter side; scale based on height
        new_height = target_height
        new_width = int(original_width * (target_height / original_height))
    
    # Ensure dimensions do not exceed target size (double check)
    if new_width > target_width:
        new_width = target_width
        new_height = int(original_height * (target_width / original_width))
    if new_height > target_height:
        new_height = target_height
        new_width = int(original_width * (target_height / original_height))
    
    if debug:
        print("new_width: ", new_width, "new_height: ", new_height)
    # Resize image (preserving aspect ratio, no transparent padding)
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_img, (new_width, new_height)


class ElementLoader:
    """Element loader."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize element loader.
        
        Args:
            base_dir: Base directory for image files; relative paths will be resolved from this directory
        """
        self.base_dir = base_dir
    
    def load_from_image(self, image_path: str, node_id: str, 
                       node_type: NodeType, 
                       metadata: Optional[dict] = None,
                       target_width: Optional[float] = None,
                       target_height: Optional[float] = None) -> LeafNode:
        """
        Load an element from a PNG image.
        
        Args:
            image_path: Image file path (absolute or relative to base_dir)
            node_id: Node ID
            node_type: Node type
            metadata: Additional metadata
            target_width: Target width (optional; resizes the image if provided)
            target_height: Target height (optional; resizes the image if provided)
            
        Returns:
            LeafNode object containing image dimensions and mask
        """
        # Resolve path
        full_path = self._resolve_path(image_path)
        
        # Load image
        img = Image.open(full_path)
        
        # Ensure RGBA format
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Get original dimensions
        original_width, original_height = img.size
        
        # If target dimensions are specified, resize the image (preserving aspect ratio, aligning to shorter side)
        if target_width is not None and target_height is not None:
            target_width = int(target_width)
            target_height = int(target_height)
            # Use aspect-ratio-preserving resize (align to shorter side, no transparent padding)
            img, (width, height) = resize_image_with_aspect_ratio(
                img, target_width, target_height
            )
        else:
            width, height = original_width, original_height
        
        # Generate mask from alpha channel
        # Mask is a binary image: pixels with alpha > 0 are 1, otherwise 0
        alpha_channel = np.array(img.split()[3])  # Extract alpha channel
        mask = (alpha_channel > 0).astype(np.uint8) * 255
        
        # Create metadata
        node_metadata = metadata or {}
        node_metadata['image_path'] = image_path
        node_metadata['image_size'] = (width, height)
        node_metadata['original_image_size'] = (original_width, original_height)
        if target_width is not None and target_height is not None:
            node_metadata['resized'] = True
        
        # Create leaf node
        node = LeafNode(
            node_id=node_id,
            node_type=node_type,
            width=float(width),
            height=float(height),
            mask=mask,
            metadata=node_metadata
        )
        
        return node
    
    def _resolve_path(self, image_path: str) -> str:
        """Resolve image path."""
        if os.path.isabs(image_path):
            return image_path
        
        if self.base_dir:
            return os.path.join(self.base_dir, image_path)
        
        return image_path
    
    def load_chart(self, image_path: str, node_id: str, 
                   metadata: Optional[dict] = None) -> LeafNode:
        """Load a chart element."""
        return self.load_from_image(image_path, node_id, NodeType.CHART, metadata)
    
    def load_image(self, image_path: str, node_id: str,
                  metadata: Optional[dict] = None) -> LeafNode:
        """Load an image element."""
        return self.load_from_image(image_path, node_id, NodeType.IMAGE, metadata)
    
    def load_text(self, image_path: str, node_id: str,
                 metadata: Optional[dict] = None) -> LeafNode:
        """Load a text element (text rendered as an image)."""
        return self.load_from_image(image_path, node_id, NodeType.TEXT, metadata)
    
    def load_shape(self, image_path: str, node_id: str,
                  metadata: Optional[dict] = None) -> LeafNode:
        """Load a shape element."""
        return self.load_from_image(image_path, node_id, NodeType.SHAPE, metadata)


def load_element_from_image(image_path: str, node_id: str, 
                           node_type: NodeType,
                           base_dir: Optional[str] = None,
                           metadata: Optional[dict] = None,
                           target_width: Optional[float] = None,
                           target_height: Optional[float] = None) -> LeafNode:
    """
    Convenience function to load an element from an image.
    
    Args:
        image_path: Image file path
        node_id: Node ID
        node_type: Node type
        base_dir: Base directory (optional)
        metadata: Additional metadata (optional)
        target_width: Target width (optional; resizes the image if provided)
        target_height: Target height (optional; resizes the image if provided)
        
    Returns:
        LeafNode object
    """
    loader = ElementLoader(base_dir=base_dir)
    return loader.load_from_image(image_path, node_id, node_type, metadata, 
                                 target_width, target_height)

