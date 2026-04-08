"""Placeholder generation utilities."""

import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple


def create_placeholder_rectangle(width: float, height: float, 
                                color: Tuple[int, int, int, int] = (224, 224, 224, 255)) -> Tuple[Image.Image, np.ndarray]:
    """
    Create a solid color rectangle placeholder.
    
    Args:
        width: Rectangle width
        height: Rectangle height
        color: RGBA color tuple, default light gray
    
    Returns:
        Tuple of (image, mask)
        - image: PIL Image in RGBA format
        - mask: Binary mask array (H, W) with all ones (fully filled rectangle)
    """
    w_int = int(width)
    h_int = int(height)
    
    # Create RGBA image with solid color
    image = Image.new("RGBA", (w_int, h_int), color)
    
    # Create mask (all ones for a filled rectangle)
    mask = np.ones((h_int, w_int), dtype=np.float32)
    
    return image, mask


def create_placeholder_rounded_rectangle(width: float, height: float,
                                        radius: float = None,
                                        color: Tuple[int, int, int, int] = (224, 224, 224, 255)) -> Tuple[Image.Image, np.ndarray]:
    """
    Create a rounded rectangle placeholder with rounded corners.
    
    Args:
        width: Rectangle width
        height: Rectangle height
        radius: Corner radius (default: min(width, height) * 0.1)
        color: RGBA color tuple, default light gray
    
    Returns:
        Tuple of (image, mask)
        - image: PIL Image in RGBA format
        - mask: Binary mask array (H, W) with rounded rectangle shape
    """
    w_int = int(width)
    h_int = int(height)
    
    if radius is None:
        radius = min(w_int, h_int) * 0.1
    
    # Create RGBA image
    image = Image.new("RGBA", (w_int, h_int), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Draw rounded rectangle
    draw.rounded_rectangle([(0, 0), (w_int-1, h_int-1)], radius=radius, fill=color)
    
    # Create mask from alpha channel
    mask = np.array(image.split()[3], dtype=np.float32) / 255.0
    
    return image, mask

