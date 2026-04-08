"""Base class for node handlers."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
from PIL import Image


class NodeHandler(ABC):
    """Base class for node handlers."""
    
    @abstractmethod
    def can_handle(self, node_type: str) -> bool:
        """Check if this handler can handle the given node type.
        
        Args:
            node_type: Type of node (e.g., "image", "text", "chart")
        
        Returns:
            True if this handler can handle the node type
        """
        pass
    
    @abstractmethod
    def load(self, node: dict, base_dir: Optional[str] = None, debug: bool = False) -> Tuple[np.ndarray, dict]:
        """Load node data and return mask and metadata.
        
        Args:
            node: Node dictionary from JSON
            base_dir: Base directory for resolving relative paths
        
        Returns:
            Tuple of (mask, metadata)
            - mask: Binary mask array (H, W) with values 0 or 1
            - metadata: Dictionary with additional node information
        """
        pass
    
    @abstractmethod
    def create_placeholder(self, width: float, height: float, 
                          metadata: Optional[dict] = None) -> Tuple[Image.Image, np.ndarray]:
        """Create placeholder image and mask for nodes without image_path.
        
        Args:
            width: Placeholder width
            height: Placeholder height
            metadata: Optional metadata dictionary
        
        Returns:
            Tuple of (image, mask)
            - image: PIL Image (RGBA format)
            - mask: Binary mask array (H, W)
        """
        pass

