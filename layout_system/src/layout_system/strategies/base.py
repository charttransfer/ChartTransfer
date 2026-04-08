"""Base class for optimization strategies."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


class OptimizationStrategy(ABC):
    """Base class for optimization strategies."""
    
    @abstractmethod
    def optimize(self, nodes: List[dict], container_bbox: Tuple[float, float, float, float],
                 constraints: dict, config: dict) -> List[Tuple[float, float, float, float]]:
        """Execute optimization and return bounding boxes for each node.
        
        Args:
            nodes: List of node dictionaries with masks and metadata
            container_bbox: Container bounding box (x, y, w, h)
            constraints: Constraints dictionary from JSON
            config: Optimization configuration
        
        Returns:
            List of optimized bounding boxes (x, y, w, h) for each node
        """
        pass
    
    @abstractmethod
    def composite(self, nodes: List[dict], bboxes: List[Tuple[float, float, float, float]],
                  container_bbox: Tuple[float, float, float, float], debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Composite optimized results and return mask and SDF.
        
        Args:
            nodes: List of node dictionaries with masks and metadata
            bboxes: List of optimized bounding boxes (x, y, w, h)
            container_bbox: Container bounding box (x, y, w, h)
        
        Returns:
            Tuple of (mask, sdf)
            - mask: Composite mask array (H, W)
            - sdf: Signed distance field array (H, W)
        """
        pass

