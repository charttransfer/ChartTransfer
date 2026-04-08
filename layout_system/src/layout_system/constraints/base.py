"""Base class for constraint processors."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import torch


class ConstraintProcessor(ABC):
    """Base class for constraint processors."""
    
    @abstractmethod
    def can_handle(self, constraint_type: str) -> bool:
        """Check if this processor can handle the given constraint type.
        
        Args:
            constraint_type: Type of constraint (e.g., "relative_size", "padding")
        
        Returns:
            True if this processor can handle the constraint type
        """
        pass
    
    @abstractmethod
    def process(self, constraint: dict, bboxes: List[Tuple[float, float, float, float]], 
                device: str = "cpu") -> torch.Tensor:
        """Process constraint and return loss tensor.
        
        Args:
            constraint: Constraint dictionary from JSON
            bboxes: List of (x, y, w, h) bounding boxes for nodes
            device: Device to create tensors on
        
        Returns:
            Loss tensor (scalar)
        """
        pass
    
    @abstractmethod
    def get_weight_key(self) -> str:
        """Get the weight parameter key for this constraint.
        
        Returns:
            Weight key name (e.g., "w_relative_size")
        """
        pass

