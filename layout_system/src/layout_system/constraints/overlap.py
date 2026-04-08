"""Overlap constraint processor."""

import torch
from typing import List, Tuple
from .base import ConstraintProcessor


class OverlapProcessor(ConstraintProcessor):
    """Processor for overlap constraints.
    
    Note: Overlap is typically handled as a hard constraint in the optimization,
    but this processor can provide additional soft penalties if needed.
    """
    
    def can_handle(self, constraint_type: str) -> bool:
        return constraint_type == "overlap"
    
    def process(self, constraint: dict, bboxes: List[Tuple[float, float, float, float]], 
                device: str = "cpu") -> torch.Tensor:
        """Process overlap constraints.
        
        Args:
            constraint: Dictionary with "overlap" key
            bboxes: List of (x, y, w, h) bounding boxes
            device: Device for tensors
        
        Returns:
            Loss tensor (typically 0, as overlap is handled as hard constraint)
        """
        # Overlap is handled as a hard constraint in the main optimization loop
        # This processor can be extended to add soft penalties if needed
        return torch.tensor(0.0, device=device)
    
    def get_weight_key(self) -> str:
        return "w_overlap"

