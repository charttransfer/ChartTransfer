"""Gap constraint processor."""

import torch
import torch.nn.functional as F
from typing import List, Tuple
from .base import ConstraintProcessor


class GapProcessor(ConstraintProcessor):
    """Processor for gap constraints."""
    
    def can_handle(self, constraint_type: str) -> bool:
        return constraint_type == "gap"
    
    def process(self, constraint: dict, bboxes: List[Tuple[float, float, float, float]], 
                device: str = "cpu") -> torch.Tensor:
        """Process gap constraints.
        
        Args:
            constraint: Dictionary with "gap" key containing gap constraint
            bboxes: List of (x, y, w, h) bounding boxes
            device: Device for tensors
        
        Returns:
            Loss tensor
        """
        gap_constraint = constraint.get("gap", {})
        if not gap_constraint:
            return torch.tensor(0.0, device=device)
        
        L_gap = torch.tensor(0.0, device=device)
        
        direction = gap_constraint.get("direction", "vertical")  # "horizontal" or "vertical"
        target_gap = gap_constraint.get("value", 0.0)  # Target gap value in pixels
        
        if len(bboxes) < 2:
            return L_gap
        
        if direction == "vertical":
            # Vertical gap: distance between bottom of first element and top of second element
            # Sort by y coordinate
            sorted_bboxes = sorted(bboxes, key=lambda b: b[1])
            for i in range(len(sorted_bboxes) - 1):
                _, y1, _, h1 = sorted_bboxes[i]
                _, y2, _, _ = sorted_bboxes[i + 1]
                
                bottom1 = y1 + h1
                top2 = y2
                actual_gap = top2 - bottom1
                
                gap_diff = actual_gap - target_gap
                L_gap += gap_diff ** 2
        
        elif direction == "horizontal":
            # Horizontal gap: distance between right edge of first element and left edge of second element
            # Sort by x coordinate
            sorted_bboxes = sorted(bboxes, key=lambda b: b[0])
            for i in range(len(sorted_bboxes) - 1):
                x1, _, w1, _ = sorted_bboxes[i]
                x2, _, _, _ = sorted_bboxes[i + 1]
                
                right1 = x1 + w1
                left2 = x2
                actual_gap = left2 - right1
                
                gap_diff = actual_gap - target_gap
                L_gap += gap_diff ** 2
        
        return L_gap
    
    def get_weight_key(self) -> str:
        return "w_gap"


