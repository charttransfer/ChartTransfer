"""Alignment constraint processor."""

import torch
import torch.nn.functional as F
from typing import List, Tuple
from .base import ConstraintProcessor


class AlignmentProcessor(ConstraintProcessor):
    """Processor for alignment constraints."""
    
    def can_handle(self, constraint_type: str) -> bool:
        return constraint_type == "alignment"
    
    def process(self, constraint: dict, bboxes: List[Tuple[float, float, float, float]], 
                device: str = "cpu") -> torch.Tensor:
        """Process alignment constraints.
        
        Args:
            constraint: Dictionary with "alignment" key containing alignment constraint
            bboxes: List of (x, y, w, h) bounding boxes
            device: Device for tensors
        
        Returns:
            Loss tensor
        """
        alignment_constraint = constraint.get("alignment", {})
        if not alignment_constraint:
            return torch.tensor(0.0, device=device)
        
        L_alignment = torch.tensor(0.0, device=device)
        
        direction = alignment_constraint.get("direction", "horizontal")  # "horizontal" or "vertical"
        value = alignment_constraint.get("value", "center")  # "left", "center", "right", "top", "bottom"
        
        if not bboxes:
            return L_alignment
        
        # Get container size (should be passed separately, but for now estimate from bboxes)
        # Calculate bounding box of all elements
        all_x = [x for x, _, _, _ in bboxes]
        all_y = [y for _, y, _, _ in bboxes]
        all_w = [w for _, _, w, _ in bboxes]
        all_h = [h for _, _, _, h in bboxes]
        
        container_w = max(x + w for x, w in zip(all_x, all_w)) if all_x else 1000.0
        container_h = max(y + h for y, h in zip(all_y, all_h)) if all_y else 1000.0
        
        if direction == "horizontal":
            # Horizontal alignment: align elements along x-axis
            if value == "left":
                # All elements should align to left edge
                for x, _, _, _ in bboxes:
                    x_t = torch.tensor(x, device=device)
                    L_alignment += x_t ** 2
            elif value == "center":
                # All elements should be centered horizontally
                for x, _, w, _ in bboxes:
                    x_t = torch.tensor(x, device=device)
                    w_t = torch.tensor(w, device=device)
                    center_x = x_t + 0.5 * w_t
                    target_center = torch.tensor(container_w / 2.0, device=device)
                    L_alignment += (center_x - target_center) ** 2
            elif value == "right":
                # All elements should align to right edge
                for x, _, w, _ in bboxes:
                    x_t = torch.tensor(x, device=device)
                    w_t = torch.tensor(w, device=device)
                    right_x = x_t + w_t
                    target_right = torch.tensor(container_w, device=device)
                    L_alignment += (right_x - target_right) ** 2
        
        elif direction == "vertical":
            # Vertical alignment: align elements along y-axis
            if value == "top":
                # All elements should align to top edge
                for _, y, _, _ in bboxes:
                    y_t = torch.tensor(y, device=device)
                    L_alignment += y_t ** 2
            elif value == "center":
                # All elements should be centered vertically
                for _, y, _, h in bboxes:
                    y_t = torch.tensor(y, device=device)
                    h_t = torch.tensor(h, device=device)
                    center_y = y_t + 0.5 * h_t
                    target_center = torch.tensor(container_h / 2.0, device=device)
                    L_alignment += (center_y - target_center) ** 2
            elif value == "bottom":
                # All elements should align to bottom edge
                for _, y, _, h in bboxes:
                    y_t = torch.tensor(y, device=device)
                    h_t = torch.tensor(h, device=device)
                    bottom_y = y_t + h_t
                    target_bottom = torch.tensor(container_h, device=device)
                    L_alignment += (bottom_y - target_bottom) ** 2
        
        return L_alignment
    
    def get_weight_key(self) -> str:
        return "w_alignment"


