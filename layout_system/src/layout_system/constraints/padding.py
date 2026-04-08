"""Padding constraint processor."""

import torch
import torch.nn.functional as F
from typing import List, Tuple
from .base import ConstraintProcessor


class PaddingProcessor(ConstraintProcessor):
    """Processor for padding constraints."""
    
    def can_handle(self, constraint_type: str) -> bool:
        return constraint_type == "padding"
    
    def process(self, constraint: dict, bboxes: List[Tuple[float, float, float, float]], 
                device: str = "cpu") -> torch.Tensor:
        """Process padding constraints.
        
        Args:
            constraint: Dictionary with "padding" key
            bboxes: List of (x, y, w, h) bounding boxes
            device: Device for tensors
        
        Returns:
            Loss tensor
        """
        padding_constraint = constraint.get("padding", {})
        if not padding_constraint:
            return torch.tensor(0.0, device=device)
        
        L_padding = torch.tensor(0.0, device=device)
        
        horiz = padding_constraint.get("horizontal", {})
        vert = padding_constraint.get("vertical", {})
        
        pad_left = horiz.get("left", 0)
        pad_right = horiz.get("right", 0)
        pad_top = vert.get("top", 0)
        pad_bottom = vert.get("bottom", 0)
        
        # Get container size from first bbox (assuming all bboxes are within container)
        if not bboxes:
            return L_padding
        
        # Estimate container size (this should be passed separately in real implementation)
        # For now, assume container is large enough
        container_w = 1000.0  # Default, should be passed as parameter
        container_h = 1000.0  # Default, should be passed as parameter
        
        for x, y, w, h in bboxes:
            x_t = torch.tensor(x, device=device)
            y_t = torch.tensor(y, device=device)
            w_t = torch.tensor(w, device=device)
            h_t = torch.tensor(h, device=device)
            
            if pad_left > 0:
                L_padding += F.relu(torch.tensor(pad_left, device=device) - x_t) ** 2
            if pad_right > 0:
                L_padding += F.relu((x_t + w_t) - (torch.tensor(container_w, device=device) - torch.tensor(pad_right, device=device))) ** 2
            if pad_top > 0:
                L_padding += F.relu(torch.tensor(pad_top, device=device) - y_t) ** 2
            if pad_bottom > 0:
                L_padding += F.relu((y_t + h_t) - (torch.tensor(container_h, device=device) - torch.tensor(pad_bottom, device=device))) ** 2
        
        return L_padding
    
    def get_weight_key(self) -> str:
        return "w_padding"

