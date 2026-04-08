"""Orientation constraint processor."""

import torch
import torch.nn.functional as F
from typing import List, Tuple
from .base import ConstraintProcessor


class OrientationProcessor(ConstraintProcessor):
    """Processor for orientation constraints."""
    
    def can_handle(self, constraint_type: str) -> bool:
        return constraint_type == "orientation"
    
    def process(self, constraint: dict, bboxes: List[Tuple[float, float, float, float]], 
                device: str = "cpu") -> torch.Tensor:
        """Process orientation constraints.
        
        Args:
            constraint: Dictionary with "orientation" key containing list of constraints
            bboxes: List of (x, y, w, h) bounding boxes
            device: Device for tensors
        
        Returns:
            Loss tensor
        """
        orientation_constraints = constraint.get("orientation", [])
        if not orientation_constraints:
            return torch.tensor(0.0, device=device)
        
        L_orientation = torch.tensor(0.0, device=device)
        
        for orient_constraint in orientation_constraints:
            src_idx = orient_constraint["source_index"]
            tgt_idx = orient_constraint["target_index"]
            position = orient_constraint["position"]
            
            if src_idx >= len(bboxes) or tgt_idx >= len(bboxes):
                continue
            
            src_x, src_y, src_w, src_h = bboxes[src_idx]
            tgt_x, tgt_y, tgt_w, tgt_h = bboxes[tgt_idx]
            
            # Compute centers
            src_cx = src_x + 0.5 * src_w
            src_cy = src_y + 0.5 * src_h
            tgt_cx = tgt_x + 0.5 * tgt_w
            tgt_cy = tgt_y + 0.5 * tgt_h
            
            # Convert to tensors
            src_cx_t = torch.tensor(src_cx, device=device)
            src_cy_t = torch.tensor(src_cy, device=device)
            tgt_cx_t = torch.tensor(tgt_cx, device=device)
            tgt_cy_t = torch.tensor(tgt_cy, device=device)
            
            # Divide target bbox into 3x3 grid
            tgt_cell_w = tgt_w / 3.0
            tgt_cell_h = tgt_h / 3.0
            
            # Map position string to grid coordinates (col, row)
            position_map = {
                "Top-Left": (0, 0),
                "Top": (1, 0),
                "Top-Right": (2, 0),
                "Left": (0, 1),
                "Center": (1, 1),
                "Right": (2, 1),
                "Bottom-Left": (0, 2),
                "Bottom": (1, 2),
                "Bottom-Right": (2, 2),
                "left": (0, 1),
                "right": (2, 1),
                "top": (1, 0),
                "bottom": (1, 2),
            }
            
            if position not in position_map:
                continue
            
            col, row = position_map[position]
            
            # Compute target region center
            tgt_region_cx = tgt_x + (col + 0.5) * tgt_cell_w
            tgt_region_cy = tgt_y + (row + 0.5) * tgt_cell_h
            
            tgt_region_cx_t = torch.tensor(tgt_region_cx, device=device)
            tgt_region_cy_t = torch.tensor(tgt_region_cy, device=device)
            
            # Compute squared distance from source center to target region center
            dist_sq = (src_cx_t - tgt_region_cx_t) ** 2 + (src_cy_t - tgt_region_cy_t) ** 2
            L_orientation += dist_sq
        
        return L_orientation
    
    def get_weight_key(self) -> str:
        return "w_orientation"

