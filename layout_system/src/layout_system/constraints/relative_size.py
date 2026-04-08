"""Relative size constraint processor."""

import torch
import torch.nn.functional as F
from typing import List, Tuple
from .base import ConstraintProcessor


class RelativeSizeProcessor(ConstraintProcessor):
    """Processor for relative size constraints."""
    
    def can_handle(self, constraint_type: str) -> bool:
        return constraint_type == "relative_size"
    
    def process(self, constraint: dict, bboxes: List[Tuple[float, float, float, float]], 
                device: str = "cpu") -> torch.Tensor:
        """Process relative size constraints.
        
        Args:
            constraint: Dictionary with "relative_size" key containing list of constraints
            bboxes: List of (x, y, w, h) bounding boxes
            device: Device for tensors
        
        Returns:
            Loss tensor
        """
        relative_size_constraints = constraint.get("relative_size", [])
        if not relative_size_constraints:
            return torch.tensor(0.0, device=device)
        
        L_relative_size = torch.tensor(0.0, device=device)
        
        for rel_constraint in relative_size_constraints:
            src_idx = rel_constraint["source_index"]
            tgt_idx = rel_constraint["target_index"]
            target_ratio = rel_constraint["ratio"]
            
            if src_idx >= len(bboxes) or tgt_idx >= len(bboxes):
                continue
            
            _, _, w_src, h_src = bboxes[src_idx]
            _, _, w_tgt, h_tgt = bboxes[tgt_idx]
            
            w_src_t = torch.tensor(w_src, device=device)
            h_src_t = torch.tensor(h_src, device=device)
            w_tgt_t = torch.tensor(w_tgt, device=device)
            h_tgt_t = torch.tensor(h_tgt, device=device)
            target_ratio_t = torch.tensor(target_ratio, device=device)
            
            if rel_constraint["type"] == "relative_height":
                actual_ratio = h_src_t / (h_tgt_t + 1e-8)
            elif rel_constraint["type"] == "relative_width":
                actual_ratio = w_src_t / (w_tgt_t + 1e-8)
            else:
                continue
            
            # Use relative error
            if target_ratio > 1e-6:
                relative_error = (actual_ratio - target_ratio_t) / target_ratio_t
                L_relative_size += relative_error ** 2
            else:
                L_relative_size += (actual_ratio - target_ratio_t) ** 2
        
        return L_relative_size
    
    def get_weight_key(self) -> str:
        return "w_relative_size"

