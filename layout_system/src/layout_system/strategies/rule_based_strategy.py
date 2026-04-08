"""Rule-based layout strategy for simple row/column layouts.

This strategy provides deterministic, fast layout calculation for simple
row and column arrangements without requiring gradient-based optimization.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from .base import OptimizationStrategy


class RuleBasedLayoutStrategy(OptimizationStrategy):
    """Rule-based layout strategy for row/column layouts.
    
    This strategy calculates positions deterministically based on:
    - Container type (row/column)
    - Alignment constraints
    - Gap/spacing between elements
    
    Much faster than SDF-based optimization for simple linear arrangements.
    """
    
    def optimize(self, nodes: List[dict], container_bbox: Tuple[float, float, float, float],
                 constraints: dict, config: dict, save_prefix: str = None) -> List[Tuple[float, float, float, float]]:
        """Execute rule-based layout optimization.
        
        Args:
            nodes: List of node dictionaries with "mask", "bbox" keys
            container_bbox: Container bounding box (x, y, w, h)
            constraints: Constraints dictionary (gap, alignment, etc.)
            config: Configuration dictionary with container_type
            save_prefix: Prefix for saving (unused in rule-based)
        
        Returns:
            List of optimized bounding boxes (x, y, w, h) for each node
        """
        container_type = config.get("container_type", "row")
        debug = config.get("debug_strategy", config.get("debug", False))
        if debug:
            print(f"[RuleBasedLayoutStrategy] Optimizing {len(nodes)} nodes in {container_type} layout")
        
        if len(nodes) == 0:
            return []
        
        if len(nodes) == 1:
            if debug:
                print("nodes: ", nodes)
            bbox = nodes[0].get("bbox", (0, 0, 100, 100))
            return [self._normalize_bbox(bbox)]
        
        if container_type == "row":
            return self._layout_row(nodes, container_bbox, constraints, debug)
        elif container_type == "column":
            return self._layout_column(nodes, container_bbox, constraints, debug)
        else:
            if debug:
                print(f"[RuleBasedLayoutStrategy] Warning: Unknown container type '{container_type}', using default layout")
            return self._layout_default(nodes, container_bbox)
    
    def _layout_row(self, nodes: List[dict], container_bbox: Tuple[float, float, float, float],
                    constraints: dict, debug: bool = False) -> List[Tuple[float, float, float, float]]:
        """Layout nodes horizontally (left to right).
        
        Preserves height ratios from original_bbox while using tight bbox aspect ratios.
        """
        x, y, container_w, container_h = container_bbox
        gap = self._get_gap(constraints)
        alignment = self._get_alignment(constraints, default='center')
        
        tight_bboxes = []
        original_heights = []
        
        for node in nodes:
            tight_bbox = self._normalize_bbox(node.get("bbox", (0, 0, 100, 100)))
            orig_bbox = self._normalize_bbox(node.get("original_bbox", tight_bbox))
            tight_bboxes.append(tight_bbox)
            original_heights.append(orig_bbox[3])
        
        max_orig_height = max(original_heights) if original_heights else 1.0
        height_ratios = [h / max_orig_height for h in original_heights]
        
        target_heights = [container_h * ratio for ratio in height_ratios]
        target_widths = []
        for i, tight_bbox in enumerate(tight_bboxes):
            aspect_ratio = tight_bbox[2] / tight_bbox[3] if tight_bbox[3] > 0 else 1.0
            target_widths.append(target_heights[i] * aspect_ratio)
        
        total_width = sum(target_widths) + gap * (len(nodes) - 1)
        
        scale = 1.0
        if total_width > container_w:
            scale = container_w / total_width
        
        current_x = x
        optimized_bboxes = []
        
        for i in range(len(nodes)):
            child_w = target_widths[i] * scale
            child_h = target_heights[i] * scale
            child_x = current_x
            child_y = self._calculate_cross_axis_position(
                y, container_h, child_h, alignment, is_vertical=True
            )
            optimized_bboxes.append((child_x, child_y, child_w, child_h))
            current_x += child_w + gap * scale
        
        if debug:
            print(f"[RuleBasedLayoutStrategy] Row layout: gap={gap}, alignment={alignment}, scale={scale:.3f}")
            print(f"[RuleBasedLayoutStrategy] optimized_bboxes: {optimized_bboxes}")
        return optimized_bboxes
    
    def _layout_column(self, nodes: List[dict], container_bbox: Tuple[float, float, float, float],
                       constraints: dict, debug: bool = False) -> List[Tuple[float, float, float, float]]:
        """Layout nodes vertically (top to bottom).
        
        Preserves width ratios from original_bbox while using tight bbox aspect ratios.
        """
        x, y, container_w, container_h = container_bbox
        gap = self._get_gap(constraints)
        alignment = self._get_alignment(constraints, default='center')
        
        tight_bboxes = []
        original_widths = []
        
        for node in nodes:
            tight_bbox = self._normalize_bbox(node.get("bbox", (0, 0, 100, 100)))
            orig_bbox = self._normalize_bbox(node.get("original_bbox", tight_bbox))
            tight_bboxes.append(tight_bbox)
            original_widths.append(orig_bbox[2])
        if debug:
            print("tight_bboxes: ", tight_bboxes)
        max_orig_width = max(original_widths) if original_widths else 1.0
        width_ratios = [w / max_orig_width for w in original_widths]
        
        target_widths = [container_w * ratio for ratio in width_ratios]
        target_heights = []
        for i, tight_bbox in enumerate(tight_bboxes):
            aspect_ratio = tight_bbox[2] / tight_bbox[3] if tight_bbox[3] > 0 else 1.0
            target_heights.append(target_widths[i] / aspect_ratio if aspect_ratio > 0 else target_widths[i])
        
        total_height = sum(target_heights) + gap * (len(nodes) - 1)
        
        scale = 1.0
        if total_height > container_h:
            scale = container_h / total_height
        
        current_y = y
        optimized_bboxes = []
        
        for i in range(len(nodes)):
            child_w = target_widths[i] * scale
            child_h = target_heights[i] * scale
            child_y = current_y
            child_x = self._calculate_cross_axis_position(
                x, container_w, child_w, alignment, is_vertical=False
            )
            optimized_bboxes.append((child_x, child_y, child_w, child_h))
            current_y += child_h + gap * scale
        
        if debug:
            print(f"[RuleBasedLayoutStrategy] Column layout: gap={gap}, alignment={alignment}, scale={scale:.3f}")
            print(f"[RuleBasedLayoutStrategy] optimized_bboxes: {optimized_bboxes}")
        return optimized_bboxes
    
    def _layout_default(self, nodes: List[dict], container_bbox: Tuple[float, float, float, float]
                       ) -> List[Tuple[float, float, float, float]]:
        """Default layout: center all nodes at their original positions.
        
        Fallback for unknown container types.
        """
        optimized_bboxes = []
        for node in nodes:
            child_bbox = self._normalize_bbox(node.get("bbox", (0, 0, 100, 100)))
            optimized_bboxes.append(child_bbox)
        return optimized_bboxes
    
    def _calculate_cross_axis_position(self, container_start: float, container_size: float,
                                      child_size: float, alignment: str, is_vertical: bool) -> float:
        """Calculate position on the cross axis based on alignment.
        
        Args:
            container_start: Starting position of container (x or y)
            container_size: Size of container (width or height)
            child_size: Size of child element (width or height)
            alignment: Alignment string ('left'/'top', 'center', 'right'/'bottom')
            is_vertical: True if calculating Y position, False for X position
        
        Returns:
            Position on cross axis
        """
        alignment_lower = alignment.lower()
        
        # Map alignment strings
        if is_vertical:
            # Vertical alignment (Y axis)
            if alignment_lower in ['top', 'start', 'flex-start']:
                return container_start
            elif alignment_lower in ['center', 'middle']:
                return container_start + (container_size - child_size) / 2
            elif alignment_lower in ['bottom', 'end', 'flex-end']:
                return container_start + container_size - child_size
            else:
                # Default: center
                return container_start + (container_size - child_size) / 2
        else:
            # Horizontal alignment (X axis)
            if alignment_lower in ['left', 'start', 'flex-start']:
                return container_start
            elif alignment_lower in ['center', 'middle']:
                return container_start + (container_size - child_size) / 2
            elif alignment_lower in ['right', 'end', 'flex-end']:
                return container_start + container_size - child_size
            else:
                # Default: center
                return container_start + (container_size - child_size) / 2
    
    def _get_gap(self, constraints: dict) -> float:
        """Extract gap value from constraints.
        
        Args:
            constraints: Constraints dictionary
        
        Returns:
            Gap value in pixels (default: 20)
        """
        if not constraints:
            return 20.0
        
        # Try to get gap - it might be a dict or a direct value
        gap = constraints.get('gap', constraints.get('spacing', 20.0))
        
        # If gap is a dict (e.g., {'type': 'gap', 'value': 32.0}), extract the value
        if isinstance(gap, dict):
            return float(gap.get('value', 20.0))
        
        return float(gap)
    
    def _get_alignment(self, constraints: dict, default: str = 'center') -> str:
        """Extract alignment from constraints.
        
        Args:
            constraints: Constraints dictionary
            default: Default alignment if not specified
        
        Returns:
            Alignment string
        """
        if not constraints:
            return default
        
        # Try to get alignment - it might be a dict or a direct value
        alignment = constraints.get('alignment', default)
        
        # If alignment is a dict (e.g., {'type': 'alignment', 'value': 'center'}), extract the value
        if isinstance(alignment, dict):
            return str(alignment.get('value', alignment.get('direction', default)))
        
        return str(alignment)
    
    def _normalize_bbox(self, bbox) -> Tuple[float, float, float, float]:
        """Normalize bbox to (x, y, w, h) tuple format.
        
        Args:
            bbox: Bbox in tuple or dict format
        
        Returns:
            Normalized bbox tuple (x, y, w, h)
        """
        if isinstance(bbox, dict):
            return (
                float(bbox.get('x', 0)),
                float(bbox.get('y', 0)),
                float(bbox.get('width', bbox.get('w', 100))),
                float(bbox.get('height', bbox.get('h', 100)))
            )
        elif isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
            return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        else:
            return (0.0, 0.0, 100.0, 100.0)
    
    def composite(self, nodes: List[dict], bboxes: List[Tuple[float, float, float, float]],
                  container_bbox: Tuple[float, float, float, float], debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Composite nodes using existing utility function.
        
        This delegates to the common composite function since rule-based layout
        doesn't need special compositing logic.
        
        Args:
            nodes: List of node dictionaries
            bboxes: List of optimized bounding boxes
            container_bbox: Container bounding box
        
        Returns:
            Tuple of (composite_mask, composite_sdf)
        """
        from ..utils.composite import composite_nodes
        return composite_nodes(nodes, bboxes, container_bbox, debug=debug)

