"""Hierarchical layout optimizer with extensible architecture."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import os
import time as _time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataclasses import dataclass, field
from . import parameters as params
from .constraints import (
    ConstraintProcessor,
    RelativeSizeProcessor,
    PaddingProcessor,
    OrientationProcessor,
    OverlapProcessor,
    AlignmentProcessor,
    GapProcessor,
)
from .handlers import (
    NodeHandler,
    ImageNodeHandler,
    TextNodeHandler,
)
from .strategies import OptimizationStrategy, SDFOptimizationStrategy, RuleBasedLayoutStrategy, GridSearchOptimizationStrategy
from .utils.parser import parse_layout_tree, LayoutNode
from .utils.composite import composite_nodes
from .utils.placeholder import create_placeholder_rectangle, create_placeholder_rounded_rectangle


@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    strategy: OptimizationStrategy = field(default_factory=SDFOptimizationStrategy)
    strategy_name: Optional[str] = None  # "sdf" (default) or "grid_search"
    constraint_processors: Dict[str, ConstraintProcessor] = field(default_factory=dict)
    node_handlers: Dict[str, NodeHandler] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    optimization_params: Dict[str, Any] = field(default_factory=dict)
    placeholder_config: Dict[str, Any] = field(default_factory=dict)
    base_dir: Optional[str] = None
    output_dir: Optional[str] = None
    device: Optional[str] = None
    debug: bool = False
    debug_hierarchical: bool = False
    debug_strategy: bool = False
    debug_save: bool = False
    use_rule_based: bool = True
    rule_based_types: List[str] = field(default_factory=lambda: ['row', 'column'])
    
    def __post_init__(self):
        """Initialize default values."""
        # Select strategy by name if provided
        if self.strategy_name == "grid_search":
            object.__setattr__(self, 'strategy', GridSearchOptimizationStrategy())
        elif self.strategy_name == "sdf":
            object.__setattr__(self, 'strategy', SDFOptimizationStrategy())

        if not self.constraint_processors:
            self.constraint_processors = {
                "relative_size": RelativeSizeProcessor(),
                "padding": PaddingProcessor(),
                "orientation": OrientationProcessor(),
                "overlap": OverlapProcessor(),
                "gap": GapProcessor(),
            }
        
        if not self.node_handlers:
            self.node_handlers = {
                "image": ImageNodeHandler(),
                "chart": ImageNodeHandler(),
                "text": TextNodeHandler(),
            }
        
        if not self.weights:
            self.weights = {
                "w_similarity": params.W_SIMILARITY,
                "w_readability": params.W_READABILITY,
                "w_alignment_consistency": params.W_ALIGNMENT_CONSISTENCY,
                "w_alignment_similarity": params.W_ALIGNMENT_SIMILARITY,
                "w_proximity": params.W_PROXIMITY,
            }
        
        _opt_defaults = {
            "opt_res_list": params.OPT_RES_LIST,
            "outer_rounds": params.OUTER_ROUNDS,
            "inner_steps": params.INNER_STEPS,
            "lr": params.LEARNING_RATE,
        }
        if not self.optimization_params:
            self.optimization_params = _opt_defaults
        else:
            # User-supplied values take priority; fill in any missing defaults
            self.optimization_params = {**_opt_defaults, **self.optimization_params}
        if self.debug and not any([self.debug_hierarchical, self.debug_strategy, self.debug_save]):
            object.__setattr__(self, 'debug_hierarchical', True)
            object.__setattr__(self, 'debug_strategy', True)
            object.__setattr__(self, 'debug_save', True)


class HierarchicalOptimizer:
    """Hierarchical layout optimizer with extensible architecture."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize optimizer.
        
        Args:
            config: Optimization configuration (uses default if None)
        """
        self.config = config or OptimizationConfig()
        self._register_default_processors()
        self._register_default_handlers()
    
    def _register_default_processors(self):
        """Register default constraint processors."""
        if not self.config.constraint_processors:
            self.config.constraint_processors = {
                "relative_size": RelativeSizeProcessor(),
                "padding": PaddingProcessor(),
                "orientation": OrientationProcessor(),
                "overlap": OverlapProcessor(),
            }
    
    def _register_default_handlers(self):
        """Register default node handlers."""
        if not self.config.node_handlers:
            self.config.node_handlers = {
                "image": ImageNodeHandler(),
                "chart": ImageNodeHandler(),
                "text": TextNodeHandler(),
            }
    
    def register_processor(self, processor: ConstraintProcessor):
        """Register a custom constraint processor.
        
        Args:
            processor: Constraint processor instance
        """
        # Register for all constraint types it can handle
        for constraint_type in ["relative_size", "padding", "orientation", "overlap", "gap", "alignment"]:
            if processor.can_handle(constraint_type):
                self.config.constraint_processors[constraint_type] = processor
    
    def register_handler(self, handler: NodeHandler):
        """Register a custom node handler.
        
        Args:
            handler: Node handler instance
        """
        # Register for all node types it can handle
        for node_type in ["image", "chart", "text", "shape", "layer", "column", "row"]:
            if handler.can_handle(node_type):
                self.config.node_handlers[node_type] = handler
    
    def set_strategy(self, strategy: OptimizationStrategy):
        """Set optimization strategy.
        
        Args:
            strategy: Optimization strategy instance
        """
        self.config.strategy = strategy
    
    def optimize_tree(self, tree_json: dict) -> Dict[str, Any]:
        """Optimize entire tree structure.
        
        Args:
            tree_json: JSON dictionary with layout tree structure
        
        Returns:
            Dictionary with optimization results for each node
        """
        t0 = _time.time()
        root_node = parse_layout_tree(tree_json)
        
        root_bbox = (
            root_node.bbox.get("x", 0),
            root_node.bbox.get("y", 0),
            root_node.bbox.get("width", 1000),
            root_node.bbox.get("height", 1000),
        )
        
        result = self._optimize_node(root_node, root_bbox, "root")
        print(f'[HierarchicalOptimizer TIMER] optimize_tree total: {_time.time()-t0:.2f}s')
        
        return result
    
    def _optimize_node(self, node: LayoutNode, parent_bbox: Tuple[float, float, float, float], 
                       node_path: str = "root") -> Dict[str, Any]:
        """Optimize a single node and its children recursively.
        
        Args:
            node: Layout node to optimize
            parent_bbox: Parent container bounding box (x, y, w, h)
            node_path: Path string for this node (e.g., "root.child0.child1")
        
        Returns:
            Dictionary with optimization results
        """
        result = {
            "type": node.type,
            "bbox": node.bbox,
            "final_bbox": None,
            "image_path": getattr(node, "image_path", None),  # Preserve image_path for saving
        }
        
        # Check if node is a container (has children)
        if node.children:
            # Check if we can use rule-based layout (much faster for row/column)
            if self._can_use_rule_based_layout(node):
                return self._rule_based_layout(node, parent_bbox, node_path)
            
            # Container node: optimize children using SDF optimization
            if self.config.debug_hierarchical:
                print(f"[HierarchicalOptimizer] Processing container node: type={node.type}, num_children={len(node.children)}, path={node_path}")
            child_results = []
            child_nodes_data = []
            
            # First, recursively optimize all children
            for i, child in enumerate(node.children):
                child_path = f"{node_path}.child{i}"
                child_result = self._optimize_node(child, parent_bbox, child_path)
                child_results.append(child_result)
            
            # Load child node data (masks, etc.)
            # Use final_bbox from child_result if available (for container nodes that have been optimized),
            # otherwise use initial bbox
            for i, child in enumerate(node.children):
                child_result = child_results[i]
                # Get bbox: use final_bbox from child_result if child is a container that has been optimized
                if child_result.get("final_bbox") is not None:
                    # Convert final_bbox tuple to dict format for consistency
                    final_bbox = child_result["final_bbox"]
                    if isinstance(final_bbox, (tuple, list)) and len(final_bbox) >= 4:
                        child_bbox = {
                            "x": final_bbox[0],
                            "y": final_bbox[1],
                            "width": final_bbox[2],
                            "height": final_bbox[3]
                        }
                    else:
                        child_bbox = child.bbox
                else:
                    child_bbox = child.bbox
                
                # Check if child has composite_mask (from previous optimization)
                # Use composite_mask if available, as it represents the actual shape
                mask = None
                metadata = {}
                if child_result.get("composite_mask") is not None:
                    mask = child_result["composite_mask"]
                    if self.config.debug_hierarchical:
                        print(f"[HierarchicalOptimizer] Using composite_mask for child {i} (type={child.type})")
                else:
                    handler = self._get_handler(child.type)
                    if handler:
                        mask, metadata = handler.load(child.__dict__, self.config.base_dir, debug=self.config.debug_strategy)
                    else:
                        # Fallback: use placeholder
                        bbox = child_bbox
                        width = bbox.get("width", 100) if isinstance(bbox, dict) else (bbox[2] if isinstance(bbox, (tuple, list)) else 100)
                        height = bbox.get("height", 100) if isinstance(bbox, dict) else (bbox[3] if isinstance(bbox, (tuple, list)) else 100)
                        # For container nodes (layer/column/row), create a rounded rectangle instead of solid rectangle
                        if child.type in ["layer", "column", "row"]:
                            try:
                                _, mask = create_placeholder_rounded_rectangle(width, height)
                            except Exception:
                                # Fallback to rectangle if rounded rectangle is unavailable
                                _, mask = create_placeholder_rectangle(width, height)
                        else:
                            _, mask = create_placeholder_rectangle(width, height)
                        metadata = {"placeholder": True}
                
                child_nodes_data.append({
                    "mask": mask,
                    "bbox": child_bbox,
                    "metadata": metadata,
                    "type": child.type,
                })
            
            # Get container bbox (use node's bbox or parent's)
            container_bbox = (
                node.bbox.get("x", 0),
                node.bbox.get("y", 0),
                node.bbox.get("width", parent_bbox[2]),
                node.bbox.get("height", parent_bbox[3]),
            )
            
            # Optimize children layout
            constraints = node.constraints or {}
            if self.config.debug_hierarchical:
                print(f"[HierarchicalOptimizer] constraints: {node.constraints}")
            
            # Add alignment from node itself if present
            if node.alignment:
                constraints['alignment'] = node.alignment
            
            # Generate unique save prefix for this node using node path
            # Replace dots and special characters to make valid filename
            save_prefix = node_path.replace(".", "_").replace(" ", "_")
            
            # Extract grandchildren info for proximity ratio calculation
            # For each child, collect its children's bboxes (grandchildren of current container)
            grandchildren_list = []
            for child_result in child_results:
                grandchildren = []
                # Check if child_result has children (from recursive optimization)
                child_children = child_result.get("children", [])
                for grandchild_result in child_children:
                    grandchild_bbox = grandchild_result.get("final_bbox")
                    if grandchild_bbox:
                        if isinstance(grandchild_bbox, (tuple, list)) and len(grandchild_bbox) >= 4:
                            grandchildren.append(tuple(grandchild_bbox[:4]))
                        elif isinstance(grandchild_bbox, dict):
                            grandchildren.append((
                                grandchild_bbox.get("x", 0),
                                grandchild_bbox.get("y", 0),
                                grandchild_bbox.get("width", grandchild_bbox.get("w", 0)),
                                grandchild_bbox.get("height", grandchild_bbox.get("h", 0))
                            ))
                grandchildren_list.append(grandchildren)
            
            config = {
                **self.config.optimization_params,
                "device": self.config.device,
                "debug": self.config.debug_strategy,
                "debug_strategy": self.config.debug_strategy,
                "w_similarity": self.config.weights.get("w_similarity", 1.0),
                "w_readability": self.config.weights.get("w_readability", 1.0),
                "w_alignment_consistency": self.config.weights.get("w_alignment_consistency", 1.0),
                "w_alignment_similarity": self.config.weights.get("w_alignment_similarity", params.W_ALIGNMENT_SIMILARITY),
                "w_proximity": self.config.weights.get("w_proximity", params.W_PROXIMITY),
                "container_type": node.type,  # Pass container type (column, row, or layer)
                "grandchildren_list": grandchildren_list,  # Pass grandchildren for proximity calculation
                "output_dir": self.config.output_dir,
            }
            
            if self.config.debug_hierarchical:
                print(f"[HierarchicalOptimizer] Calling strategy.optimize for container_type={node.type}, num_children={len(child_nodes_data)}")
                print("config:", config)
            t_strat = _time.time()
            optimized_bboxes = self.config.strategy.optimize(
                child_nodes_data,
                container_bbox,
                constraints,
                config,
                save_prefix=save_prefix,
            )
            print(f'[HierarchicalOptimizer TIMER] strategy.optimize ({node.type}, {len(child_nodes_data)} children, path={node_path}): {_time.time()-t_strat:.2f}s')
            
            if self.config.debug_hierarchical:
                print(f"[HierarchicalOptimizer] Strategy returned optimized_bboxes: {optimized_bboxes}")
            # Calculate actual container bbox based on children's layout results
            # Note: optimized_bboxes are relative to container origin (0,0)
            if optimized_bboxes:
                # Convert bbox to tuple format if needed (handle both dict and tuple formats)
                def bbox_to_tuple(bbox):
                    if isinstance(bbox, dict):
                        return (
                            bbox.get("x", 0),
                            bbox.get("y", 0),
                            bbox.get("width", bbox.get("w", 0)),
                            bbox.get("height", bbox.get("h", 0))
                        )
                    elif isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
                        return tuple(bbox[:4])
                    else:
                        return (0, 0, 0, 0)
                
                bbox_tuples = [bbox_to_tuple(bbox) for bbox in optimized_bboxes]
                
                if self.config.debug_hierarchical:
                    print("bbox_tuples:", bbox_tuples)
                # Find the bounding box that contains all children (for position adjustment)
                min_x = min(bbox[0] for bbox in bbox_tuples)
                min_y = min(bbox[1] for bbox in bbox_tuples)
                
                # Adjust children's bboxes: shift them so that min_x and min_y become 0
                # This makes children relative to the new container origin (0,0)
                # For container nodes, preserve their own calculated width/height
                adjusted_bboxes_for_composite = []
                for i, (child_result, opt_bbox) in enumerate(zip(child_results, optimized_bboxes)):
                    bbox_tuple = bbox_to_tuple(opt_bbox)
                    # Adjust coordinates: subtract min_x and min_y to start from (0,0)
                    adjusted_x = bbox_tuple[0] - min_x
                    adjusted_y = bbox_tuple[1] - min_y
                    adjusted_bbox = (adjusted_x, adjusted_y, bbox_tuple[2], bbox_tuple[3])
                    
                    if child_result.get("children") and child_result.get("final_bbox"):
                        old_bbox = child_result["final_bbox"]
                        if isinstance(old_bbox, (tuple, list)) and len(old_bbox) >= 4:
                            old_w, old_h = old_bbox[2], old_bbox[3]
                            new_w, new_h = bbox_tuple[2], bbox_tuple[3]
                            if old_w > 0 and old_h > 0:
                                scale_x = new_w / old_w
                                scale_y = new_h / old_h
                                self._scale_children_bboxes(child_result, scale_x, scale_y)
                    
                    adjusted_bboxes_for_composite.append(adjusted_bbox)
                    
                    # Store relative coordinates (relative to parent container) in final_bbox
                    # save_result.py will convert to absolute coordinates by adding parent offsets
                    child_result["final_bbox"] = adjusted_bbox
                    if i < len(node.children):
                        node.children[i].final_bbox = adjusted_bbox
                        if self.config.debug_hierarchical:
                            print(f"node.children[{i}].final_bbox:", node.children[i].final_bbox)
                
                # Calculate actual container size based on adjusted bboxes
                # This ensures the container size is based on children's preserved widths/heights
                actual_max_x = max(bbox[0] + bbox[2] for bbox in adjusted_bboxes_for_composite)
                actual_max_y = max(bbox[1] + bbox[3] for bbox in adjusted_bboxes_for_composite)
                actual_width = actual_max_x
                actual_height = actual_max_y
                if self.config.debug_hierarchical:
                    print(f"actual_container_size: width={actual_width:.2f}, height={actual_height:.2f}")
                
                # Container position: keep original container position
                # Container size: actual size needed to contain all children
                # Children are now adjusted to start from (0,0) relative to container
                actual_container_bbox = (
                    container_bbox[0],  # Keep original x position
                    container_bbox[1],  # Keep original y position
                    actual_width,       # Actual width needed
                    actual_height       # Actual height needed
                )
            else:
                # Fallback to original container bbox if no children
                actual_container_bbox = container_bbox
                adjusted_bboxes_for_composite = optimized_bboxes
                # Update child results with optimized bboxes (no adjustment needed)
                for i, (child_result, opt_bbox) in enumerate(zip(child_results, optimized_bboxes)):
                    child_result["final_bbox"] = opt_bbox
                    if i < len(node.children):
                        node.children[i].final_bbox = opt_bbox
            
            # Composite children results (use actual container bbox and adjusted bboxes)
            composite_mask, composite_sdf = self.config.strategy.composite(
                child_nodes_data,
                adjusted_bboxes_for_composite,
                actual_container_bbox,
                debug=self.config.debug_strategy,
            )
            
            result["final_bbox"] = actual_container_bbox
            result["composite_mask"] = composite_mask
            result["composite_sdf"] = composite_sdf
            result["children"] = child_results
            
        else:
            # Leaf node: just load data
            handler = self._get_handler(node.type)
            if handler:
                mask, metadata = handler.load(node.__dict__, self.config.base_dir, debug=self.config.debug_strategy)
                result["mask"] = mask
                node_meta = getattr(node, "metadata", None) or {}
                result["metadata"] = {**metadata, **node_meta}
                result["final_bbox"] = (
                    node.bbox.get("x", 0),
                    node.bbox.get("y", 0),
                    node.bbox.get("width", 100),
                    node.bbox.get("height", 100),
                )
        
        return result
    
    def _get_handler(self, node_type: str) -> Optional[NodeHandler]:
        """Get handler for node type.
        
        Args:
            node_type: Type of node
        
        Returns:
            Node handler or None
        """
        return self.config.node_handlers.get(node_type)
    
    def _can_use_rule_based_layout(self, node: LayoutNode) -> bool:
        """Check if node can use rule-based layout instead of SDF optimization.
        
        Rule-based layout is much faster and more accurate for simple row/column layouts.
        
        Args:
            node: Layout node to check
        
        Returns:
            True if rule-based layout can be used
        """
        # Check if rule-based is enabled
        if not self.config.use_rule_based:
            return False
        
        # Check if node type is in the allowed list
        if node.type not in self.config.rule_based_types:
            return False
        
        # # Need at least 2 children to benefit from rule-based layout
        # if not node.children or len(node.children) < 2:
        #     return False
        
        # Check for complex overlap constraints that require SDF
        constraints = node.constraints or {}
        if 'overlap' in constraints:
            # Overlap constraints need precise collision detection - use SDF
            return False
        
        return True
    
    def _rule_based_layout(self, node: LayoutNode, parent_bbox: Tuple[float, float, float, float],
                          node_path: str) -> Dict[str, Any]:
        """Execute rule-based layout for simple row/column arrangements.
        
        Args:
            node: Layout node to optimize
            parent_bbox: Parent container bounding box (x, y, w, h)
            node_path: Path string for this node
        
        Returns:
            Dictionary with optimization results
        """
        if self.config.debug_hierarchical:
            print(f"[HierarchicalOptimizer] Using rule-based layout for {node.type} node: {node_path}")
        
        result = {
            "type": node.type,
            "bbox": node.bbox,
            "final_bbox": None,
            "image_path": getattr(node, "image_path", None),
        }
        
        # First, recursively optimize all children
        child_results = []
        for i, child in enumerate(node.children):
            child_path = f"{node_path}.child{i}"
            child_result = self._optimize_node(child, parent_bbox, child_path)
            child_results.append(child_result)
        
        # Load child node data
        child_nodes_data = []
        for i, child in enumerate(node.children):
            child_result = child_results[i]
            
            # Get mask first
            mask = None
            metadata = {}
            if child_result.get("composite_mask") is not None:
                mask = child_result["composite_mask"]
            else:
                handler = self._get_handler(child.type)
                if handler:
                    mask, metadata = handler.load(child.__dict__, self.config.base_dir, debug=self.config.debug_strategy)
                else:
                    bbox = child.bbox
                    width = bbox.get("width", 100) if isinstance(bbox, dict) else bbox[2]
                    height = bbox.get("height", 100) if isinstance(bbox, dict) else bbox[3]
                    if child.type in ["layer", "column", "row"]:
                        _, mask = create_placeholder_rounded_rectangle(width, height)
                    else:
                        _, mask = create_placeholder_rectangle(width, height)
                    metadata = {"placeholder": True}
            
            # Get bbox - always use tight bbox from mask to ensure correct aspect ratio
            if mask is not None:
                ys, xs = np.where(mask > 0.5)
                if len(xs) > 0 and len(ys) > 0:
                    x0, x1 = xs.min(), xs.max() + 1
                    y0, y1 = ys.min(), ys.max() + 1
                    mask = mask[y0:y1, x0:x1]
                    child_bbox = {"x": 0, "y": 0, "width": x1-x0, "height": y1-y0}
                else:
                    child_bbox = {"x": 0, "y": 0, "width": mask.shape[1], "height": mask.shape[0]}
            elif child_result.get("final_bbox") is not None:
                final_bbox = child_result["final_bbox"]
                if isinstance(final_bbox, (tuple, list)) and len(final_bbox) >= 4:
                    child_bbox = {
                        "x": final_bbox[0],
                        "y": final_bbox[1],
                        "width": final_bbox[2],
                        "height": final_bbox[3]
                    }
                else:
                    child_bbox = child.bbox
            else:
                child_bbox = child.bbox
            
            child_nodes_data.append({
                "mask": mask,
                "bbox": child_bbox,
                "original_bbox": child.bbox,
                "metadata": metadata,
                "type": child.type,
            })
        
        # Get container bbox
        container_bbox = (
            node.bbox.get("x", 0),
            node.bbox.get("y", 0),
            node.bbox.get("width", parent_bbox[2]),
            node.bbox.get("height", parent_bbox[3]),
        )
        
        # Use rule-based strategy
        from .strategies import RuleBasedLayoutStrategy
        rule_strategy = RuleBasedLayoutStrategy()
        
        constraints = node.constraints or {}
        
        # Add alignment from node itself if present
        if node.alignment:
            constraints['alignment'] = node.alignment
        
        config = {
            "container_type": node.type,
            "debug_strategy": self.config.debug_strategy,
        }
        
        optimized_bboxes = rule_strategy.optimize(
            child_nodes_data,
            container_bbox,
            constraints,
            config,
        )
        
        # Debug visualization for rule-based layout
        if self.config.debug and self.config.output_dir:
            self._visualize_rule_based_layout(
                child_nodes_data, optimized_bboxes, container_bbox, node_path
            )
        
        # Adjust bboxes to start from (0, 0) relative to container
        if optimized_bboxes:
            min_x = min(bbox[0] for bbox in optimized_bboxes)
            min_y = min(bbox[1] for bbox in optimized_bboxes)
            
            adjusted_bboxes_for_composite = []
            for i, (child_result, opt_bbox) in enumerate(zip(child_results, optimized_bboxes)):
                adjusted_x = opt_bbox[0] - min_x
                adjusted_y = opt_bbox[1] - min_y
                adjusted_bbox = (adjusted_x, adjusted_y, opt_bbox[2], opt_bbox[3])
                
                # If this is a container node, scale its children to fit new size
                if child_result.get("children") and child_result.get("final_bbox"):
                    old_bbox = child_result["final_bbox"]
                    if isinstance(old_bbox, (tuple, list)) and len(old_bbox) >= 4:
                        old_w, old_h = old_bbox[2], old_bbox[3]
                        new_w, new_h = opt_bbox[2], opt_bbox[3]
                        if old_w > 0 and old_h > 0:
                            scale_x = new_w / old_w
                            scale_y = new_h / old_h
                            self._scale_children_bboxes(child_result, scale_x, scale_y)
                
                adjusted_bboxes_for_composite.append(adjusted_bbox)
                child_result["final_bbox"] = adjusted_bbox
                if i < len(node.children):
                    node.children[i].final_bbox = adjusted_bbox
            
            # Calculate actual container size
            actual_max_x = max(bbox[0] + bbox[2] for bbox in adjusted_bboxes_for_composite)
            actual_max_y = max(bbox[1] + bbox[3] for bbox in adjusted_bboxes_for_composite)
            actual_width = actual_max_x
            actual_height = actual_max_y
            
            actual_container_bbox = (
                container_bbox[0],
                container_bbox[1],
                actual_width,
                actual_height
            )
        else:
            actual_container_bbox = container_bbox
            adjusted_bboxes_for_composite = optimized_bboxes
            for i, (child_result, opt_bbox) in enumerate(zip(child_results, optimized_bboxes)):
                child_result["final_bbox"] = opt_bbox
                if i < len(node.children):
                    node.children[i].final_bbox = opt_bbox
        
        # Composite children results
        composite_mask, composite_sdf = rule_strategy.composite(
            child_nodes_data,
            adjusted_bboxes_for_composite,
            actual_container_bbox,
            debug=self.config.debug_strategy,
        )
        
        result["final_bbox"] = actual_container_bbox
        result["composite_mask"] = composite_mask
        result["composite_sdf"] = composite_sdf
        result["children"] = child_results
        
        return result
    
    def _visualize_rule_based_layout(self, child_nodes_data, optimized_bboxes, container_bbox, node_path):
        """Visualize rule-based layout for debugging."""

        
        cx, cy, cw, ch = container_bbox
        cw_int, ch_int = int(cw), int(ch)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Input masks and original bboxes
        ax1 = axes[0]
        combined_input = np.zeros((ch_int, cw_int, 3))
        colors = plt.cm.tab10(np.linspace(0, 1, len(child_nodes_data)))
        
        for i, child_data in enumerate(child_nodes_data):
            mask = child_data.get("mask")
            bbox = child_data.get("bbox", {})
            if isinstance(bbox, dict):
                bw = int(bbox.get("width", 100))
                bh = int(bbox.get("height", 100))
            else:
                bw, bh = int(bbox[2]), int(bbox[3])
            
            if mask is not None:
                ax1.text(10, 30 + i*20, f"Node {i}: mask={mask.shape}, bbox=({bw}x{bh})", fontsize=8, color=colors[i])
        
        ax1.set_xlim(0, cw_int)
        ax1.set_ylim(ch_int, 0)
        ax1.set_title(f"Input: {len(child_nodes_data)} nodes\nContainer: {cw_int}x{ch_int}")
        ax1.set_aspect('equal')
        
        # 2. Optimized layout
        ax2 = axes[1]
        combined_opt = np.zeros((ch_int, cw_int, 3))
        
        for i, (child_data, opt_bbox) in enumerate(zip(child_nodes_data, optimized_bboxes)):
            mask = child_data.get("mask")
            x, y, w, h = opt_bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                     edgecolor=colors[i], facecolor='none', linestyle='--')
            ax2.add_patch(rect)
            ax2.text(x + w/2, y + h/2, f"Node {i}\n{w}x{h}", ha='center', va='center', 
                    fontsize=8, color=colors[i])
            
            if mask is not None and w > 0 and h > 0:
                from scipy.ndimage import zoom
                if mask.shape[0] > 0 and mask.shape[1] > 0:
                    zoom_y = h / mask.shape[0]
                    zoom_x = w / mask.shape[1]
                    mask_resized = zoom(mask, (zoom_y, zoom_x), order=1)
                    y_end = min(y + h, ch_int)
                    x_end = min(x + w, cw_int)
                    y_start = max(0, y)
                    x_start = max(0, x)
                    if y_end > y_start and x_end > x_start:
                        mask_crop = mask_resized[:y_end-y_start, :x_end-x_start]
                        for c in range(3):
                            combined_opt[y_start:y_end, x_start:x_end, c] = np.maximum(
                                combined_opt[y_start:y_end, x_start:x_end, c],
                                mask_crop * colors[i][c]
                            )
        
        ax2.imshow(combined_opt)
        ax2.set_xlim(0, cw_int)
        ax2.set_ylim(ch_int, 0)
        ax2.set_title("Optimized Layout (masks + bboxes)")
        ax2.set_aspect('equal')
        
        # 3. Info panel
        ax3 = axes[2]
        ax3.axis('off')
        info_text = f"Rule-Based Layout Debug\n{'='*30}\n\n"
        info_text += f"Container: ({cx}, {cy}, {cw}, {ch})\n\n"
        info_text += "Input Bboxes:\n"
        for i, child_data in enumerate(child_nodes_data):
            bbox = child_data.get("bbox", {})
            mask = child_data.get("mask")
            if isinstance(bbox, dict):
                bw, bh = bbox.get("width", 0), bbox.get("height", 0)
            else:
                bw, bh = bbox[2], bbox[3]
            mask_shape = mask.shape if mask is not None else "None"
            info_text += f"  Node {i}: bbox=({bw:.0f}x{bh:.0f}), mask={mask_shape}\n"
        
        info_text += "\nOptimized Bboxes:\n"
        for i, opt_bbox in enumerate(optimized_bboxes):
            x, y, w, h = opt_bbox
            info_text += f"  Node {i}: ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})\n"
        
        ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        safe_path = node_path.replace(".", "_")
        save_path = os.path.join(self.config.output_dir, f"{safe_path}_rule_based_debug.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        if self.config.debug_hierarchical:
            print(f"[RuleBasedLayout] Debug visualization saved to: {save_path}")
    
    def _scale_children_bboxes(self, node_result: Dict[str, Any], scale_x: float, scale_y: float):
        """Recursively scale all children's bboxes when parent is resized."""
        children = node_result.get("children", [])
        for child in children:
            child_bbox = child.get("final_bbox")
            if child_bbox and isinstance(child_bbox, (tuple, list)) and len(child_bbox) >= 4:
                scaled_bbox = (
                    child_bbox[0] * scale_x,
                    child_bbox[1] * scale_y,
                    child_bbox[2] * scale_x,
                    child_bbox[3] * scale_y
                )
                child["final_bbox"] = scaled_bbox
            if child.get("children"):
                self._scale_children_bboxes(child, scale_x, scale_y)

