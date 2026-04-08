"""SDF-based optimization strategy."""

import numpy as np
import tempfile
import os
from PIL import Image
from typing import List, Tuple, Dict, Any, Set
from .base import OptimizationStrategy
from layout_system import parameters as params
from layout_system.sdf import optimize as sdf_optimize
from layout_system.utils.composite import composite_nodes


def _get_not_overlap_path(image_path: str, debug: bool = False) -> str:
    """Get the _not_overlap.png version of an image path if it exists.
    
    Similar to get_no_grid_version in image_handler.py, but for not_overlap version.
    Handles both absolute and relative paths by checking relative to project root.
    
    Args:
        image_path: Original image path (can be relative or absolute)
    
    Returns:
        Path to _not_overlap.png version if exists, otherwise original path
    """
    if not image_path:
        return image_path
    
    # If path is relative, try to find it relative to project root
    # layout_system/src -> project root is two levels up
    if not os.path.isabs(image_path) and not os.path.exists(image_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
        abs_image_path = os.path.join(project_root, image_path)
        if debug:
            print("abs_image_path:", abs_image_path)
        if os.path.exists(abs_image_path):
            image_path = abs_image_path
    if debug:
        print("image_path:", image_path)
    base_name, ext = os.path.splitext(image_path)
    not_overlap_path = f"{base_name}_not_overlap{ext}"
    if debug:
        print("not_overlap_path:", not_overlap_path)
    
    if os.path.exists(not_overlap_path):
        if debug:
            print(f"[SDFOptimizationStrategy] Using not_overlap version: {os.path.basename(not_overlap_path)}")
        return not_overlap_path
    else:
        return image_path


def _get_no_grid_path(image_path: str, debug: bool = False) -> str:
    if not image_path:
        return image_path
    if not os.path.isabs(image_path) and not os.path.exists(image_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
        abs_image_path = os.path.join(project_root, image_path)
        if debug:
            print("abs_image_path:", abs_image_path)
        if os.path.exists(abs_image_path):
            image_path = abs_image_path
    if debug:
        print("image_path:", image_path)
    base_name, ext = os.path.splitext(image_path)
    no_grid_path = f"{base_name}_no_grid{ext}"
    if debug:
        print("no_grid_path:", no_grid_path)
    if os.path.exists(no_grid_path):
        if debug:
            print(f"[SDFOptimizationStrategy] Using no_grid version: {os.path.basename(no_grid_path)}")
        return no_grid_path
    else:
        return image_path


def _get_fully_overlap_pairs(constraints: dict, nodes: List[dict]) -> List[Tuple[int, int]]:
    overlap_constraints = constraints.get("overlap", [])
    if not overlap_constraints or not nodes:
        return []
    pairs = []
    for c in overlap_constraints:
        if c.get("type") != "fully_overlap":
            continue
        si, ti = c.get("source_index"), c.get("target_index")
        if si is None or ti is None or si >= len(nodes) or ti >= len(nodes):
            continue
        src_type = c.get("source_type") or nodes[si].get("type", "")
        tgt_type = c.get("target_type") or nodes[ti].get("type", "")
        if src_type == "chart":
            pairs.append((ti, si))
        elif tgt_type == "chart":
            pairs.append((si, ti))
    return pairs


def _build_chart_dual_masks_for_fully_overlap(constraints: dict, nodes: List[dict], debug: bool = False) -> Dict[int, Dict[str, str]]:
    overlap_constraints = constraints.get("overlap", [])
    if not overlap_constraints:
        return {}
    chart_dual_masks = {}
    for c in overlap_constraints:
        if c.get("type") != "fully_overlap":
            continue
        si, ti = c.get("source_index"), c.get("target_index")
        if si is None or ti is None or si >= len(nodes) or ti >= len(nodes):
            continue
        src_type = c.get("source_type") or nodes[si].get("type", "")
        tgt_type = c.get("target_type") or nodes[ti].get("type", "")
        if src_type == "chart":
            chart_idx = si
        elif tgt_type == "chart":
            chart_idx = ti
        else:
            continue
        if nodes[chart_idx].get("type") != "chart":
            continue
        image_path = nodes[chart_idx].get("metadata", {}).get("image_path") or nodes[chart_idx].get("image_path")
        if not image_path:
            continue
        if not os.path.isabs(image_path) and not os.path.exists(image_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
            abs_path = os.path.join(project_root, image_path)
            if os.path.exists(abs_path):
                image_path = abs_path
        if not os.path.exists(image_path):
            continue
        no_grid_path = _get_no_grid_path(image_path, debug=debug)
        not_overlap_path = _get_not_overlap_path(image_path, debug=debug)
        if not os.path.exists(no_grid_path) or not os.path.exists(not_overlap_path):
            continue
        chart_dual_masks[chart_idx] = {"no_grid_path": no_grid_path, "not_overlap_path": not_overlap_path}
    return chart_dual_masks


def _find_nodes_with_overlap(constraints: dict, nodes: List[dict]) -> Set[int]:
    """Find node indices that have partially_overlap or fully_overlap constraints with other nodes.
    
    Only chart nodes involved in overlap constraints need to use _not_overlap version.
    
    Args:
        constraints: Constraints dictionary containing overlap constraints
        nodes: List of node dictionaries
    
    Returns:
        Set of node indices that should use _not_overlap version
    """
    overlap_node_indices = set()
    
    overlap_constraints = constraints.get("overlap", [])
    if not overlap_constraints:
        return overlap_node_indices
    
    for constraint in overlap_constraints:
        overlap_type = constraint.get("type", "")
        if overlap_type in ("partially_overlap", "fully_overlap"):
            source_idx = constraint.get("source_index")
            target_idx = constraint.get("target_index")
            if source_idx is not None and source_idx < len(nodes) and nodes[source_idx].get("type") == "chart":
                overlap_node_indices.add(source_idx)
            if target_idx is not None and target_idx < len(nodes) and nodes[target_idx].get("type") == "chart":
                overlap_node_indices.add(target_idx)
    
    return overlap_node_indices


class SDFOptimizationStrategy(OptimizationStrategy):
    """SDF-based optimization strategy using existing optimize function."""
    
    def optimize(self, nodes: List[dict], container_bbox: Tuple[float, float, float, float],
                 constraints: dict, config: dict, save_prefix: str = None) -> List[Tuple[float, float, float, float]]:
        """Execute SDF-based optimization.
        
        Args:
            nodes: List of node dictionaries with "mask", "image_path", and "bbox" keys
            container_bbox: Container bounding box (x, y, w, h)
            constraints: Constraints dictionary from JSON (deprecated, kept for compatibility)
            config: Optimization configuration
        
        Returns:
            List of optimized bounding boxes (x, y, w, h) for each node
        """
        container_type = config.get("container_type", "unknown")
        debug = config.get("debug_strategy", config.get("debug", False))
        if debug:
            print(f"[SDFOptimizationStrategy] optimize called: container_type={container_type}, num_nodes={len(nodes)}")
            print(f"[SDFOptimizationStrategy] constraints: {constraints}")
        if len(nodes) < 2:
            # For single node, return its initial bbox
            if nodes:
                bbox = nodes[0].get("bbox", (0, 0, 100, 100))
                return [bbox]
            return []
        overlap_constraints = constraints.get("overlap", [])
        chart_dual_masks = _build_chart_dual_masks_for_fully_overlap(constraints, nodes, debug=debug)
        fully_overlap_pairs = _get_fully_overlap_pairs(constraints, nodes)
        overlap_node_indices = _find_nodes_with_overlap(constraints, nodes)
        if overlap_node_indices and debug:
            print(f"[SDFOptimizationStrategy] Nodes with overlap constraints (will use _not_overlap version): {overlap_node_indices}")
        
        # Extract image paths and masks, create temp files if needed
        image_paths = []
        original_paths = []  # Store original paths for saving composite image
        temp_files = []  # Track temp files for cleanup
        
        for i, node in enumerate(nodes):
            metadata = node.get("metadata", {})
            original_image_path = metadata.get("image_path") or node.get("image_path")
            mask = node.get("mask")
            node_type = node.get("type", "")
            
            # Store original path for composite image saving
            original_paths.append(original_image_path)
            
            if debug:
                print("node_type:", node_type)
                print("image_path:", original_image_path)
                print("overlap_node_indices:", overlap_node_indices)
            image_path = original_image_path
            if node_type == "chart" and image_path:
                if i in overlap_node_indices:
                    image_path = _get_not_overlap_path(image_path, debug=debug)
                else:
                    image_path = _get_no_grid_path(image_path, debug=debug)
            
            if image_path and os.path.exists(image_path):
                image_paths.append(image_path)
            elif mask is not None:
                # Create temporary image from mask for optimization
                # But we'll use original path for saving composite image
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_image = Image.fromarray(mask_uint8, mode='L')
                # Convert to RGBA with white background
                rgba_image = Image.new('RGBA', mask_image.size, (255, 255, 255, 255))
                rgba_image.putalpha(mask_image)
                
                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                rgba_image.save(temp_file.name)
                temp_files.append(temp_file.name)
                image_paths.append(temp_file.name)
            else:
                # Fallback: use placeholder
                image_paths.append(None)
        
        if debug:
            print(f"[SDFOptimizationStrategy] Processing {len(nodes)}-node case for container_type={container_type}")
        
        # Check if all image paths are valid
        if not all(image_paths):
            # Cleanup temp files
            for tf in temp_files:
                if os.path.exists(tf):
                    os.unlink(tf)
            # Fallback: return initial bboxes if no image paths
            return [node.get("bbox", (0, 0, 100, 100)) for node in nodes]
        
        # Get container dimensions
        _, _, Wc, Hc = container_bbox
        Wc = int(Wc)
        Hc = int(Hc)
        
        # Extract optimization parameters from config
        opt_res_list = config.get("opt_res_list", (256, 512, 1000))
        outer_rounds = config.get("outer_rounds", 12)
        inner_steps = config.get("inner_steps", 300)
        lr = config.get("lr", 0.05)
        w_similarity = config.get("w_similarity", 1.0)
        w_readability = config.get("w_readability", 1.0)
        w_alignment_consistency = config.get("w_alignment_consistency", 1.0)
        w_alignment_similarity = config.get("w_alignment_similarity", params.W_ALIGNMENT_SIMILARITY)
        w_data_ink = config.get("w_data_ink", params.W_DATA_INK)
        enable_grid_search_init = config.get("enable_grid_search_init", True)
        grid_search_config = config.get("grid_search_config", None)
        initial_only = config.get("initial_only", False)
        
        # Extract reference bboxes from nodes (initial bboxes from Example layout)
        reference_bboxes = []
        for node in nodes:
            bbox = node.get("bbox", (0, 0, 100, 100))
            if isinstance(bbox, dict):
                # Convert dict bbox to tuple
                ref_bbox = (
                    bbox.get("x", 0),
                    bbox.get("y", 0),
                    bbox.get("width", bbox.get("w", 100)),
                    bbox.get("height", bbox.get("h", 100))
                )
            else:
                ref_bbox = bbox
            reference_bboxes.append(ref_bbox)
        
        # Reference parent container bbox (from container_bbox parameter)
        reference_parent_bbox = container_bbox
        
        # Extract size rules from constraints (only significant size differences)
        size_rules = []
        if constraints:
            print("relative_size:", constraints.get("relative_size"))
            relative_size_constraints = constraints.get("relative_size", [])
            for rel_constraint in relative_size_constraints:
                ratio = rel_constraint.get("ratio", 1.0)
                source_idx = rel_constraint.get("source_index")
                target_idx = rel_constraint.get("target_index")
                
                if source_idx is not None and target_idx is not None:
                    # Only keep rules with significant size differences
                    # ratio >= SIZE_RATIO_THRESHOLD: source should be larger than target
                    # ratio <= 1/SIZE_RATIO_THRESHOLD: target should be larger than source
                    threshold = params.SIZE_RATIO_THRESHOLD
                    if ratio >= threshold:
                        size_rules.append((source_idx, target_idx))
                    elif ratio <= 1.0 / threshold:
                        size_rules.append((target_idx, source_idx))
        if debug:
            print("size rules:", size_rules)
        
        # Extract min_width and min_height from each node
        min_sizes = []
        for node in nodes:
            # Check multiple possible locations for min_width and min_height
            min_width = node.get("min_width")
            min_height = node.get("min_height")
            
            # If not found at node level, check bbox dict
            if min_width is None or min_height is None:
                bbox = node.get("bbox", {})
                if isinstance(bbox, dict):
                    min_width = min_width or bbox.get("min_width")
                    min_height = min_height or bbox.get("min_height")
            
            # If still not found, check metadata
            if min_width is None or min_height is None:
                metadata = node.get("metadata", {})
                min_width = min_width or metadata.get("min_width")
                min_height = min_height or metadata.get("min_height")
            
            # Use defaults if not found
            if min_width is None:
                min_width = params.MIN_WIDTH_DEFAULT
            if min_height is None:
                min_height = params.MIN_HEIGHT_DEFAULT
            
            min_sizes.append((float(min_width), float(min_height)))
        
        if debug:
            print(f"Min sizes extracted: {min_sizes}")
        
        # Extract proximity info for proximity ratio loss
        # Simplified for bottom-up layout: only considers current layer's children spacing
        # and children's children spacing
        proximity_info = None
        w_proximity = config.get("w_proximity", params.W_PROXIMITY)
        
        if w_proximity > 0:
            # Get container type from config (passed from hierarchical_optimizer)
            container_type = config.get("container_type", "row")  # Default to row
            
            # Get grandchildren list from config (passed from hierarchical_optimizer)
            grandchildren_list_from_config = config.get("grandchildren_list", [])
            
            # Build proximity_info for current container
            # Container is the parent container
            container_bbox = container_bbox
            
            # Children are the N elements (will be updated during optimization)
            # For now, use initial bboxes
            child_bboxes = []
            grandchild_bboxes_list = []  # List of lists: grandchildren for each child
            
            # Ensure grandchildren_list_from_config is a list of lists
            # If it's a flat list, we need to handle it differently
            if len(grandchildren_list_from_config) > 0:
                first_elem = grandchildren_list_from_config[0]
                if isinstance(first_elem, (tuple, list)) and len(first_elem) == 4 and isinstance(first_elem[0], (int, float)):
                    # Flat list: all grandchildren in one list, need to distribute to children
                    # This shouldn't happen, but handle it gracefully
                    # print(f"Warning: grandchildren_list_from_config appears to be a flat list, expected list of lists")
                    # For now, assign all grandchildren to first child (not ideal, but better than crashing)
                    if len(nodes) > 0:
                        grandchild_bboxes_list = [grandchildren_list_from_config] + [[]] * (len(nodes) - 1)
                    else:
                        grandchild_bboxes_list = []
                else:
                    # Proper nested structure: list of lists
                    grandchild_bboxes_list = grandchildren_list_from_config[:len(nodes)]
                    # Pad with empty lists if needed
                    while len(grandchild_bboxes_list) < len(nodes):
                        grandchild_bboxes_list.append([])
            
            for i, node in enumerate(nodes):
                bbox = node.get("bbox", (0, 0, 100, 100))
                if isinstance(bbox, dict):
                    child_bboxes.append((
                        bbox.get("x", 0),
                        bbox.get("y", 0),
                        bbox.get("width", bbox.get("w", 100)),
                        bbox.get("height", bbox.get("h", 100))
                    ))
                else:
                    child_bboxes.append(bbox)
                
                # Use grandchildren from grandchild_bboxes_list (already processed above)
                # If not available, try to extract from node metadata as fallback
                if i >= len(grandchild_bboxes_list) or not grandchild_bboxes_list[i]:
                    grandchildren = []
                    node_children = node.get("children", [])
                    if isinstance(node_children, list):
                        for grandchild in node_children:
                            grandchild_bbox = grandchild.get("bbox") if isinstance(grandchild, dict) else None
                            if grandchild_bbox:
                                if isinstance(grandchild_bbox, dict):
                                    grandchildren.append((
                                        grandchild_bbox.get("x", 0),
                                        grandchild_bbox.get("y", 0),
                                        grandchild_bbox.get("width", grandchild_bbox.get("w", 0)),
                                        grandchild_bbox.get("height", grandchild_bbox.get("h", 0))
                                    ))
                                elif isinstance(grandchild_bbox, (tuple, list)) and len(grandchild_bbox) >= 4:
                                    grandchildren.append(tuple(grandchild_bbox[:4]))
                    if i < len(grandchild_bboxes_list):
                        grandchild_bboxes_list[i] = grandchildren
                    else:
                        grandchild_bboxes_list.append(grandchildren)
            
            proximity_info = {
                "containers": [container_bbox],
                "children": [child_bboxes],
                "grandchildren": grandchild_bboxes_list,
                "types": [container_type],
                "weights": None  # Will use area-based weighting
            }
            if debug:
                print(f"Proximity info prepared: container={container_bbox}, type={container_type}, "
                      f"children={len(child_bboxes)}, grandchildren={[len(gc) for gc in grandchild_bboxes_list]}")
        
        # Store original image paths for saving composite image
        original_png_list = original_paths
        
        # Get save prefix from parameter or config
        save_prefix = save_prefix or config.get("save_prefix")
        output_dir = config.get("output_dir")
        # Call optimize function with N-node support
        # Use temp files for optimization, but pass original paths for saving
        if debug:
            print(f"[SDFOptimizationStrategy] Calling test_sdf.optimize for container_type={container_type}, num_nodes={len(nodes)}")
        w_fully_inside = config.get("w_fully_inside", params.W_FULLY_INSIDE)
        optimized_bboxes = sdf_optimize(
            png_list=image_paths,
            original_png_list=original_png_list,
            Wc=Wc,
            Hc=Hc,
            opt_res_list=opt_res_list,
            outer_rounds=outer_rounds,
            inner_steps=inner_steps,
            lr=lr,
            min_sizes=min_sizes,
            reference_bboxes=reference_bboxes,
            reference_parent_bbox=reference_parent_bbox,
            w_similarity=w_similarity,
            size_rules=size_rules,
            w_readability=w_readability,
            w_alignment_consistency=w_alignment_consistency,
            w_alignment_similarity=w_alignment_similarity,
            proximity_info=proximity_info,
            w_proximity=w_proximity,
            w_data_ink=w_data_ink,
            overlap_constraints=overlap_constraints,
            chart_dual_masks=chart_dual_masks,
            fully_overlap_pairs=fully_overlap_pairs,
            w_fully_inside=w_fully_inside,
            enable_grid_search_init=enable_grid_search_init,
            grid_search_config=grid_search_config,
            initial_only=initial_only,
            device=config.get("device"),
            save_prefix=save_prefix,
            output_dir=output_dir,
            debug=debug,
        )
        
        # Note: Don't cleanup temp files here because save_composite_image 
        # is called inside sdf_optimize and needs the files
        # Temp files will be cleaned up by Python's garbage collector or manually later
        
        return optimized_bboxes
    
    def composite(self, nodes: List[dict], bboxes: List[Tuple[float, float, float, float]],
                  container_bbox: Tuple[float, float, float, float], debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Composite optimized results.
        
        Args:
            nodes: List of node dictionaries
            bboxes: List of optimized bounding boxes
            container_bbox: Container bounding box
        
        Returns:
            Tuple of (mask, sdf)
        """
        return composite_nodes(nodes, bboxes, container_bbox, debug=debug)

