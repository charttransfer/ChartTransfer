"""Grid-search-based optimization strategy (ablation baseline for SDF strategy)."""

import numpy as np
import tempfile
import os
from PIL import Image
from typing import List, Tuple, Dict, Any, Set
from .base import OptimizationStrategy
from layout_system import parameters as params
from layout_system.sdf import grid_search_optimize
from layout_system.utils.composite import composite_nodes

from .sdf_strategy import (
    _get_not_overlap_path,
    _get_no_grid_path,
    _get_fully_overlap_pairs,
    _build_chart_dual_masks_for_fully_overlap,
    _find_nodes_with_overlap,
)


class GridSearchOptimizationStrategy(OptimizationStrategy):
    """Grid-search-based optimization strategy.

    Drop-in replacement for SDFOptimizationStrategy that uses discrete
    coordinate-descent search instead of gradient-based optimization.
    All loss functions are identical to the SDF strategy.
    """

    def optimize(self, nodes: List[dict], container_bbox: Tuple[float, float, float, float],
                 constraints: dict, config: dict, save_prefix: str = None) -> List[Tuple[float, float, float, float]]:
        container_type = config.get("container_type", "unknown")
        debug = config.get("debug_strategy", config.get("debug", False))
        if debug:
            print(f"[GridSearchStrategy] optimize called: container_type={container_type}, num_nodes={len(nodes)}")

        if len(nodes) < 2:
            if nodes:
                bbox = nodes[0].get("bbox", (0, 0, 100, 100))
                return [bbox]
            return []

        overlap_constraints = constraints.get("overlap", [])
        chart_dual_masks = _build_chart_dual_masks_for_fully_overlap(constraints, nodes, debug=debug)
        fully_overlap_pairs = _get_fully_overlap_pairs(constraints, nodes)
        overlap_node_indices = _find_nodes_with_overlap(constraints, nodes)

        # Build image paths (same logic as SDFOptimizationStrategy)
        image_paths = []
        original_paths = []
        temp_files = []

        for i, node in enumerate(nodes):
            metadata = node.get("metadata", {})
            original_image_path = metadata.get("image_path") or node.get("image_path")
            mask = node.get("mask")
            node_type = node.get("type", "")

            original_paths.append(original_image_path)

            image_path = original_image_path
            if node_type == "chart" and image_path:
                if i in overlap_node_indices:
                    image_path = _get_not_overlap_path(image_path, debug=debug)
                else:
                    image_path = _get_no_grid_path(image_path, debug=debug)

            if image_path and os.path.exists(image_path):
                image_paths.append(image_path)
            elif mask is not None:
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_image = Image.fromarray(mask_uint8, mode='L')
                rgba_image = Image.new('RGBA', mask_image.size, (255, 255, 255, 255))
                rgba_image.putalpha(mask_image)
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                rgba_image.save(temp_file.name)
                temp_files.append(temp_file.name)
                image_paths.append(temp_file.name)
            else:
                image_paths.append(None)

        if not all(image_paths):
            for tf in temp_files:
                if os.path.exists(tf):
                    os.unlink(tf)
            return [node.get("bbox", (0, 0, 100, 100)) for node in nodes]

        _, _, Wc, Hc = container_bbox
        Wc, Hc = int(Wc), int(Hc)

        # Extract optimisation parameters
        w_similarity = config.get("w_similarity", 1.0)
        w_readability = config.get("w_readability", 1.0)
        w_alignment_consistency = config.get("w_alignment_consistency", 1.0)
        w_alignment_similarity = config.get("w_alignment_similarity", params.W_ALIGNMENT_SIMILARITY)
        w_data_ink = config.get("w_data_ink", params.W_DATA_INK)
        enable_grid_search_init = config.get("enable_grid_search_init", True)
        grid_search_config = config.get("grid_search_config", None)
        initial_only = config.get("initial_only", False)

        # Grid search specific parameters
        num_rounds = config.get("num_rounds", 3)
        position_grid_size = config.get("position_grid_size", 15)
        scale_steps = config.get("scale_steps", 8)

        # Reference bboxes
        reference_bboxes = []
        for node in nodes:
            bbox = node.get("bbox", (0, 0, 100, 100))
            if isinstance(bbox, dict):
                ref_bbox = (
                    bbox.get("x", 0), bbox.get("y", 0),
                    bbox.get("width", bbox.get("w", 100)),
                    bbox.get("height", bbox.get("h", 100)),
                )
            else:
                ref_bbox = bbox
            reference_bboxes.append(ref_bbox)

        reference_parent_bbox = container_bbox

        # Size rules
        size_rules = []
        if constraints:
            relative_size_constraints = constraints.get("relative_size", [])
            for rel_constraint in relative_size_constraints:
                ratio = rel_constraint.get("ratio", 1.0)
                source_idx = rel_constraint.get("source_index")
                target_idx = rel_constraint.get("target_index")
                if source_idx is not None and target_idx is not None:
                    threshold = params.SIZE_RATIO_THRESHOLD
                    if ratio >= threshold:
                        size_rules.append((source_idx, target_idx))
                    elif ratio <= 1.0 / threshold:
                        size_rules.append((target_idx, source_idx))

        # Min sizes
        min_sizes = []
        for node in nodes:
            min_width = node.get("min_width")
            min_height = node.get("min_height")
            if min_width is None or min_height is None:
                bbox = node.get("bbox", {})
                if isinstance(bbox, dict):
                    min_width = min_width or bbox.get("min_width")
                    min_height = min_height or bbox.get("min_height")
            if min_width is None or min_height is None:
                metadata = node.get("metadata", {})
                min_width = min_width or metadata.get("min_width")
                min_height = min_height or metadata.get("min_height")
            if min_width is None:
                min_width = params.MIN_WIDTH_DEFAULT
            if min_height is None:
                min_height = params.MIN_HEIGHT_DEFAULT
            min_sizes.append((float(min_width), float(min_height)))

        # Proximity info
        proximity_info = None
        w_proximity = config.get("w_proximity", params.W_PROXIMITY)
        if w_proximity > 0:
            container_type = config.get("container_type", "row")
            grandchildren_list_from_config = config.get("grandchildren_list", [])

            child_bboxes = []
            grandchild_bboxes_list = []

            if len(grandchildren_list_from_config) > 0:
                first_elem = grandchildren_list_from_config[0]
                if isinstance(first_elem, (tuple, list)) and len(first_elem) == 4 and isinstance(first_elem[0], (int, float)):
                    if len(nodes) > 0:
                        grandchild_bboxes_list = [grandchildren_list_from_config] + [[]] * (len(nodes) - 1)
                else:
                    grandchild_bboxes_list = grandchildren_list_from_config[:len(nodes)]
                    while len(grandchild_bboxes_list) < len(nodes):
                        grandchild_bboxes_list.append([])

            for i, node in enumerate(nodes):
                bbox = node.get("bbox", (0, 0, 100, 100))
                if isinstance(bbox, dict):
                    child_bboxes.append((
                        bbox.get("x", 0), bbox.get("y", 0),
                        bbox.get("width", bbox.get("w", 100)),
                        bbox.get("height", bbox.get("h", 100)),
                    ))
                else:
                    child_bboxes.append(bbox)

                if i >= len(grandchild_bboxes_list) or not grandchild_bboxes_list[i]:
                    grandchildren = []
                    node_children = node.get("children", [])
                    if isinstance(node_children, list):
                        for grandchild in node_children:
                            grandchild_bbox = grandchild.get("bbox") if isinstance(grandchild, dict) else None
                            if grandchild_bbox:
                                if isinstance(grandchild_bbox, dict):
                                    grandchildren.append((
                                        grandchild_bbox.get("x", 0), grandchild_bbox.get("y", 0),
                                        grandchild_bbox.get("width", grandchild_bbox.get("w", 0)),
                                        grandchild_bbox.get("height", grandchild_bbox.get("h", 0)),
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
                "weights": None,
            }

        save_prefix = save_prefix or config.get("save_prefix")
        output_dir = config.get("output_dir")
        w_fully_inside = config.get("w_fully_inside", params.W_FULLY_INSIDE)

        optimized_bboxes = grid_search_optimize(
            png_list=image_paths,
            original_png_list=original_paths,
            Wc=Wc, Hc=Hc,
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
            num_rounds=num_rounds,
            position_grid_size=position_grid_size,
            scale_steps=scale_steps,
            device=config.get("device"),
            save_prefix=save_prefix,
            output_dir=output_dir,
            debug=debug,
        )

        return optimized_bboxes

    def composite(self, nodes: List[dict], bboxes: List[Tuple[float, float, float, float]],
                  container_bbox: Tuple[float, float, float, float], debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        return composite_nodes(nodes, bboxes, container_bbox, debug=debug)
