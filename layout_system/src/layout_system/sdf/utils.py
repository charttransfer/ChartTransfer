"""Shared utilities for the SDF layout optimizer family.

Functions here are used by both the gradient-based SDF optimizer
(optimizer.py) and the grid-search ablation baseline
(grid_search_optimizer.py) to avoid code duplication.
"""

from typing import List, Optional, Tuple

import torch

from .bbox import bbox_aspect_from_unconstrained, unconstrained_from_bbox
from .losses import _get_pair_overlap_type


def check_all_non_overlap(overlap_constraints, num_nodes: int) -> bool:
    """Return True if every element pair has at most a non-overlap constraint.

    Grid-search initialisation is only safe when no pair is constrained to
    overlap (fully or partially).  Both the SDF and GridSearch optimizers
    gate that init path through this check.
    """
    if not overlap_constraints:
        return True
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if _get_pair_overlap_type(overlap_constraints, i, j) in (
                "fully_overlap",
                "partially_overlap",
            ):
                return False
    return True


def normalize_reference_bboxes(
    reference_bboxes: list,
    reference_parent_bbox: Optional[Tuple],
) -> Tuple[list, Optional[Tuple]]:
    """Subtract the minimum x/y so all reference coordinates start from 0.

    Both optimizers apply this normalisation after loading reference bboxes;
    this helper centralises the logic.
    """
    if not reference_bboxes:
        return reference_bboxes, reference_parent_bbox

    x_min = min(bbox[0] for bbox in reference_bboxes)
    y_min = min(bbox[1] for bbox in reference_bboxes)

    normalized: list = []
    for bbox in reference_bboxes:
        if isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
            normalized.append((bbox[0] - x_min, bbox[1] - y_min, bbox[2], bbox[3]))
        elif isinstance(bbox, dict):
            normalized.append(
                {
                    "x": bbox.get("x", 0) - x_min,
                    "y": bbox.get("y", 0) - y_min,
                    "width": bbox.get("width", bbox.get("w", 100)),
                    "height": bbox.get("height", bbox.get("h", 100)),
                }
            )
        else:
            normalized.append(bbox)

    if reference_parent_bbox is not None:
        x_p, y_p, w_p, h_p = reference_parent_bbox
        reference_parent_bbox = (x_p - x_min, y_p - y_min, w_p, h_p)

    return normalized, reference_parent_bbox


def unconstrained_from_reference_bboxes(
    reference_bboxes: list,
    num_nodes: int,
    min_sizes: List[Tuple[float, float]],
    ratios: List[float],
    Wc: int,
    Hc: int,
    size_min: float,
    debug: bool = False,
) -> List[Tuple[float, float, float]]:
    """Parse reference bboxes and return unconstrained (tx, ty, ts) float triples.

    The unconstrained parameterisation respects per-element aspect ratio and
    minimum-size constraints.  The SDF optimizer wraps these as
    ``torch.nn.Parameter`` objects; for concrete bbox tuples see
    :func:`init_bboxes_from_reference`.
    """
    params_list: List[Tuple[float, float, float]] = []
    for i in range(num_nodes):
        rb = reference_bboxes[i]
        if isinstance(rb, (tuple, list)) and len(rb) >= 4:
            ref_x, ref_y, ref_w, ref_h = rb[0], rb[1], rb[2], rb[3]
        elif isinstance(rb, dict):
            ref_x = rb.get("x", 0)
            ref_y = rb.get("y", 0)
            ref_w = rb.get("width", rb.get("w", 100))
            ref_h = rb.get("height", rb.get("h", 100))
        else:
            ref_x, ref_y, ref_w, ref_h = 0.0, 0.0, 100.0, 100.0

        mw, mh = min_sizes[i]
        tx, ty, ts = unconstrained_from_bbox(
            ref_x, ref_y, ref_w, ref_h, Wc, Hc, ratios[i],
            min_width=mw, min_height=mh, size_min=size_min, debug=debug,
        )
        params_list.append((tx, ty, ts))
    return params_list


def init_bboxes_from_reference(
    reference_bboxes: list,
    num_nodes: int,
    min_sizes: List[Tuple[float, float]],
    ratios: List[float],
    Wc: int,
    Hc: int,
    size_min: float,
    device: str = "cpu",
) -> List[Tuple[float, float, float, float]]:
    """Convert reference bboxes to concrete (x, y, w, h) tuples.

    Internally calls :func:`unconstrained_from_reference_bboxes` and then
    projects back to actual bounding boxes via the unconstrained
    parameterisation, ensuring aspect-ratio and min-size constraints are
    respected.  Used by the GridSearch optimizer's reference-init path.
    """
    unconstrained = unconstrained_from_reference_bboxes(
        reference_bboxes, num_nodes, min_sizes, ratios, Wc, Hc, size_min,
    )
    bboxes: List[Tuple[float, float, float, float]] = []
    for i, (tx_i, ty_i, ts_i) in enumerate(unconstrained):
        mw, mh = min_sizes[i]
        x, y, w, h = bbox_aspect_from_unconstrained(
            torch.tensor(tx_i, device=device),
            torch.tensor(ty_i, device=device),
            torch.tensor(ts_i, device=device),
            Wc, Hc, ratios[i],
            min_width=mw, min_height=mh, size_min=size_min,
        )
        bboxes.append((x.item(), y.item(), w.item(), h.item()))
    return bboxes
