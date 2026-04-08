"""Reusable layout quality evaluator.

This module extracts and generalizes the loss aggregation logic used for ranking
layout candidates. It is independent from bilevel search and can be reused by
hierarchical ablation scripts or other pipelines.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from layout_system import parameters as params
from layout_system.sdf.core import load_binary_mask_from_rgba
from layout_system.sdf.losses import (
    compute_alignment_consistency_loss,
    compute_contrast_loss,
    compute_data_ink_loss_mask,
    compute_overlap_loss_mask,
    compute_proximity_ratio_loss,
    compute_repetition_loss,
    compute_visual_balance_loss,
)


LOSS_KEYS: Tuple[str, ...] = (
    "alignment",
    "visual_balance",
    "data_ink",
    "proximity",
    "contrast",
    "repetition",
)


class LayoutQualityEvaluator:
    """Compute raw and normalized layout quality metrics.

    Raw metrics are computed from a (tree, result) pair and then converted to a
    single total score via min-max normalization and weighted aggregation.

    Notes:
    - The data_ink key intentionally follows existing bilevel behavior:
      normalized display value uses (1 - L_norm), while total score still adds
      +weight * L_norm.
    """

    def __init__(
        self,
        base_dir: str = ".",
        weights: Optional[Dict[str, float]] = None,
        normalize_by_canvas: bool = False,
    ) -> None:
        self.base_dir = base_dir
        self.normalize_by_canvas = normalize_by_canvas
        self.weights = self.default_weights()
        if weights:
            self.weights.update(weights)
        self._mask_cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def default_weights() -> Dict[str, float]:
        """Return default weights aligned with bilevel ranking settings."""
        return {
            "alignment": float(getattr(params, "BILEVEL_W_ALIGNMENT", 1.0)),
            "visual_balance": float(getattr(params, "BILEVEL_W_BALANCE", 1.0)),
            "data_ink": float(getattr(params, "BILEVEL_W_INK", 1.0)),
            "proximity": float(getattr(params, "BILEVEL_W_PROXIMITY", 1.0)),
            "contrast": float(getattr(params, "BILEVEL_W_CONTRAST", 1.0)),
            "repetition": float(getattr(params, "BILEVEL_W_REPETITION", 1.0)),
        }

    def compute_raw_losses(
        self,
        tree: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute raw metric breakdown for one layout result."""
        similarity_raw = self._traverse_and_compute(tree, result, mode="sum")
        similarity_value = float(similarity_raw if not isinstance(similarity_raw, tuple) else similarity_raw[1])
        if self.normalize_by_canvas:
            element_count = max(1, len(self._flatten_with_context(tree, result)))
            similarity_value = similarity_value / float(element_count)

        loss_info = self._compute_container_losses(tree, result)
        loss_info["similarity"] = similarity_value
        return loss_info

    def compute_minmax_bounds(
        self,
        raw_dicts: Iterable[Dict[str, float]],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute global min/max bounds from raw dictionaries."""
        mins: Dict[str, float] = {}
        maxs: Dict[str, float] = {}
        initialized = False

        for raw in raw_dicts:
            if not initialized:
                for k in LOSS_KEYS:
                    v = self._to_sort_value(k, raw.get(k, 0.0))
                    mins[k] = v
                    maxs[k] = v
                initialized = True
                continue

            for k in LOSS_KEYS:
                v = self._to_sort_value(k, raw.get(k, 0.0))
                mins[k] = min(mins[k], v)
                maxs[k] = max(maxs[k], v)

        if not initialized:
            mins = {k: 0.0 for k in LOSS_KEYS}
            maxs = {k: 0.0 for k in LOSS_KEYS}

        return mins, maxs

    def score_raw(
        self,
        raw_dict: Dict[str, float],
        mins: Dict[str, float],
        maxs: Dict[str, float],
    ) -> Dict[str, Any]:
        """Score one raw dict using provided min/max bounds."""
        total = 0.0
        normalized: Dict[str, float] = {}

        for k in LOSS_KEYS:
            value = self._to_sort_value(k, raw_dict.get(k, 0.0))
            r = maxs[k] - mins[k]
            if r > 1e-12:
                l_norm = (value - mins[k]) / r
            else:
                l_norm = 0.5

            if k == "data_ink":
                normalized[k] = 1.0 - l_norm
            else:
                normalized[k] = l_norm

            total += float(self.weights.get(k, 1.0)) * l_norm

        return {
            "raw": dict(raw_dict),
            "normalized": normalized,
            "total": float(total),
        }

    def score_raw_batch(
        self,
        raw_dicts: List[Dict[str, float]],
        mins: Optional[Dict[str, float]] = None,
        maxs: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
        """Score a batch of raw dicts under shared min/max bounds."""
        if mins is None or maxs is None:
            mins, maxs = self.compute_minmax_bounds(raw_dicts)

        scored = [self.score_raw(raw, mins, maxs) for raw in raw_dicts]
        return scored, mins, maxs

    def find_worst_element(
        self,
        tree: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Tuple[str, float]:
        """Return (node_path, loss) for the worst text/image element."""
        out = self._traverse_and_compute(tree, result, mode="max")
        if isinstance(out, tuple):
            return out
        return "", float(out)

    @staticmethod
    def _to_sort_value(key: str, value: float) -> float:
        value = float(value)
        if key == "data_ink":
            return -value
        return abs(value)

    def _compute_container_losses(
        self,
        tree: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, float]:
        device = "cpu"
        all_metric_keys = list(LOSS_KEYS) + ["similarity", "overlap"]
        loss_info = {k: 0.0 for k in all_metric_keys}

        proximity_count = 0

        for tree_node, result_node in self._iter_containers(tree, result):
            t_children = tree_node.get("children", [])
            r_children = result_node.get("children", [])

            if len(t_children) < 2 or len(r_children) < 2:
                continue

            ref_bboxes = [self._ensure_bbox_tuple(c.get("bbox")) for c in t_children]
            gen_bboxes = [
                self._ensure_bbox_tuple(c.get("final_bbox") or c.get("bbox"))
                for c in r_children
            ]

            if len(ref_bboxes) != len(gen_bboxes):
                continue

            p_t = self._ensure_bbox_tuple(tree_node.get("bbox"), (0, 0, 1000, 1000))
            p_r = self._ensure_bbox_tuple(
                result_node.get("final_bbox") or result_node.get("bbox"),
                (0, 0, 1000, 1000),
            )
            wc, hc = int(p_r[2]), int(p_r[3])
            container_area = max(1.0, float(wc * hc))
            container_diag_sq = max(1.0, float(wc * wc + hc * hc))
            container_perimeter = max(1.0, float(wc + hc))

            gen_tensors = [torch.tensor(b, dtype=torch.float32, device=device) for b in gen_bboxes]

            l_align = compute_alignment_consistency_loss(
                ref_bboxes,
                gen_tensors,
                p_t,
                p_r,
                device=device,
            )
            v_align = float(l_align.item())
            if self.normalize_by_canvas:
                v_align = v_align / container_perimeter
            loss_info["alignment"] += v_align

            priorities_list = [c.get("priority") for c in t_children]
            priorities = (
                [float(p) for p in priorities_list]
                if all(p is not None for p in priorities_list)
                else None
            )
            l_contrast = compute_contrast_loss(ref_bboxes, gen_bboxes, priorities)
            if self.normalize_by_canvas:
                l_contrast = l_contrast / container_area
            loss_info["contrast"] += l_contrast

            element_types = [c.get("type", "unknown") for c in t_children]
            l_repetition = compute_repetition_loss(gen_bboxes, element_types)
            loss_info["repetition"] += l_repetition

            masks: List[Any] = []
            bboxes_rel: List[Tuple[float, float, float, float]] = []
            masks_for_ink: List[np.ndarray] = []
            bboxes_for_ink: List[Tuple[float, float, float, float]] = []

            for tc, rc in zip(t_children, r_children):
                bbox = self._ensure_bbox_tuple(rc.get("final_bbox") or rc.get("bbox"))
                mask, has_real_image_mask = self._load_mask_or_placeholder(tc.get("image_path"), bbox)

                masks.append(mask)
                bboxes_rel.append(bbox)

                if has_real_image_mask:
                    masks_for_ink.append(mask)
                    bboxes_for_ink.append(bbox)

            bboxes_for_balance = [
                (
                    torch.tensor(b[0], dtype=torch.float32, device=device),
                    torch.tensor(b[1], dtype=torch.float32, device=device),
                    torch.tensor(b[2], dtype=torch.float32, device=device),
                    torch.tensor(b[3], dtype=torch.float32, device=device),
                )
                for b in gen_bboxes
            ]
            l_balance = compute_visual_balance_loss(
                bboxes_for_balance,
                wc,
                hc,
                masks=masks,
                device=device,
            )
            v_balance = float(l_balance.item())
            if self.normalize_by_canvas:
                # Distance-squared to center normalized by the squared half-diagonal.
                v_balance = (4.0 * v_balance) / container_diag_sq
            loss_info["visual_balance"] += v_balance

            container_type = tree_node.get("type", "row")
            grandchildren = []
            for rc in r_children:
                gc = rc.get("children", [])
                gc_bboxes = [
                    self._ensure_bbox_tuple(x.get("final_bbox") or x.get("bbox"))
                    for x in gc
                ]
                grandchildren.append(gc_bboxes)

            l_prox = compute_proximity_ratio_loss(
                [p_r],
                [gen_bboxes],
                [grandchildren],
                [container_type],
                epsilon=params.PROXIMITY_EPSILON,
                device=device,
            )
            v_prox = float(l_prox.item())
            if v_prox != 0.0:
                loss_info["proximity"] += v_prox
                proximity_count += 1

            if masks_for_ink and bboxes_for_ink:
                l_ink = float(compute_data_ink_loss_mask(masks_for_ink, bboxes_for_ink, wc, hc))
                if self.normalize_by_canvas:
                    l_ink = l_ink / container_area
                loss_info["data_ink"] += l_ink

            if masks and bboxes_rel and len(masks) >= 2:
                overlap_constraints = tree_node.get("constraints", {}).get("overlap")
                l_overlap = float(
                    compute_overlap_loss_mask(masks, bboxes_rel, wc, hc, overlap_constraints)
                )
                if self.normalize_by_canvas:
                    l_overlap = l_overlap / container_area
                loss_info["overlap"] += l_overlap

        if proximity_count > 1:
            loss_info["proximity"] /= proximity_count

        return loss_info

    def _load_mask_or_placeholder(
        self,
        image_path: Optional[str],
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, bool]:
        if image_path:
            full_path = image_path if os.path.isabs(image_path) else os.path.join(self.base_dir, image_path)
            if full_path in self._mask_cache:
                return self._mask_cache[full_path], True
            if os.path.exists(full_path):
                mask = load_binary_mask_from_rgba(full_path)
                self._mask_cache[full_path] = mask
                return mask, True

        x, y, w, h = bbox
        _ = x, y
        return np.ones((max(1, int(h)), max(1, int(w))), dtype=np.float32), False

    def _iter_containers(self, tree_node: Dict[str, Any], result_node: Dict[str, Any]):
        t_children = tree_node.get("children", [])
        r_children = result_node.get("children", [])

        if t_children and r_children:
            yield tree_node, result_node

        for tc, rc in zip(t_children, r_children):
            yield from self._iter_containers(tc, rc)

    def _ensure_bbox_tuple(
        self,
        bbox: Any,
        default: Tuple[float, float, float, float] = (0, 0, 100, 100),
    ) -> Tuple[float, float, float, float]:
        if bbox is None:
            return default
        if isinstance(bbox, dict):
            return (
                bbox.get("x", 0),
                bbox.get("y", 0),
                bbox.get("w", bbox.get("width", 0)),
                bbox.get("h", bbox.get("height", 0)),
            )
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return tuple(bbox[:4])
        return default

    def _traverse_and_compute(
        self,
        tree_node: Dict[str, Any],
        result_node: Dict[str, Any],
        mode: str = "sum",
    ):
        elements = self._flatten_with_context(tree_node, result_node)

        total_loss = 0.0
        max_loss = -1.0
        max_path = ""

        for item in elements:
            path = item["path"]
            t_bbox = item["t_bbox"]
            r_bbox = item["r_bbox"]
            p_t_bbox = item["p_t_bbox"]
            p_r_bbox = item["p_r_bbox"]

            loss = self._compute_single_element_loss(t_bbox, r_bbox, p_t_bbox, p_r_bbox)
            total_loss += loss

            # Keep existing behavior used by bilevel worst-node selection.
            if loss > max_loss and item["node"].get("type") in ("text", "image"):
                max_loss = loss
                max_path = path

        if mode == "sum":
            return total_loss
        return max_path, max_loss

    def _flatten_with_context(
        self,
        tree_node: Dict[str, Any],
        result_node: Dict[str, Any],
        path: str = "",
        parent_t_bbox: Optional[Tuple[float, float, float, float]] = None,
        parent_r_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []

        def get_bbox(node: Dict[str, Any], is_result: bool = False):
            if is_result and "final_bbox" in node and node["final_bbox"]:
                return node["final_bbox"]
            return node.get("bbox", [0, 0, 100, 100])

        def ensure_tuple(bbox: Any) -> Tuple[float, float, float, float]:
            if isinstance(bbox, dict):
                return (
                    bbox.get("x", 0),
                    bbox.get("y", 0),
                    bbox.get("w", bbox.get("width", 0)),
                    bbox.get("h", bbox.get("height", 0)),
                )
            if isinstance(bbox, (list, tuple)):
                return tuple(bbox[:4])
            return (0, 0, 100, 100)

        current_t_bbox = ensure_tuple(get_bbox(tree_node))
        current_r_bbox = ensure_tuple(get_bbox(result_node, is_result=True))

        if parent_t_bbox is not None:
            items.append(
                {
                    "path": path,
                    "node": tree_node,
                    "t_bbox": current_t_bbox,
                    "r_bbox": current_r_bbox,
                    "p_t_bbox": parent_t_bbox,
                    "p_r_bbox": parent_r_bbox,
                }
            )

        t_children = tree_node.get("children", [])
        r_children = result_node.get("children", [])

        if len(t_children) != len(r_children):
            raise ValueError(
                f"Tree and result structure mismatch at {path}: "
                f"{len(t_children)} vs {len(r_children)}"
            )

        for i, (tc, rc) in enumerate(zip(t_children, r_children)):
            child_path = f"{path}-{i}" if path else str(i)
            items.extend(
                self._flatten_with_context(
                    tc,
                    rc,
                    child_path,
                    current_t_bbox,
                    current_r_bbox,
                )
            )

        return items

    @staticmethod
    def _compute_single_element_loss(
        t_bbox: Tuple[float, float, float, float],
        r_bbox: Tuple[float, float, float, float],
        p_t_bbox: Tuple[float, float, float, float],
        p_r_bbox: Tuple[float, float, float, float],
    ) -> float:
        tx, ty, tw, th = t_bbox
        rx, ry, rw, rh = r_bbox

        ptx, pty, ptw, pth = p_t_bbox
        prx, pry, prw, prh = p_r_bbox

        ptw = max(ptw, 1.0)
        pth = max(pth, 1.0)
        prw = max(prw, 1.0)
        prh = max(prh, 1.0)

        vt = np.array([(tx - ptx) / ptw, (ty - pty) / pth, tw / ptw, th / pth])
        vr = np.array([(rx - prx) / prw, (ry - pry) / prh, rw / prw, rh / prh])

        return float(np.linalg.norm(vt - vr))
