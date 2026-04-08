import copy
from collections.abc import Iterator
from typing import Optional


def split_node_path(path_str) -> list[int]:
    """Convert a path string like "0-1-2" into a list of integers [0, 1, 2]."""
    if not path_str:
        return []
    return [int(part) for part in path_str.split("-")]


def join_node_path(parts) -> str:
    """Convert a list of integers [0, 1, 2] into a path string like "0-1-2"."""
    return "-".join(map(str, parts))


def get_node_by_path(tree, path):
    """
    Given a tree and a dash-separated index path, return the node at that path.
    Example path: "0-1" (root's 0th child -> its 1st child). Root path is "".
    """
    if path is None:
        return None  # Explicitly handle None to avoid confusion with root path
    if path == "":
        return tree  # Root node
    parts = split_node_path(path)
    current = tree
    try:
        for idx in parts:
            current = current["children"][idx]
        return current
    except (KeyError, IndexError, TypeError):
        return None


def _normalize_node_type(t):
    return str(t).lower() if t else "unknown"


def get_structure_signature(node, exclude_node=None):
    """
    Generates a structural signature string for a scene tree node.

    The signature captures the hierarchy, node types, and specific attributes like alignment.
    - For all nodes, alignment is included (e.g., "column:left").
    - For 'layer' nodes, children signatures are sorted to ensure order independence.

    Args:
        node (dict): The scene tree node.
        exclude_node (dict|None): Optional node to exclude from the signature.

    Returns:
        str: A string signature like "column:center(text:left,image:center)".
    """
    raw_type = _normalize_node_type(node.get("type", "unknown"))
    t = raw_type

    alignment = node.get("alignment")
    if alignment:
        t = f"{t}:{alignment}"

    children = node.get("children", [])
    if exclude_node is not None:
        children = [c for c in children if c is not exclude_node]

    if not children:
        return t  # Leaf node

    c_sigs = [get_structure_signature(c, exclude_node=exclude_node) for c in children]

    if len(c_sigs) == 1:
        return c_sigs[0]  # Flatten single-child groups in signature

    # For layer, sort children signatures (order agnostic)
    if raw_type == "layer":
        c_sigs.sort()

    return f"{t}({','.join(c_sigs)})"


def _collect_leaf_types(node):
    children = node.get("children", [])
    if not children:
        return [_normalize_node_type(node.get("type", "unknown"))]
    types = []
    for c in children:
        types.extend(_collect_leaf_types(c))
    return types


def get_content_signature(node):
    # Tuple of sorted leaf types
    leaves = _collect_leaf_types(node)
    return tuple(sorted(leaves))


def remove_element_from_tree(tree, target_path_str):
    """
    Remove a node from the scene graph tree specified by its path.

    This function handles:
    1. Removing the node from its parent's 'children' list.
    2. Updating constraints in the parent node:
       - Removing constraints that reference the deleted node index.
       - Decrementing indices for constraints referencing subsequent nodes.
    3. Handling degeneracy:
       - If a 'row', 'column', or 'layer' node has only one child left after removal,
         it is replaced by that single child (flattening).
         - If the degenerate node is the root, the root's content is replaced in-place.
         - If the degenerate node is a child, it is replaced in its parent (the grandparent).

    Args:
        tree (dict): The root of the layout tree (modified in-place).
        target_path_str (str): Dash-separated path to the node to remove (e.g., "0-1").
            The root path is an empty string ("").

    Returns:
        bool: True if the modification was successful, False otherwise (e.g., path not found, invalid index).
    """

    # --- Helper to traverse and capture lineage ---
    def get_lineage(root, path_parts):
        """
        Traverses the tree based on path parts and returns the lineage.
        Returns a list of (parent_node, index_of_child_in_parent).
        The last item in the list represents the connection to the target node.
        """
        current = root
        lineage = []  # Store tuples of (node, index_to_next_child)

        for idx in path_parts:
            lineage.append((current, idx))
            try:
                current = current["children"][idx]
            except (KeyError, IndexError, TypeError):
                return None  # Invalid path

        return lineage

    # 1. Parse Path and Get Node Lineage
    if not target_path_str:
        return False  # Cannot remove root node via this function

    parts = split_node_path(target_path_str)
    lineage = get_lineage(tree, parts)

    if not lineage:
        return False

    # unpack the immediate parent and the index of the node to remove
    parent, target_index = lineage[-1]

    # 2. Update Constraints in Parent
    # We must do this before popping the child to ensure indices match current state
    if "constraints" in parent and isinstance(parent["constraints"], dict):
        new_constraints_dict = {}

        for c_type, c_obj in parent["constraints"].items():
            # Only process list-based constraints (like relative_size, orientation, overlap)
            # which typically use source_index/target_index.
            # Dict-based constraints (alignment, padding, gap) usually apply to the container
            # or all items uniformly and don't strictly require index updates.

            if isinstance(c_obj, list):
                new_list = []
                for c in c_obj:
                    # Skip if not a dictionary (unlikely but safe)
                    if not isinstance(c, dict):
                        new_list.append(c)
                        continue

                    s_idx = c.get("source_index")
                    t_idx = c.get("target_index")

                    # Pass through constraints that don't use indices
                    if s_idx is None and t_idx is None:
                        new_list.append(c)
                        continue

                    # 2a. Remove constraint if it involves the deleted node
                    if s_idx == target_index or t_idx == target_index:
                        continue

                    # 2b. Shift indices > target_index
                    new_c = c.copy()

                    if s_idx is not None and s_idx > target_index:
                        new_c["source_index"] = s_idx - 1

                    if t_idx is not None and t_idx > target_index:
                        new_c["target_index"] = t_idx - 1

                    new_list.append(new_c)

                if new_list:
                    new_constraints_dict[c_type] = new_list
            else:
                # Keep non-list constraints (e.g., dicts) as is
                new_constraints_dict[c_type] = c_obj

        parent["constraints"] = new_constraints_dict

    # 3. Remove the child node
    parent["children"].pop(target_index)

    # 4. Handle Degeneracy (Single-child Group Flattening)
    # If a structural group (row/column/layer) has only 1 child left, replace it with that child.
    if (
        parent.get("type") in ["row", "column", "layer"]
        and len(parent["children"]) == 1
    ):
        single_child = parent["children"][0]

        if len(lineage) >= 2:
            # Case A: Parent has a parent (Grandparent exists)
            grandparent, parent_index_in_gp = lineage[-2]

            # Replace 'parent' with 'single_child' in 'grandparent'
            grandparent["children"][parent_index_in_gp] = single_child

            # NOTE: Constraints in grandparent that referred to 'parent' (via parent_index_in_gp)
            # will now effectively refer to 'single_child'.
            # This preserves structural intent (e.g. "Item A is left of [Group B]")
            # becomes "Item A is left of [Content of B]".

        else:
            # Case B: Parent is the Root (lineage has length 1)
            # We cannot replace the root dict object itself in the caller's variable,
            # so we clear and update it in-place.
            parent.clear()
            parent.update(single_child)

    return True


def _extract_bbox_list(node: dict) -> Optional[list[float]]:
    """Return bbox as [x, y, w, h] list when available, else None."""
    bbox = node.get("bbox")
    if isinstance(bbox, dict):
        x = bbox.get("x")
        y = bbox.get("y")
        w = bbox.get("width")
        h = bbox.get("height")
        if None in (x, y, w, h):
            return None
        return [x, y, w, h]  # type: ignore
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return list(bbox)
    return None


def _set_bbox_from_list(node: dict, bbox_list: list[float]) -> None:
    """Set bbox values on node, supporting dict or list-based bbox formats."""
    bbox = node.get("bbox")
    x, y, w, h = bbox_list
    if isinstance(bbox, dict):
        bbox["x"] = x
        bbox["y"] = y
        bbox["width"] = w
        bbox["height"] = h
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        node["bbox"] = [x, y, w, h]


def _apply_bbox_transform(node, src_bbox, dst_bbox):
    """Scale and translate all bbox values so src_bbox aligns with dst_bbox."""
    sx = 1.0
    sy = 1.0

    if src_bbox[2] and src_bbox[3]:
        sx = dst_bbox[2] / src_bbox[2]
        sy = dst_bbox[3] / src_bbox[3]

    def _recurse(current):
        bbox_list = _extract_bbox_list(current)
        if bbox_list:
            x, y, w, h = bbox_list
            nx = dst_bbox[0] + (x - src_bbox[0]) * sx
            ny = dst_bbox[1] + (y - src_bbox[1]) * sy
            nw = w * sx
            nh = h * sy
            _set_bbox_from_list(current, [nx, ny, nw, nh])

        for child in current.get("children", []):
            _recurse(child)

    _recurse(node)


def collect_leaf_nodes(node: dict) -> list[dict]:
    """Collect leaf nodes (nodes without children) under the given node."""
    children = node.get("children", [])
    if not children:
        return [node]
    leaves = []
    for child in children:
        leaves.extend(collect_leaf_nodes(child))
    return leaves


def collect_leaf_paths(node: dict, prefix: str = "") -> list[tuple[str, dict]]:
    """
    Collect leaf nodes along with their dash-separated paths from the root.
    Returns a list of tuples: (path_str, leaf_node_dict).
    """
    children = node.get("children", [])
    if not children:
        return [(prefix, node)]

    paths = []
    for i, child in enumerate(children):
        child_path = f"{prefix}-{i}" if prefix else str(i)
        paths.extend(collect_leaf_paths(child, child_path))
    return paths


def _layer_sort_key(node: dict, index: int, exclude_node=None):
    """Sort key for layer children: type first (chart, text, image) then signature."""
    signature = get_structure_signature(node, exclude_node=exclude_node)
    bbox_list = _extract_bbox_list(node) or [0, 0, 0, 0]
    area = bbox_list[2] * bbox_list[3]
    return (signature, area, bbox_list[2], bbox_list[3], index)


def _order_layer_children(children: list[dict], exclude_node=None) -> list[dict]:
    """Return layer children ordered by signature and bbox heuristics."""
    indexed = list(enumerate(children))
    indexed.sort(key=lambda item: _layer_sort_key(item[1], item[0], exclude_node))
    return [item[1] for item in indexed]


def _collapse_single_child_chain(
    node: dict, skip_node: Optional[dict]
) -> tuple[dict, list[dict]]:
    """Collapse single-child chains after removing skip_node; return node and its children."""
    while True:
        children = node.get("children", [])
        if skip_node is not None:
            children = [c for c in children if c is not skip_node]
        if len(children) == 1:
            node = children[0]
        else:
            return node, children


def _map_leaf_pairs(
    source_node, target_node, v_source, v_target
) -> Iterator[tuple[dict, dict]]:
    """Map source leaves to target leaves based on structural correspondence."""
    source_node, source_children = _collapse_single_child_chain(source_node, v_source)
    target_node, target_children = _collapse_single_child_chain(target_node, v_target)

    if source_node is v_source or target_node is v_target:
        return

    if not source_children and not target_children:
        assert source_node["type"] == target_node["type"]
        yield source_node, target_node
        return

    if len(source_children) != len(target_children):
        raise ValueError(
            f"Structural mismatch: source has {len(source_children)} children, "
            f"target has {len(target_children)} children."
        )

    raw_type = _normalize_node_type(source_node.get("type", "unknown"))
    if raw_type == "layer":
        ordered_source = _order_layer_children(source_children, exclude_node=v_source)
        ordered_target = _order_layer_children(target_children, exclude_node=v_target)
    else:
        ordered_source = list(source_children)
        ordered_target = list(target_children)

    for s_child, t_child in zip(ordered_source, ordered_target):
        yield from _map_leaf_pairs(s_child, t_child, v_source, v_target)


def replace_subtree(t_source, t_target, v_source_path=None, v_target_path=None):
    """
    Replace a target subtree with a source subtree structure, while preserving
    image_path mapping from the target subtree.

    Args:
        t_source (dict): The source subtree providing the new structure.
        t_target (dict): The target subtree to be replaced.
        v_source_path (str|None): The optional source leaf path
            that represents a changed or added element.
        v_target_path (str|None): The optional target leaf path
            that corresponds to v_source_path when both are provided. If both are provided,
            the image_path for v_source_path will be taken from v_target_path.

    Returns:
        dict: A new tree with the target subtree replaced by the source subtree structure,
              and image paths preserved from the target subtree where applicable.
    """
    new_tree = copy.deepcopy(t_source)

    # print("matching source:", get_structure_signature(t_source))
    # print("matching target:", get_structure_signature(t_target))

    v_source = get_node_by_path(new_tree, v_source_path)
    v_target = get_node_by_path(t_target, v_target_path)

    pairs = list(_map_leaf_pairs(new_tree, t_target, v_source, v_target))

    if v_source is not None:
        pairs.append((v_source, v_target or {}))
        if v_target is not None:
            assert v_source.get("type") == v_target.get("type")

    assert pairs

    # print("mapped pairs:")
    # for s, t in pairs:
    #     source_sig = get_structure_signature(s)
    #     target_sig = get_structure_signature(t)
    #     print(f"  source: {source_sig} <-> target: {target_sig}")
    # print()

    for source_leaf, target_leaf in pairs:
        source_leaf["image_path"] = target_leaf.get("image_path")

    src_bbox = _extract_bbox_list(new_tree)
    dst_bbox = _extract_bbox_list(t_target)
    if src_bbox and dst_bbox:
        _apply_bbox_transform(new_tree, src_bbox, dst_bbox)

    return new_tree


def attach_final_bboxes_to_tree(
    result_node, tree_node, is_root=True, parent_abs_x=0, parent_abs_y=0
):
    """
    Recursively compute absolute bboxes from result's final_bbox and attach them to tree nodes.

    In the result tree:
    - Root node's final_bbox is in absolute coordinates
    - Children's final_bbox are relative to their parent container

    This function converts all bboxes to absolute coordinates and adds them as 'final_bbox' to tree nodes.

    Args:
        result_node: Result node dict containing 'final_bbox' (from optimizer output).
        tree_node: Tree node dict to write the absolute 'final_bbox' into.
        is_root: Whether this is the root node (absolute coords) or child (relative coords).
        parent_abs_x: Absolute X of the parent node.
        parent_abs_y: Absolute Y of the parent node.
    """
    bbox = result_node.get("final_bbox")
    if bbox is None:
        return

    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        x, y, w, h = bbox
    elif isinstance(bbox, dict):
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("width", bbox.get("w", 0))
        h = bbox.get("height", bbox.get("h", 0))
    else:
        return

    if is_root:
        x_abs = int(float(x))
        y_abs = int(float(y))
    else:
        x_abs = parent_abs_x + int(float(x))
        y_abs = parent_abs_y + int(float(y))

    w_int = max(1, int(float(w)))
    h_int = max(1, int(float(h)))

    tree_node["final_bbox"] = {"x": x_abs, "y": y_abs, "width": w_int, "height": h_int}

    r_children = result_node.get("children", [])
    t_children = tree_node.get("children", [])

    for rc, tc in zip(r_children, t_children):
        attach_final_bboxes_to_tree(
            rc, tc, is_root=False, parent_abs_x=x_abs, parent_abs_y=y_abs
        )
