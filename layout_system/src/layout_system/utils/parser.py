"""JSON parser for layout tree structure."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from layout_system.utils.tree_manipulation import get_structure_signature


@dataclass
class LayoutNode:
    """Layout node data structure."""
    type: str
    bbox: Dict[str, float]
    constraints: Dict[str, Any] = field(default_factory=dict)
    children: List['LayoutNode'] = field(default_factory=list)
    image_path: Optional[str] = None
    content: Optional[str] = None
    alignment: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optimization results (filled after optimization)
    final_bbox: Optional[tuple] = None
    composite_mask: Optional[Any] = None
    composite_sdf: Optional[Any] = None


def parse_layout_tree(json_data: dict) -> LayoutNode:
    """Parse JSON layout tree structure.
    
    Args:
        json_data: JSON dictionary with layout tree structure
    
    Returns:
        Root LayoutNode
    """
    # Handle different JSON formats
    if "scene_tree" in json_data:
        tree_data = json_data["scene_tree"]
    else:
        tree_data = json_data
    print(f"[parse_layout_tree] tree_data: {get_structure_signature(tree_data)}")
    return _parse_node(tree_data)


def _parse_node(node_data: dict) -> LayoutNode:
    exclude_keys = {"type", "bbox", "constraints", "children", "image_path", "content", "alignment"}
    metadata = {k: v for k, v in node_data.items() if k not in exclude_keys and v is not None}
    node = LayoutNode(
        type=node_data.get("type", "unknown"),
        bbox=node_data.get("bbox", {}),
        constraints=node_data.get("constraints", {}),
        image_path=node_data.get("image_path"),
        content=node_data.get("content"),
        alignment=node_data.get("alignment"),
        metadata=metadata,
    )
    
    # Parse children recursively
    children_data = node_data.get("children", [])
    for child_data in children_data:
        child_node = _parse_node(child_data)
        node.children.append(child_node)
    
    return node

