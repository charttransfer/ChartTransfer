"""Save hierarchical layout optimization results."""

from PIL import Image, ImageDraw
from typing import Dict, Any, Tuple, Optional
import os


def _parse_bg_color(bg_color) -> Tuple[int, int, int]:
    if bg_color is None:
        return (255, 255, 255)
    if isinstance(bg_color, (list, tuple)) and len(bg_color) >= 3:
        return (int(bg_color[0]), int(bg_color[1]), int(bg_color[2]))
    if isinstance(bg_color, str) and bg_color.startswith("#"):
        return (int(bg_color[1:3], 16), int(bg_color[3:5], 16), int(bg_color[5:7], 16))
    return (255, 255, 255)


def save_hierarchical_result(result: Dict[str, Any], save_path: str = "hierarchical_result.png",
                            base_dir: str = ".", bg_color=None, debug: bool = False,
                            draw_bbox: bool = True) -> None:
    """Save hierarchical optimization result as an image.
    
    Args:
        result: Result dictionary from HierarchicalOptimizer.optimize_tree()
        save_path: Path to save the final image
        base_dir: Base directory for resolving image paths
    """
    # Get root bbox
    root_bbox = result.get("final_bbox")
    if root_bbox is None:
        if debug:
            print("Warning: No final_bbox found in result")
        return
    
    if isinstance(root_bbox, (tuple, list)) and len(root_bbox) == 4:
        x, y, w, h = root_bbox
    elif isinstance(root_bbox, dict):
        x = root_bbox.get("x", 0)
        y = root_bbox.get("y", 0)
        w = root_bbox.get("width", root_bbox.get("w", 1000))
        h = root_bbox.get("height", root_bbox.get("h", 1000))
    else:
        if debug:
            print(f"Warning: Invalid root_bbox format: {root_bbox}")
        return
    
    root_x = int(float(x))
    root_y = int(float(y))
    Wc = int(float(w))
    Hc = int(float(h))
    
    # First pass: calculate the actual bounding box needed for all nodes
    # Root node's final_bbox is absolute, so pass 0,0 as offset (will use final_bbox directly)
    min_x, min_y, max_x, max_y = _calculate_bounds(result, offset_x=0, offset_y=0, is_root=True)
    
    # Add some padding to ensure nothing is cut off
    padding = 10
    canvas_width = max_x - min_x + padding * 2
    canvas_height = max_y - min_y + padding * 2
    canvas_offset_x = min_x - padding
    canvas_offset_y = min_y - padding
    
    bg_rgb = _parse_bg_color(bg_color)
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (*bg_rgb, 255))
    
    if debug:
        print(f"\n[Saving hierarchical result]")
        print(f"  Root bbox: ({root_x}, {root_y}, {Wc}, {Hc})")
        print(f"  Content bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")
        print(f"  Canvas size: {canvas_width}x{canvas_height} (offset: {canvas_offset_x}, {canvas_offset_y})")
    
    # Second pass: recursively composite all nodes
    # Root node's final_bbox is absolute, so we need to adjust for canvas offset
    # For root node, we pass a flag indicating it's the root
    _composite_node_to_canvas(result, canvas, base_dir,
                              offset_x=root_x - canvas_offset_x,
                              offset_y=root_y - canvas_offset_y,
                              is_root=True, debug=debug, draw_bbox=draw_bbox)
    
    canvas_rgb = Image.new("RGB", canvas.size, bg_rgb)
    canvas_rgb.paste(canvas, mask=canvas.split()[3])
    canvas_rgb.save(save_path, "PNG")
    if debug:
        print(f"Hierarchical layout result saved to: {save_path}\n")
    
    no_box_path = save_path.rsplit(".", 1)[0] + "_no_box.png"
    canvas_no_box = Image.new("RGBA", (canvas_width, canvas_height), (*bg_rgb, 255))
    _composite_node_to_canvas(result, canvas_no_box, base_dir,
                              offset_x=root_x - canvas_offset_x,
                              offset_y=root_y - canvas_offset_y,
                              is_root=True, debug=debug, draw_bbox=False)
    canvas_no_box_rgb = Image.new("RGB", canvas_no_box.size, bg_rgb)
    canvas_no_box_rgb.paste(canvas_no_box, mask=canvas_no_box.split()[3])
    canvas_no_box_rgb.save(no_box_path, "PNG")
    if debug:
        print(f"Hierarchical layout result (no box) saved to: {no_box_path}\n")


def _draw_bbox(canvas: Image.Image, x: int, y: int, w: int, h: int, node_type: str) -> None:
    """Draw bounding box on canvas.
    
    Args:
        canvas: PIL Image canvas to draw on
        x: X coordinate (absolute)
        y: Y coordinate (absolute)
        w: Width
        h: Height
        node_type: Type of node (for color selection)
    """
    # Clip coordinates to canvas bounds
    x_clip = max(0, min(x, canvas.width - 1))
    y_clip = max(0, min(y, canvas.height - 1))
    x_end = min(x + w, canvas.width)
    y_end = min(y + h, canvas.height)
    
    if x_end <= x_clip or y_end <= y_clip:
        return
    
    # Choose color based on node type
    color_map = {
        "column": (255, 0, 0, 255),      # Red for column
        "row": (0, 255, 0, 255),         # Green for row
        "layer": (0, 0, 255, 255),       # Blue for layer
        "chart": (255, 165, 0, 255),     # Orange for chart
        "image": (255, 0, 255, 255),     # Magenta for image
        "text": (0, 255, 255, 255),      # Cyan for text
    }
    color = color_map.get(node_type, (128, 128, 128, 255))  # Gray for unknown types
    
    # Draw rectangle
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([x_clip, y_clip, x_end - 1, y_end - 1], outline=color, width=2)


def _calculate_bounds(node_result: Dict[str, Any], offset_x: int = 0, offset_y: int = 0, 
                      is_root: bool = False) -> Tuple[int, int, int, int]:
    """Calculate the bounding box of all nodes in the tree.
    
    Args:
        node_result: Node result dictionary
        offset_x: X offset accumulated from parent containers (for relative coordinates)
        offset_y: Y offset accumulated from parent containers (for relative coordinates)
        is_root: Whether this is the root node (root's final_bbox is absolute, others are relative)
    
    Returns:
        Tuple of (min_x, min_y, max_x, max_y) in absolute coordinates
    """
    # Get node bbox
    bbox = node_result.get("final_bbox")
    if bbox is None:
        return (0, 0, 0, 0)
    
    # Handle different bbox formats
    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        x, y, w, h = bbox
    elif isinstance(bbox, dict):
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("width", bbox.get("w", 0))
        h = bbox.get("height", bbox.get("h", 0))
    else:
        return (0, 0, 0, 0)
    
    # Root node's bbox is absolute, other nodes' bboxes are relative to parent container
    if is_root:
        # Root bbox is already absolute, use it directly
        x_abs = int(float(x))
        y_abs = int(float(y))
    else:
        # Convert relative coordinates to absolute by adding parent offset
        x_abs = offset_x + int(float(x))
        y_abs = offset_y + int(float(y))
    
    w_int = max(1, int(float(w)))
    h_int = max(1, int(float(h)))
    
    # Initialize bounds with this node's bounds
    min_x = x_abs
    min_y = y_abs
    max_x = x_abs + w_int
    max_y = y_abs + h_int
    
    # Recursively process children
    # Children are not root nodes, so pass is_root=False
    children = node_result.get("children", [])
    for child_result in children:
        child_min_x, child_min_y, child_max_x, child_max_y = _calculate_bounds(
            child_result, offset_x=x_abs, offset_y=y_abs, is_root=False
        )
        if child_min_x < min_x:
            min_x = child_min_x
        if child_min_y < min_y:
            min_y = child_min_y
        if child_max_x > max_x:
            max_x = child_max_x
        if child_max_y > max_y:
            max_y = child_max_y
    
    return (min_x, min_y, max_x, max_y)


def _composite_node_to_canvas(node_result: Dict[str, Any], canvas: Image.Image,
                              base_dir: str, offset_x: int = 0, offset_y: int = 0,
                              is_root: bool = False, debug: bool = False, draw_bbox: bool = True) -> None:
    """Recursively composite node results onto canvas.
    
    Args:
        node_result: Node result dictionary
        canvas: PIL Image canvas to composite onto
        base_dir: Base directory for resolving image paths
        offset_x: X offset accumulated from parent containers (for canvas offset adjustment)
        offset_y: Y offset accumulated from parent containers (for canvas offset adjustment)
        is_root: Whether this is the root node (root's final_bbox is absolute, others are relative)
    """
    # Get node bbox
    bbox = node_result.get("final_bbox")
    if bbox is None:
        return
    
    # Handle different bbox formats
    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        x, y, w, h = bbox
    elif isinstance(bbox, dict):
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("width", bbox.get("w", 0))
        h = bbox.get("height", bbox.get("h", 0))
    else:
        return
    
    # Root node's bbox is absolute, other nodes' bboxes are relative to parent container
    if is_root:
        # Root bbox is already absolute, offset_x/offset_y here are canvas offsets (root_x - canvas_offset_x)
        # For root: final_bbox x is root_x (absolute), offset_x = root_x - canvas_offset_x
        # We want canvas-relative: root_x - canvas_offset_x = offset_x
        # So we use offset_x directly (it's already the canvas-relative position)
        x_abs = offset_x
        y_abs = offset_y
    else:
        # Child node bbox is relative to parent container
        # Convert to absolute coordinates by adding parent offset
        x_abs = offset_x + int(float(x))
        y_abs = offset_y + int(float(y))
    w_int = max(1, int(float(w)))
    h_int = max(1, int(float(h)))
    
    node_type = node_result.get("type", "unknown")
    metadata = node_result.get("metadata", {})

    image_path = metadata.get("image_path") or metadata.get("full_path") or node_result.get("image_path")
    
    # Try to resolve path
    if image_path:
        if os.path.isabs(image_path):
            full_path = image_path
        else:
            full_path = os.path.join(base_dir, image_path)
        
        if os.path.exists(full_path):
            img = Image.open(full_path).convert("RGBA")
            img_w, img_h = img.size
            scale_w = w_int / img_w if img_w > 0 else 1.0
            scale_h = h_int / img_h if img_h > 0 else 1.0
            scale = min(scale_w, scale_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            x_offset = (w_int - new_w) // 2
            y_offset = (h_int - new_h) // 2
            x_final = x_abs + x_offset
            y_final = y_abs + y_offset

            x_clip = max(0, min(x_final, canvas.width - 1))
            y_clip = max(0, min(y_final, canvas.height - 1))
            x_end = min(x_clip + new_w, canvas.width)
            y_end = min(y_clip + new_h, canvas.height)
            w_fit = x_end - x_clip
            h_fit = y_end - y_clip

            if w_fit > 0 and h_fit > 0:
                if w_fit < new_w or h_fit < new_h:
                    img_resized = img_resized.crop((0, 0, w_fit, h_fit))
                canvas.paste(img_resized, (x_clip, y_clip), img_resized)
                if debug:
                    coord_type = "absolute" if is_root else "relative"
                    print(f"  [Save] Node '{node_type}': image={os.path.basename(image_path)}, "
                          f"bbox=({int(float(x))}, {int(float(y))}, {w_int}, {h_int}) [{coord_type}], "
                          f"placed_at=({x_final}, {y_final}) [canvas-relative]")
        else:
            if debug:
                coord_type = "absolute" if is_root else "relative"
                print(f"  [Save] Node '{node_type}': bbox=({int(float(x))}, {int(float(y))}, {w_int}, {h_int}) [{coord_type}], "
                      f"image_path={image_path} (not found)")
    else:
        if debug:
            coord_type = "absolute" if is_root else "relative"
            print(f"  [Save] Node '{node_type}': bbox=({int(float(x))}, {int(float(y))}, {w_int}, {h_int}) [{coord_type}], no image")
    
    if draw_bbox:
        _draw_bbox(canvas, x_abs, y_abs, w_int, h_int, node_type)
    
    children = node_result.get("children", [])
    for child_result in children:
        _composite_node_to_canvas(child_result, canvas, base_dir, offset_x=x_abs, offset_y=y_abs,
                                  is_root=False, debug=debug, draw_bbox=draw_bbox)

