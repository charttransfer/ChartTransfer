"""
Core node definitions for the layout system.

Layout system based on template.ebnf supporting:
- Flow Layout (ROW/COLUMN)
- Non-Flow Layout (Z-layer)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


# ============= Enum Definitions =============

class NodeType(Enum):
    """Node type."""
    GROUP = "GROUP"
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    CHART = "CHART"
    SHAPE = "SHAPE"


class LayoutType(Enum):
    """Layout type."""
    FLOW = "FLOW"
    NON_FLOW = "NON_FLOW"


class FlowDirection(Enum):
    """Flow layout direction."""
    ROW = "ROW"           # Horizontal arrangement
    COLUMN = "COLUMN"     # Vertical arrangement
    # Future extensions:
    # GRID = "GRID"
    # CIRCULAR = "CIRCULAR"
    # IRREGULAR = "IRREGULAR"


class MainAlignment(Enum):
    """Main axis alignment."""
    START = "START"
    CENTER = "CENTER"
    END = "END"


class CrossAlignment(Enum):
    """Cross axis alignment."""
    START = "START"
    CENTER = "CENTER"
    END = "END"
    STRETCH = "STRETCH"


class PositionAlign(Enum):
    """Non-flow layout position alignment."""
    TOP_LEFT = "top-left"
    TOP_CENTER = "top-center"
    TOP_RIGHT = "top-right"
    CENTER_LEFT = "left"
    CENTER = "center"
    CENTER_RIGHT = "right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_CENTER = "bottom-center"
    BOTTOM_RIGHT = "bottom-right"


class Alignment(Enum):
    """Alignment (for each node's own alignment)."""
    START = "START"
    CENTER = "CENTER"
    END = "END"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    # Special values (for layer child nodes)
    BACKGROUND = "BACKGROUND"
    TOP_LEFT = "TOP_LEFT"
    TOP_CENTER = "TOP_CENTER"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_CENTER = "BOTTOM_CENTER"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"


# ============= Data Class Definitions =============

@dataclass
class BoundingBox:
    """Bounding box (x, y, width, height)."""
    x: float
    y: float
    width: float
    height: float
    
    @property
    def left(self) -> float:
        return self.x
    
    @property
    def right(self) -> float:
        return self.x + self.width
    
    @property
    def top(self) -> float:
        return self.y
    
    @property
    def bottom(self) -> float:
        return self.y + self.height
    
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass
class Padding:
    """Padding."""
    top: float = 0
    right: float = 0
    bottom: float = 0
    left: float = 0
    
    @classmethod
    def uniform(cls, value: float) -> 'Padding':
        """Create uniform padding."""
        return cls(value, value, value, value)
    
    @property
    def horizontal(self) -> float:
        """Total horizontal padding."""
        return self.left + self.right
    
    @property
    def vertical(self) -> float:
        """Total vertical padding."""
        return self.top + self.bottom


@dataclass
class FlowAlignment:
    """Flow layout alignment configuration."""
    main: MainAlignment = MainAlignment.START
    cross: CrossAlignment = CrossAlignment.START


# ============= Abstract Base Class =============

class Node(ABC):
    """Abstract base class for nodes."""
    
    def __init__(self, node_id: str, node_type: NodeType, alignment: Optional[str] = None, parent: Optional['Node'] = None):
        self.id = node_id
        self.type = node_type
        self.alignment = alignment  # Each node's own alignment
        self.bbox: Optional[BoundingBox] = None  # Bounding box after layout computation
        self.parent: Optional['Node'] = parent  # Parent node reference
        
    @abstractmethod
    def compute_intrinsic_size(self) -> Tuple[float, float]:
        """
        Compute intrinsic size (width, height) of the node.
        
        For Leaf Nodes: returns the actual content dimensions.
        For Non-Leaf Nodes: computed based on children and layout rules.
        """
        pass
    
    @abstractmethod
    def layout(self, x: float, y: float, available_width: Optional[float] = None, 
               available_height: Optional[float] = None) -> BoundingBox:
        """
        Perform layout computation.
        
        Args:
            x: Starting x coordinate
            y: Starting y coordinate
            available_width: Available width (optional)
            available_height: Available height (optional)
            
        Returns:
            Computed bounding box
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        pass


# ============= Leaf Node =============

class LeafNode(Node):
    """Leaf node.
    
    Represents a concrete visual element with fixed dimensions and an optional mask.
    """
    
    def __init__(self, node_id: str, node_type: NodeType, 
                 width: float, height: float, 
                 mask: Optional[np.ndarray] = None,
                 metadata: Optional[dict] = None,
                 alignment: Optional[str] = None,
                 parent: Optional[Node] = None):
        super().__init__(node_id, node_type, alignment, parent)
        self.width = width
        self.height = height
        self.mask = mask  # Optional binary mask for irregular shapes
        self.metadata = metadata or {}  # Stores additional metadata (e.g., content, role, src)
        
    def compute_intrinsic_size(self) -> Tuple[float, float]:
        """Intrinsic size of a leaf node is simply its width and height."""
        return (self.width, self.height)
    
    def layout(self, x: float, y: float, available_width: Optional[float] = None,
               available_height: Optional[float] = None) -> BoundingBox:
        """Leaf node layout simply places it at the specified position."""
        self.bbox = BoundingBox(x, y, self.width, self.height)
        return self.bbox
    
    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "type": self.type.value,
            "bbox": {
                "x": self.bbox.x if self.bbox else 0,
                "y": self.bbox.y if self.bbox else 0,
                "width": self.width,
                "height": self.height
            }
        }
        # Add metadata
        if self.metadata:
            result.update(self.metadata)
        return result


# ============= Non-Leaf Node =============

class GroupNode(Node):
    """Group node (non-leaf node).
    
    Contains multiple child nodes and arranges them according to the layout type.
    """
    
    def __init__(self, node_id: str, layout_type: LayoutType, 
                 children: List[Node], padding: Optional[Padding] = None,
                 alignment: Optional[str] = None,
                 parent: Optional[Node] = None):
        super().__init__(node_id, NodeType.GROUP, alignment, parent)
        self.layout_type = layout_type
        self.children = children
        # Set parent reference for each child node
        for child in self.children:
            child.parent = self
        self.padding = padding or Padding()
        self.mask: Optional[np.ndarray] = None  # Merged from child node masks
        
        # Layout-specific attributes (set by subclasses)
        self.layout_attrs = {}
        
    def compute_intrinsic_size(self) -> Tuple[float, float]:
        """
        Compute intrinsic size based on children and layout rules.
        Implemented by concrete layout subclasses.
        """
        raise NotImplementedError("Subclass must implement compute_intrinsic_size")
    
    def layout(self, x: float, y: float, available_width: Optional[float] = None,
               available_height: Optional[float] = None) -> BoundingBox:
        """
        Perform layout for the group node.
        Specific layout algorithms are implemented by subclasses.
        """
        raise NotImplementedError("Subclass must implement layout")
    
    def _compute_mask_from_children(self):
        """
        Compute the parent node's mask from all child node masks.
        
        Transforms each child's mask into the parent coordinate system based on
        its bounding box position, then merges all masks using an OR operation.
        """
        if not self.bbox or not self.children:
            self.mask = None
            return
        
        # Collect all child nodes that have masks
        children_with_mask = [
            child for child in self.children 
            if child.mask is not None and child.bbox is not None
        ]
        
        if not children_with_mask:
            self.mask = None
            return
        
        # Create parent node mask (all zeros)
        parent_width = int(self.bbox.width)
        parent_height = int(self.bbox.height)
        parent_mask = np.zeros((parent_height, parent_width), dtype=np.uint8)
        
        # Transform each child's mask into parent coordinate system and merge
        for child in children_with_mask:
            child_mask = child.mask
            child_bbox = child.bbox
            
            # Compute child position in parent coordinate system (relative to parent top-left)
            child_x_in_parent = int(child_bbox.x - self.bbox.x)
            child_y_in_parent = int(child_bbox.y - self.bbox.y)
            child_width = int(child_bbox.width)
            child_height = int(child_bbox.height)
            
            # Ensure child mask dimensions match bbox
            if child_mask.shape != (child_height, child_width):
                # If dimensions don't match, resize mask (nearest-neighbor interpolation)
                mask_h, mask_w = child_mask.shape
                
                # Compute source coordinates
                y_coords = np.clip(
                    (np.arange(child_height) * mask_h / child_height).astype(int),
                    0, mask_h - 1
                )
                x_coords = np.clip(
                    (np.arange(child_width) * mask_w / child_width).astype(int),
                    0, mask_w - 1
                )
                
                # Perform nearest-neighbor interpolation using NumPy advanced indexing
                y_indices, x_indices = np.meshgrid(y_coords, x_coords, indexing='ij')
                child_mask = child_mask[y_indices, x_indices]
            
            # Compute position range within parent mask
            y_start = max(0, child_y_in_parent)
            y_end = min(parent_height, child_y_in_parent + child_height)
            x_start = max(0, child_x_in_parent)
            x_end = min(parent_width, child_x_in_parent + child_width)
            
            # Compute corresponding range within child mask
            child_y_start = max(0, -child_y_in_parent)
            child_y_end = child_y_start + (y_end - y_start)
            child_x_start = max(0, -child_x_in_parent)
            child_x_end = child_x_start + (x_end - x_start)
            
            # Copy child mask to corresponding position in parent mask (OR operation)
            if (y_end > y_start and x_end > x_start and 
                child_y_end > child_y_start and child_x_end > child_x_start):
                parent_mask[y_start:y_end, x_start:x_end] = np.maximum(
                    parent_mask[y_start:y_end, x_start:x_end],
                    child_mask[child_y_start:child_y_end, child_x_start:child_x_end]
                )
        
        self.mask = parent_mask
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "layout": self.layout_type.value,
            "layoutAttrs": self.layout_attrs,
            "children": [child.to_dict() for child in self.children],
            "bbox": {
                "x": self.bbox.x if self.bbox else 0,
                "y": self.bbox.y if self.bbox else 0,
                "width": self.bbox.width if self.bbox else 0,
                "height": self.bbox.height if self.bbox else 0
            }
        }

