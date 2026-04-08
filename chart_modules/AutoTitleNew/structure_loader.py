"""
Title Structure Loader

Load and parse title structure data from extracted_title_structures.
Only loads from pre-extracted cache files; does not contain extraction logic.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SegmentStyle:
    """Text segment style"""
    fontSize: str  # large/medium/small
    fontWeight: str  # bold/normal
    emphasis: str  # high/medium/low
    color: Optional[str] = None  # hex color
    fontFamily: Optional[str] = None  # font family name
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            fontSize=data.get('fontSize', 'medium'),
            fontWeight=data.get('fontWeight', 'normal'),
            emphasis=data.get('emphasis', 'medium'),
            color=data.get('color'),
            fontFamily=data.get('fontFamily')
        )


@dataclass
class Segment:
    """Text segment"""
    id: str
    type: str  # TEXT/GROUP/SHAPE
    content: str
    role: str  # TITLE_PRIMARY/SUBTITLE/etc
    style: SegmentStyle
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            id=data.get('id', ''),
            type=data.get('type', 'TEXT'),
            content=data.get('content', ''),
            role=data.get('role', ''),
            style=SegmentStyle.from_dict(data.get('style', {}))
        )


@dataclass
class Alignment:
    """Alignment configuration"""
    main: str  # START/CENTER/END
    cross: str  # START/CENTER/END
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            main=data.get('main', 'START'),
            cross=data.get('cross', 'START')
        )


@dataclass
class RootNode:
    """Root node"""
    direction: str  # COLUMN/ROW
    alignment: Alignment
    spacing: int  # pixels
    children: List[Segment]
    
    @classmethod
    def from_dict(cls, data: dict):
        children = []
        for child_data in data.get('children', []):
            if child_data.get('type') == 'TEXT':
                children.append(Segment.from_dict(child_data))
        
        return cls(
            direction=data.get('direction', 'COLUMN'),
            alignment=Alignment.from_dict(data.get('alignment', {})),
            spacing=data.get('spacing', 10),
            children=children
        )


@dataclass
class VisualProperties:
    """Visual properties"""
    overall_alignment: str  # LEFT/CENTER/RIGHT
    colors: Dict[str, str]  # structured colors
    color_scheme: str  # color description
    highlight_colors: List[str]
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            overall_alignment=data.get('overall_alignment', 'LEFT'),
            colors=data.get('colors', {}),
            color_scheme=data.get('color_scheme', ''),
            highlight_colors=data.get('highlight_colors', [])
        )


@dataclass
class TitleStructure:
    """Complete Title Structure"""
    root: RootNode
    visual_properties: VisualProperties
    metadata: dict
    
    @classmethod
    def from_dict(cls, data: dict):
        scene_graph = data.get('scene_graph', {})
        return cls(
            root=RootNode.from_dict(scene_graph.get('root', {})),
            visual_properties=VisualProperties.from_dict(
                scene_graph.get('visual_properties', {})
            ),
            metadata=scene_graph.get('metadata', {})
        )


class StructureLoader:
    """Title Structure loader (loads from cache only)"""
    
    def __init__(self, structure_dir: str = "extracted_title_structures"):
        self.structure_dir = structure_dir
    
    def load(self, example_name: str) -> TitleStructure:
        """
        Load the title structure for a specified example
        
        Args:
            example_name: example name (without suffix)
        
        Returns:
            TitleStructure object
        
        Raises:
            FileNotFoundError: if the structure file cannot be found
        """
        filepath = os.path.join(self.structure_dir, f"{example_name}_structure.json")
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return TitleStructure.from_dict(data)
        
        raise FileNotFoundError(
            f"Title structure not found: {filepath}\n"
            f"Please provide a pre-extracted structure file."
        )
    
    def list_available_structures(self) -> List[str]:
        """List all available title structures"""
        if not os.path.exists(self.structure_dir):
            return []
        
        structures = []
        for filename in os.listdir(self.structure_dir):
            if filename.endswith('_structure.json'):
                example_name = filename.replace('_structure.json', '')
                structures.append(example_name)
        
        return sorted(structures)
    
    def get_segment_colors(self, structure: TitleStructure) -> Dict[str, str]:
        """
        Extract the color mapping for each segment
        
        Returns:
            {segment_id: color_hex}
        """
        colors = {}
        for segment in structure.root.children:
            if segment.style.color:
                colors[segment.id] = segment.style.color
        return colors
    
    def get_background_color(self, structure: TitleStructure) -> str:
        """Get background color"""
        return structure.visual_properties.colors.get('background', '#FFFFFF')
