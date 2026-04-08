"""
AutoTitleNew - New title generation module based on Title Structure

Learns style and layout from extracted title structures to generate new titles
"""

from typing import Optional, Dict
from .structure_loader import StructureLoader, TitleStructure
from .text_adapter import TextAdapter
from .style_migrator import StyleMigrator, StyledSegment, LayoutConfig
from .svg_renderer import SVGRenderer


class TitleGenerator:
    """Main title generator class"""
    
    def __init__(
        self,
        structure_dir: str = "extracted_title_structures",
        width: int = 800,
        height: int = 200
    ):
        """
        Initialize the title generator
        
        Args:
            structure_dir: directory containing title structure files
            width: output SVG width
            height: output SVG height
        """
        self.structure_loader = StructureLoader(structure_dir)
        self.text_adapter = TextAdapter()
        self.style_migrator = StyleMigrator()
        self.svg_renderer = SVGRenderer(width, height)
    
    def _split_text_into_lines(self, text: str, line_count: int) -> list:
        """
        Split text into multiple lines by average length
        
        Args:
            text: text to split
            line_count: target number of lines
        
        Returns:
            list of split lines
        """
        if line_count <= 1:
            return [text]
        
        # Calculate average characters per line
        total_length = len(text)
        avg_chars_per_line = total_length // line_count
        
        lines = []
        words = text.split()
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + (1 if current_line else 0)  # +1 for space
            
            # If adding this word exceeds the average length and it's not the first word, start a new line
            if current_length + word_length > avg_chars_per_line and current_line and len(lines) < line_count - 1:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += word_length
        
        # Add the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def generate(
        self,
        new_title: str,
        example_name: str,
        color_override: Optional[Dict[str, str]] = None,
        adapt_length: bool = True,
        output_path: Optional[str] = None,
        text_constraints: Optional[Dict] = None
    ) -> str:
        """
        Generate a new title SVG
        
        Args:
            new_title: new title text
            example_name: reference example name (without _structure.json suffix)
            color_override: optional color override, format: {"TITLE_PRIMARY": "#FF0000", ...}
            adapt_length: whether to use LLM for text length adaptation
            output_path: optional output file path; if provided, also saves to file
            text_constraints: optional text constraints containing bbox (x, y, width, height), etc.
        
        Returns:
            SVG string
        """
        # print(f"\n{'='*70}")
        # print(f"Generating new title")
        # print(f"{'='*70}")
        # print(f"New title text: {new_title}")
        # print(f"Reference example: {example_name[:60]}...")
        # print(f"Text length adaptation: {'enabled' if adapt_length else 'disabled'}")
        # print()
        
        # If complete text_constraints are provided, use the constraint-based generation method
        if text_constraints and all(key in text_constraints for key in ['width', 'height', 'font_family', 'font_size', 'font_weight', 'color']):
            print("[Info] Complete text_constraints detected, using constraint-based generation method")
            return self.generate_from_constraints(
                new_title,
                text_constraints,
                adapt_length=adapt_length,
                output_path=output_path
            )
        
        # 1. Load reference structure
        print("[1/5] Loading reference title structure...")
        structure = self.structure_loader.load(example_name)
        print(f"  ✓ Loaded successfully")
        print(f"    - Layout direction: {structure.root.direction}")
        print(f"    - Alignment: {structure.visual_properties.overall_alignment}")
        print(f"    - Segment count: {len(structure.root.children)}")
        
        # 2. Text length adaptation
        print(f"\n[2/5] Text length adaptation...")
        if adapt_length:
            adapted_texts = self.text_adapter.adapt_text(new_title, structure)
            print(f"  ✓ Text adaptation complete")
            for item in adapted_texts:
                print(f"    - {item['role']}: {item['text'][:50]}...")
        else:
            adapted_texts = self._fallback_split(new_title, structure)
            print(f"  ✓ Using simple split")
        
        # 3. Style migration
        print(f"\n[3/5] Style migration...")
        styled_segments = []
        
        # Match text with reference segments
        matches = self.style_migrator.match_segments(
            adapted_texts,
            structure.root.children
        )
        
        for text, ref_segment in matches:
            # Check for color override
            override_color = None
            if color_override and ref_segment.role in color_override:
                override_color = color_override[ref_segment.role]
            
            styled = self.style_migrator.migrate_segment_style(
                text,
                ref_segment,
                color_override=override_color
            )
            styled_segments.append(styled)
            
            print(f"  ✓ {styled.role}:")
            print(f"    Text: {styled.text[:40]}...")
            print(f"    Style: {styled.font_size}px, weight={styled.font_weight}, color={styled.color}")
        
        # 4. Get layout and background color
        print(f"\n[4/5] Configuring layout...")
        layout = self.style_migrator.migrate_layout(structure)
        bg_color = self.style_migrator.get_background_color(structure)
        print(f"  ✓ Layout: {layout.direction}, align: {layout.overall_align}")
        print(f"  ✓ Background color: {bg_color}")
        
        # 5. Render SVG
        print(f"\n[5/5] Rendering SVG...")
        svg = self.svg_renderer.render(styled_segments, layout, bg_color)
        print(f"  ✓ Rendering complete")
        
        # Save to file (if specified)
        if output_path:
            self.svg_renderer.render_to_file(
                styled_segments, layout, output_path, bg_color
            )
            print(f"  ✓ Saved to: {output_path}")
        
        print(f"\n{'='*70}")
        print(f"✅ Title generation complete")
        print(f"{'='*70}\n")
        
        return svg
    
    def _fallback_split(self, text: str, structure: TitleStructure) -> list:
        """Fallback simple text splitting method"""
        segments = []
        
        # Count each role
        for seg in structure.root.children:
            if seg.role == "TITLE_PRIMARY":
                segments.append({
                    "role": "TITLE_PRIMARY",
                    "text": text
                })
            elif seg.role == "SUBTITLE":
                segments.append({
                    "role": "SUBTITLE",
                    "text": ""
                })
        
        return segments if segments else [{"role": "TITLE_PRIMARY", "text": text}]
    
    def generate_from_constraints(
        self,
        text_content: str,
        text_constraints: Dict,
        adapt_length: bool = True,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate text directly from text constraints (without reference example)
        
        Enhanced version: supports richer style attributes and inline multi-style
        
        Args:
            text_content: text content to generate
            text_constraints: dictionary with the following fields:
                - width: text area width (pixels)
                - height: text area height (pixels)
                - line_count: number of lines
                - font_family: font family
                - font_size: font size ("large"/"medium"/"small")
                - font_size_px: specific pixel size (takes priority over font_size)
                - font_weight: font weight ("bold"/"normal"/"light")
                - font_style: italic ("normal"/"italic")
                - color: color (hex)
                - letter_spacing: letter spacing (em)
                - text_transform: case transformation ("none"/"uppercase"/"lowercase")
                - background: background config {color, padding, border_radius}
                - shadow: shadow config {blur, offset_x, offset_y, color}
                - alignment: alignment ("left"/"center"/"right")
                - has_inline_styles: whether it has inline multi-style
                - inline_segments: inline multi-style config
            adapt_length: whether to perform length adaptation
            output_path: optional output file path
        
        Returns:
            SVG string
        """
        # 1. Length Adaptation
        print("[1/4] Text length adaptation...")
        if adapt_length:
            adapted_text = self.text_adapter.adapt_text_to_constraints(
                text_content,
                text_constraints.get('width', 800),
                text_constraints.get('height', 200),
                text_constraints.get('line_count', 1),
                text_constraints.get('font_size', 'medium')
            )
        else:
            adapted_text = text_content
        
        # 2. Create StyledSegment (with enhanced style support)
        print("[2/4] Applying styles...")
        
        # Determine font size (font_size_px takes priority)
        if text_constraints.get('font_size_px'):
            font_size_px = text_constraints['font_size_px']
        else:
            font_size_str = text_constraints.get('font_size', 'medium')
            font_size_map = {"large": 48, "medium": 32, "small": 28}
            font_size_px = font_size_map.get(font_size_str, 32)
        
        font_weight_raw = text_constraints.get('font_weight', 'normal')
        try:
            font_weight = int(font_weight_raw)
        except (ValueError, TypeError):
            font_weight_map = {"bold": 700, "semibold": 600, "normal": 400, "light": 300}
            font_weight = font_weight_map.get(font_weight_raw, 400)
        
        font_family = text_constraints.get('font_family', 'Arial, sans-serif')
        color = text_constraints.get('color', '#000000')
        
        # Get enhanced style attributes
        font_style = text_constraints.get('font_style', 'normal')
        letter_spacing = text_constraints.get('letter_spacing', 0.0)
        text_transform = text_constraints.get('text_transform', 'none')
        
        # Handle background
        background = text_constraints.get('background', {})
        background_color = background.get('color') if background else None
        background_padding = background.get('padding', 0) if background else 0
        background_radius = background.get('border_radius', 0) if background else 0
        
        # Handle shadow
        shadow = text_constraints.get('shadow', {})
        shadow_blur = shadow.get('blur', 0) if shadow else 0
        shadow_offset_x = shadow.get('offset_x', 0) if shadow else 0
        shadow_offset_y = shadow.get('offset_y', 0) if shadow else 0
        shadow_color = shadow.get('color') if shadow else None
        
        # Handle possible escaped newline characters
        if '\\n' in adapted_text and '\n' not in adapted_text:
            adapted_text = adapted_text.replace('\\n', '\n')
        
        # Check for inline multi-style
        if text_constraints.get('has_inline_styles') and text_constraints.get('inline_segments'):
            print("  ✓ Inline multi-style detected, using StyleMigrator to create segments")
            styled_segments = self.style_migrator.create_inline_segments(
                full_text=adapted_text,
                inline_specs=text_constraints['inline_segments'],
                role=text_constraints.get('role', 'TITLE_PRIMARY'),
                inline_group=0
            )
            print(f"  ✓ Created {len(styled_segments)} inline segments")
        else:
            # Normal mode: handle text line-breaking based on line_count
            target_line_count = text_constraints.get('line_count', 1)
            
            if target_line_count == 1:
                lines = [adapted_text.replace('\n', ' ')]
            elif '\n' in adapted_text:
                lines = adapted_text.split('\n')
            else:
                lines = self._split_text_into_lines(adapted_text, target_line_count)
            
            styled_segments = []
            for line in lines:
                if line.strip():
                    styled_segment = StyledSegment(
                        text=line.strip(),
                        font_size=font_size_px,
                        font_weight=font_weight,
                        opacity=1.0,
                        color=color,
                        role=text_constraints.get('role', 'TITLE_PRIMARY'),
                        font_family=font_family,
                        font_style=font_style,
                        letter_spacing=letter_spacing,
                        text_transform=text_transform,
                        background_color=background_color,
                        background_padding=background_padding,
                        background_radius=background_radius,
                        shadow_blur=shadow_blur,
                        shadow_offset=(shadow_offset_x, shadow_offset_y),
                        shadow_color=shadow_color
                    )
                    styled_segments.append(styled_segment)
        
        for idx, seg in enumerate(styled_segments):
            print(f"    Segment {idx+1}: {seg.text[:40]}... (inline_group={seg.inline_group}, color={seg.color})")
        
        # 3. Configure layout
        print("[3/4] Configuring layout...")
        alignment = text_constraints.get('alignment', 'left')
        align_map = {"left": "START", "center": "CENTER", "right": "END"}
        overall_align = align_map.get(alignment, "START")
        
        layout = LayoutConfig(
            direction="COLUMN",
            main_align="START",
            cross_align="START",
            spacing=6,
            overall_align=overall_align
        )
        
        # 4. Render SVG
        print("[4/4] Rendering SVG...")
        width = text_constraints.get('width', 800)
        height = text_constraints.get('height', 200)
        self.svg_renderer.initial_width = width
        self.svg_renderer.initial_height = height
        
        bg_color = "#FFFFFF"
        svg = self.svg_renderer.render(styled_segments, layout, bg_color)
        
        # Save to file (if specified)
        if output_path:
            self.svg_renderer.render_to_file(
                styled_segments, layout, output_path, bg_color
            )
        
        return svg
    
    def list_available_examples(self):
        """List all available examples"""
        return self.structure_loader.list_available_structures()


# Convenience function
def generate_title(
    new_title: str,
    example_name: str,
    **kwargs
) -> str:
    """
    Convenience function: generate a title
    
    Args:
        new_title: new title text
        example_name: reference example name
        **kwargs: additional arguments passed to TitleGenerator.generate()
    
    Returns:
        SVG string
    """
    generator = TitleGenerator()
    return generator.generate(new_title, example_name, **kwargs)

