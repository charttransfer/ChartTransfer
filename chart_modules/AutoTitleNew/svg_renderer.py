"""
SVG Renderer

Renders styled text segments into SVG

Enhanced version with support for:
- Inline multi-style (multiple colors/styles within the same line)
- Text backgrounds
- Shadow effects
- Letter spacing and text-transform
"""

import html
import random
from typing import List, Tuple, Dict
from .style_migrator import StyledSegment, LayoutConfig, InlineSegmentGroup


class SVGRenderer:
    """SVG renderer (enhanced version)"""
    
    def __init__(self, width: int = 800, height: int = 200):
        """
        Args:
            width: initial SVG width (dynamically adjusted based on content)
            height: initial SVG height (dynamically adjusted based on content)
        """
        self.initial_width = width
        self.initial_height = height
        self._filter_counter = 0
    
    def render(
        self,
        styled_segments: List[StyledSegment],
        layout: LayoutConfig,
        background_color: str = "#FFFFFF"
    ) -> str:
        """
        Render to SVG string (enhanced version)
        
        Supports:
        - Inline multi-style (multiple colors on the same line)
        - Text backgrounds
        - Shadow effects
        - Letter spacing and text-transform
        
        Args:
            styled_segments: list of styled text segments
            layout: layout configuration
            background_color: background color
        
        Returns:
            SVG string
        """
        actual_width = self._calculate_required_width(styled_segments)
        
        processed_segments = self._process_text_wrapping(styled_segments, layout, actual_width)
        
        grouped_segments = self._group_by_inline(processed_segments)
        
        print(f"  [DEBUG] Grouping result: {len(grouped_segments)} groups")
        for g_idx, group in enumerate(grouped_segments):
            print(f"    Group {g_idx}: {len(group)} segments - {[s.text[:20] for s in group]}")
        
        actual_height = self._calculate_required_height(processed_segments, layout)
        
        filters_svg = self._generate_filters(processed_segments)
        
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{actual_width}" height="{actual_height}" viewBox="0 0 {actual_width} {actual_height}">',
        ]
        
        
        if filters_svg:
            svg_parts.append('  <defs>')
            svg_parts.append(filters_svg)
            svg_parts.append('  </defs>')
        
        
        x = self._get_x_position(layout, actual_width)
        text_anchor = self._get_text_anchor(layout)
        y = self._get_initial_y(processed_segments, layout)
        
        
        for group_idx, group in enumerate(grouped_segments):
            if len(group) == 1:
                segment = group[0]
                svg_parts.extend(self._render_single_segment(segment, x, y, text_anchor, layout))
                
                lines = segment.text.split('\n') if '\n' in segment.text else [segment.text]
                for line_idx, line in enumerate(lines):
                    if not line.strip():
                        continue
                    if line_idx < len(lines) - 1:
                        descender = int(segment.font_size * 0.25)
                        ascender = int(segment.font_size * 0.75)
                        y += descender + 6 + ascender
                
                
                if group_idx < len(grouped_segments) - 1:
                    next_group = grouped_segments[group_idx + 1]
                    next_segment = next_group[0]
                    
                    if segment.role == "TITLE_PRIMARY" and next_segment.role == "SUBTITLE":
                        bbox_spacing = 12
                    else:
                        bbox_spacing = layout.spacing
                    
                    descender = int(segment.font_size * 0.25)
                    ascender = int(next_segment.font_size * 0.75)
                    baseline_spacing = descender + bbox_spacing + ascender
                    y += baseline_spacing
            else:
                svg_parts.extend(self._render_inline_group(group, x, y, text_anchor, actual_width))
                
                max_font_size = max(seg.font_size for seg in group)
                
                if group_idx < len(grouped_segments) - 1:
                    next_group = grouped_segments[group_idx + 1]
                    next_segment = next_group[0]
                    
                    bbox_spacing = layout.spacing
                    descender = int(max_font_size * 0.25)
                    ascender = int(next_segment.font_size * 0.75)
                    baseline_spacing = descender + bbox_spacing + ascender
                    y += baseline_spacing
        
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)
    
    def _group_by_inline(self, segments: List[StyledSegment]) -> List[List[StyledSegment]]:
        """
        Group segments by inline_group
        
        Segments with inline_group == -1 form their own group.
        Segments with the same inline_group >= 0 are merged into one group.
        
        Returns:
            list of grouped segments
        """
        result = []
        inline_groups: Dict[int, List[StyledSegment]] = {}
        
        for segment in segments:
            if segment.inline_group < 0:
                result.append([segment])
            else:
                if segment.inline_group not in inline_groups:
                    inline_groups[segment.inline_group] = []
                inline_groups[segment.inline_group].append(segment)
        
        for group_id in sorted(inline_groups.keys()):
            result.append(inline_groups[group_id])
        
        return result
    
    def _render_single_segment(
        self,
        segment: StyledSegment,
        x: int,
        y: int,
        text_anchor: str,
        layout: LayoutConfig
    ) -> List[str]:
        """
        Render a single segment (supports new style attributes)
        
        Returns:
            list of SVG element strings
        """
        svg_parts = []
        
        if not segment.text:
            return svg_parts
        
        lines = segment.text.split('\n') if '\n' in segment.text else [segment.text]
        current_y = y
        
        for line_idx, line in enumerate(lines):
            if not line.strip():
                continue
            
            
            display_text = self._apply_text_transform(line, segment.text_transform)
            escaped_text = self._escape_xml(display_text)
            
            
            text_width = self._estimate_text_width(display_text, segment.font_size, segment.letter_spacing,
                                                   segment.font_weight, segment.text_transform)
            
            
            if segment.background_color:
                bg_svg = self._render_background(
                    segment, x, current_y, text_width, segment.font_size, text_anchor
                )
                if bg_svg:
                    svg_parts.append(bg_svg)
            
            
            attrs = [
                f'x="{x}"',
                f'y="{current_y}"',
                f'font-family="{segment.font_family}"',
                f'font-size="{segment.font_size}"',
                f'font-weight="{segment.font_weight}"',
                f'fill="{segment.color}"',
                f'opacity="{segment.opacity}"',
                f'text-anchor="{text_anchor}"',
            ]
            
            
            if segment.font_style != "normal":
                attrs.append(f'font-style="{segment.font_style}"')
            
            if segment.letter_spacing > 0:
                attrs.append(f'letter-spacing="{segment.letter_spacing}em"')
            
            
            if segment.shadow_blur > 0 and segment.shadow_color:
                filter_id = f"shadow_{id(segment)}"
                attrs.append(f'filter="url(#{filter_id})"')
            
            svg_parts.append(f'  <text {" ".join(attrs)}>{escaped_text}</text>')
            
            
            if line_idx < len(lines) - 1:
                descender = int(segment.font_size * 0.25)
                ascender = int(segment.font_size * 0.75)
                current_y += descender + 6 + ascender
        
        return svg_parts
    
    def _render_inline_group(
        self,
        segments: List[StyledSegment],
        x: int,
        y: int,
        text_anchor: str,
        svg_width: int
    ) -> List[str]:
        """
        Render multiple segments on the same line (using tspan)
        
        Args:
            segments: list of same-line segments
            x: X coordinate
            y: Y coordinate
            text_anchor: text anchor
            svg_width: SVG width
        
        Returns:
            list of SVG element strings
        """
        svg_parts = []
        
        if not segments:
            return svg_parts
        
        
        total_width = 0
        segment_widths = []
        for seg in segments:
            width = self._estimate_text_width(seg.text, seg.font_size, seg.letter_spacing,
                                              seg.font_weight, seg.text_transform)
            segment_widths.append(width)
            total_width += width
        
        
        if text_anchor == "middle":
            start_x = x - total_width // 2
        elif text_anchor == "end":
            start_x = x - total_width
        else:
            start_x = x
        
        
        current_x = start_x
        for seg_idx, segment in enumerate(segments):
            if segment.background_color:
                bg_svg = self._render_background(
                    segment, current_x, y, segment_widths[seg_idx], segment.font_size, "start"
                )
                if bg_svg:
                    svg_parts.append(bg_svg)
            current_x += segment_widths[seg_idx]
        
        
        first_seg = segments[0]
        max_font_size = max(seg.font_size for seg in segments)
        
        text_attrs = [
            f'x="{x}"',
            f'y="{y}"',
            f'text-anchor="{text_anchor}"',
        ]
        
        svg_parts.append(f'  <text {" ".join(text_attrs)}>')
        
        for seg_idx, segment in enumerate(segments):
            if not segment.text:
                continue
            
            display_text = self._apply_text_transform(segment.text, segment.text_transform)
            escaped_text = self._escape_xml(display_text)
            
            # Append a space to non-last segments to ensure spacing between tspans
            if seg_idx < len(segments) - 1 and not escaped_text.endswith(' '):
                escaped_text = escaped_text + ' '
            
            
            tspan_attrs = [
                f'font-family="{segment.font_family}"',
                f'font-size="{segment.font_size}"',
                f'font-weight="{segment.font_weight}"',
                f'fill="{segment.color}"',
                f'opacity="{segment.opacity}"',
            ]
            
            if segment.font_style != "normal":
                tspan_attrs.append(f'font-style="{segment.font_style}"')
            
            if segment.letter_spacing > 0:
                tspan_attrs.append(f'letter-spacing="{segment.letter_spacing}em"')
            
            svg_parts.append(f'    <tspan {" ".join(tspan_attrs)}>{escaped_text}</tspan>')
        
        svg_parts.append('  </text>')
        
        return svg_parts
    
    def _render_background(
        self,
        segment: StyledSegment,
        x: int,
        y: int,
        text_width: int,
        text_height: int,
        text_anchor: str
    ) -> str:
        """
        Render a text background rectangle
        
        Args:
            segment: StyledSegment
            x: text X coordinate
            y: text Y coordinate (baseline)
            text_width: text width
            text_height: font size
            text_anchor: text anchor
        
        Returns:
            SVG string for the background rect
        """
        if not segment.background_color:
            return ""
        
        padding = segment.background_padding
        radius = segment.background_radius
        
        
        rect_height = text_height + padding * 2
        rect_width = text_width + padding * 2
        
        
        if text_anchor == "middle":
            rect_x = x - text_width // 2 - padding
        elif text_anchor == "end":
            rect_x = x - text_width - padding
        else:
            rect_x = x - padding
        
        # Adjust y from baseline to rect top (baseline is ~75% of font height)
        rect_y = y - int(text_height * 0.75) - padding
        
        return (
            f'  <rect '
            f'x="{rect_x}" '
            f'y="{rect_y}" '
            f'width="{rect_width}" '
            f'height="{rect_height}" '
            f'rx="{radius}" '
            f'ry="{radius}" '
            f'fill="{segment.background_color}"/>'
        )
    
    def _generate_filters(self, segments: List[StyledSegment]) -> str:
        """
        Generate required SVG filters (e.g. shadows)
        
        Returns:
            filters SVG string
        """
        filter_parts = []
        
        for segment in segments:
            if segment.shadow_blur > 0 and segment.shadow_color:
                filter_id = f"shadow_{id(segment)}"
                offset_x, offset_y = segment.shadow_offset
                
                filter_svg = (
                    f'    <filter id="{filter_id}" x="-50%" y="-50%" width="200%" height="200%">\n'
                    f'      <feDropShadow dx="{offset_x}" dy="{offset_y}" '
                    f'stdDeviation="{segment.shadow_blur / 2}" '
                    f'flood-color="{segment.shadow_color}" flood-opacity="0.5"/>\n'
                    f'    </filter>'
                )
                filter_parts.append(filter_svg)
        
        return '\n'.join(filter_parts)
    
    def _apply_text_transform(self, text: str, transform: str) -> str:
        """
        Apply text case transformation
        
        Args:
            text: original text
            transform: transformation type (none/uppercase/lowercase/capitalize)
        
        Returns:
            transformed text
        """
        if transform == "uppercase":
            return text.upper()
        elif transform == "lowercase":
            return text.lower()
        elif transform == "capitalize":
            return text.title()
        return text
    
    def _estimate_text_width(self, text: str, font_size: int, letter_spacing: float = 0,
                             font_weight: int = 400, text_transform: str = 'none') -> int:
        """
        Estimate text width
        
        Args:
            text: text content
            font_size: font size
            letter_spacing: letter spacing (em)
            font_weight: font weight (100-900)
            text_transform: text case transformation
        
        Returns:
            estimated text width in pixels
        """
        display_text = self._apply_text_transform(text, text_transform)

        if font_weight >= 700:
            char_factor = 0.72
        elif font_weight >= 600:
            char_factor = 0.66
        else:
            char_factor = 0.6

        upper_count = sum(1 for c in display_text if c.isupper())
        lower_count = sum(1 for c in display_text if c.islower())
        other_count = len(display_text) - upper_count - lower_count
        base_width = (upper_count * char_factor * 1.15 + lower_count * char_factor + other_count * char_factor * 0.7) * font_size

        if letter_spacing > 0:
            spacing_width = len(display_text) * font_size * letter_spacing
            base_width += spacing_width
        
        return int(base_width)
    
    def _get_x_position(self, layout: LayoutConfig, width: int) -> int:
        """Calculate X coordinate based on alignment"""
        if layout.overall_align == "CENTER":
            return width // 2
        elif layout.overall_align == "RIGHT":
            return width - 40
        else:  # LEFT
            return 40
    
    def _get_text_anchor(self, layout: LayoutConfig) -> str:
        """Get text-anchor based on alignment"""
        anchor_map = {
            "LEFT": "start",
            "CENTER": "middle",
            "RIGHT": "end"
        }
        return anchor_map.get(layout.overall_align, "start")
    
    def _get_initial_y(self, segments: List[StyledSegment], layout: LayoutConfig) -> int:
        """Calculate initial Y coordinate"""
        if not segments:
            return 50
        
        
        first_size = segments[0].font_size
        
        return first_size + 30
    
    def _calculate_required_width(self, segments: List[StyledSegment]) -> int:
        """
        Calculate the required SVG width
        
        Based on the longest TITLE_PRIMARY line.
        Supports inline multi-style (total width of multiple segments on the same line).
        
        Args:
            segments: list of text segments
        
        Returns:
            required width in pixels
        """
        max_width = self.initial_width
        
        
        grouped = self._group_by_inline(segments)
        
        for group in grouped:
            
            group_width = 0
            
            for segment in group:
                if not segment.text:
                    continue
                
                
                lines = segment.text.split('\n') if '\n' in segment.text else [segment.text]
                for line in lines:
                    line_width = self._estimate_text_width(line, segment.font_size, segment.letter_spacing,
                                                           segment.font_weight, segment.text_transform)
                    group_width = max(group_width, line_width)
            
            estimated_width = group_width + 100
            max_width = max(max_width, estimated_width)
        
        
        max_width = min(max_width, 10000)
        
        
        return max(max_width, 600)
    
    def _process_text_wrapping(
        self, 
        segments: List[StyledSegment],
        layout: LayoutConfig,
        svg_width: int
    ) -> List[StyledSegment]:
        """
        Handle text wrapping, especially for SUBTITLE
        
        Subtitles should auto-wrap based on title width.
        All new style attributes are preserved.
        """
        processed = []
        
        
        title_segments = [s for s in segments if s.role == "TITLE_PRIMARY"]
        
        for segment in segments:
            if segment.role == "SUBTITLE" and segment.text:
                width_multiplier = random.uniform(1.0, 2.0)
                
                
                chars_per_line = int((svg_width - 80) / (segment.font_size * 0.6) * width_multiplier)
                chars_per_line = max(30, min(chars_per_line, 120))
                
                wrapped_text = self._wrap_text(segment.text, chars_per_line)
                
                
                wrapped_segment = StyledSegment(
                    text=wrapped_text,
                    font_size=segment.font_size,
                    font_weight=segment.font_weight,
                    opacity=segment.opacity,
                    color=segment.color,
                    role=segment.role,
                    font_family=segment.font_family,
                    font_style=segment.font_style,
                    letter_spacing=segment.letter_spacing,
                    text_transform=segment.text_transform,
                    background_color=segment.background_color,
                    background_padding=segment.background_padding,
                    background_radius=segment.background_radius,
                    shadow_blur=segment.shadow_blur,
                    shadow_offset=segment.shadow_offset,
                    shadow_color=segment.shadow_color,
                    inline_group=segment.inline_group
                )
                processed.append(wrapped_segment)
            else:
                processed.append(segment)
        
        return processed
    
    def _wrap_text(self, text: str, max_chars: int) -> str:
        """
        Simple text wrapping
        
        Args:
            text: text to wrap
            max_chars: maximum characters per line
        
        Returns:
            wrapped text (separated by \\n)
        """
        if len(text) <= max_chars:
            return text
        
        lines = []
        words = text.split()
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > max_chars and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += word_length
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def _calculate_required_height(
        self, 
        segments: List[StyledSegment],
        layout: LayoutConfig
    ) -> int:
        """
        Calculate the actual height required for rendering
        
        Args:
            segments: processed text segments
            layout: layout configuration
        
        Returns:
            required height in pixels
        """
        total_height = 30
        
        for segment in segments:
            if not segment.text:
                continue
            
            
            lines = segment.text.split('\n') if '\n' in segment.text else [segment.text]
            num_lines = len([l for l in lines if l.strip()])
            
            
            line_height = segment.font_size + int(segment.font_size * 0.3)
            total_height += line_height * num_lines
            
            
            total_height += layout.spacing
        
        
        total_height -= layout.spacing
        
        
        return max(total_height, 0)
    
    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters"""
        return html.escape(text)
    
    def render_to_file(
        self,
        styled_segments: List[StyledSegment],
        layout: LayoutConfig,
        output_path: str,
        background_color: str = "#FFFFFF"
    ):
        """
        Render and save to file
        
        Args:
            styled_segments: list of styled text segments
            layout: layout configuration
            output_path: output file path
            background_color: background color
        """
        svg_content = self.render(styled_segments, layout, background_color)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
    
    @staticmethod
    def svg_to_png(svg_path: str, png_path: str = None, scale: float = 2.0, crop_whitespace: bool = True) -> bool:
        """
        Convert SVG to PNG (using cairosvg), with optional whitespace cropping
        
        Args:
            svg_path: SVG file path
            png_path: PNG output path (if None, uses the same name)
            scale: scale factor (default 2.0 for higher quality)
            crop_whitespace: whether to crop surrounding whitespace (default True)
        
        Returns:
            whether conversion succeeded
        """
        try:
            import cairosvg
            from PIL import Image
            
            if png_path is None:
                png_path = svg_path.replace('.svg', '.png')
            
            
            cairosvg.svg2png(url=svg_path, write_to=png_path, scale=scale)
            
            
            if crop_whitespace:
                SVGRenderer._crop_whitespace(png_path)
            
            return True
        except ImportError as e:
            print(f"  ⚠️  PNG conversion skipped: {e}")
            if 'cairosvg' in str(e):
                print(f"      Install with: pip install cairosvg")
            return False
        except Exception as e:
            print(f"  ⚠️  PNG conversion failed: {e}")
            return False
    
    @staticmethod
    def _crop_whitespace(png_path: str, padding: int = 0) -> bool:
        """
        Crop whitespace around a PNG image
        
        Args:
            png_path: PNG file path
            padding: margin to preserve (pixels), default 0
        
        Returns:
            whether cropping succeeded
        """
        try:
            from PIL import Image
            
            
            img = Image.open(png_path)
            
            
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            
            bbox = img.getbbox()
            
            if bbox is None:
                print(f"  ⚠️  Image is empty, skipping crop")
                return False
            
            
            cropped = img.crop(bbox)
            
            
            if padding > 0:
                
                new_width = cropped.width + padding * 2
                new_height = cropped.height + padding * 2
                new_img = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 0))
                new_img.paste(cropped, (padding, padding), cropped)
                cropped = new_img
            
            
            cropped.save(png_path, 'PNG')
            print(f"  ✓ Cropped PNG: {img.size} -> {cropped.size}")
            return True
            
        except Exception as e:
            print(f"  ⚠️  Failed to crop whitespace: {e}")
            return False

