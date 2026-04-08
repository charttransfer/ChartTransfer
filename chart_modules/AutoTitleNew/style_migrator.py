"""
Style Migrator

Migrate styles from a reference title structure to new text.

Enhanced: supports richer style attributes:
- Letter spacing (letter_spacing)
- Case transform (text_transform)
- Italic (font_style)
- Text background (background_color, background_padding, background_radius)
- Shadow (shadow_blur, shadow_offset, shadow_color)
- Same-line multi-style (inline_group)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from .structure_loader import TitleStructure, Segment


@dataclass
class StyledSegment:
    """Styled text segment (enhanced)."""
    text: str
    font_size: int  # pixels
    font_weight: int  # 100-900
    opacity: float  # 0.0-1.0
    color: str  # hex color
    role: str  # role id
    font_family: str = "Arial, sans-serif"  # font family

    # Additional style fields
    font_style: str = "normal"  # normal/italic
    letter_spacing: float = 0.0  # em
    text_transform: str = "none"  # none/uppercase/lowercase/capitalize

    # Text background
    background_color: Optional[str] = None  # background color (hex)
    background_padding: int = 0  # background padding
    background_radius: int = 0  # background corner radius

    # Shadow
    shadow_blur: int = 0  # shadow blur
    shadow_offset: Tuple[int, int] = (0, 0)  # shadow offset (x, y)
    shadow_color: Optional[str] = None  # shadow color

    # Inline multi-style
    inline_group: int = -1  # -1 = own line; 0,1,2,... = same-line group id


@dataclass
class InlineSegmentGroup:
    """Multiple segments on one line."""
    segments: List[StyledSegment] = field(default_factory=list)
    y_position: int = 0
    group_id: int = 0


@dataclass
class LayoutConfig:
    """Layout configuration."""
    direction: str  # COLUMN/ROW
    main_align: str  # START/CENTER/END
    cross_align: str  # START/CENTER/END
    spacing: int  # pixels
    overall_align: str  # LEFT/CENTER/RIGHT


class StyleMigrator:
    """Style migrator."""

    # Semantic font size to pixel mapping
    FONT_SIZE_MAP = {
        "large": 48,
        "medium": 32,
        "small": 28  # 28px to match subtitle size in reference images
    }

    # Font weight mapping
    FONT_WEIGHT_MAP = {
        "bold": 700,
        "semibold": 600,
        "normal": 400,
        "light": 300
    }

    # Emphasis level to opacity mapping
    EMPHASIS_MAP = {
        "high": 1.0,
        "medium": 0.85,
        "low": 0.7
    }

    def __init__(self, font_size_base: int = 48):
        """
        Args:
            font_size_base: Base font size (used for large).
        """
        self.font_size_base = font_size_base
        # Derive mapping from base size
        self.FONT_SIZE_MAP = {
            "large": font_size_base,
            "medium": int(font_size_base * 0.67),
            "small": int(font_size_base * 0.583)  # ~28px when base=48
        }

    def migrate_segment_style(
        self,
        new_text: str,
        reference_segment: Segment,
        color_override: Optional[str] = None,
        enhanced_typography: Optional[Dict] = None,
        inline_group: int = -1
    ) -> StyledSegment:
        """
        Apply reference segment styles to new text (enhanced).

        Args:
            new_text: New text content
            reference_segment: Reference segment
            color_override: Optional color override
            enhanced_typography: Enhanced typography dict (from LLM extraction)
            inline_group: Same-line group id (-1 means standalone line)

        Returns:
            Styled segment
        """
        style = reference_segment.style

        # Prefer enhanced_typography; otherwise legacy path
        if enhanced_typography:
            font_size = enhanced_typography.get('font_size_px', self.FONT_SIZE_MAP.get(style.fontSize, 32))
            font_weight = self.FONT_WEIGHT_MAP.get(enhanced_typography.get('weight', style.fontWeight), 400)
            font_style = enhanced_typography.get('font_style', 'normal')
            letter_spacing = enhanced_typography.get('letter_spacing', 0.0)
            text_transform = enhanced_typography.get('text_transform', 'none')
            color = color_override or enhanced_typography.get('color') or style.color or "#000000"
            font_family = self._map_font_family(enhanced_typography.get('font_family', style.fontFamily or 'Arial'))

            background = enhanced_typography.get('background', {})
            background_color = background.get('color') if background else None
            background_padding = background.get('padding', 0) if background else 0
            background_radius = background.get('border_radius', 0) if background else 0

            shadow = enhanced_typography.get('shadow', {})
            shadow_blur = shadow.get('blur', 0) if shadow else 0
            shadow_offset_x = shadow.get('offset_x', 0) if shadow else 0
            shadow_offset_y = shadow.get('offset_y', 0) if shadow else 0
            shadow_color = shadow.get('color') if shadow else None
        else:
            font_size = self.FONT_SIZE_MAP.get(style.fontSize, 32)
            font_weight = self.FONT_WEIGHT_MAP.get(style.fontWeight, 400)
            font_style = "normal"
            letter_spacing = 0.0
            text_transform = "none"
            color = color_override or style.color or "#000000"
            font_family = self._map_font_family(style.fontFamily) if style.fontFamily else "Arial, sans-serif"
            background_color = None
            background_padding = 0
            background_radius = 0
            shadow_blur = 0
            shadow_offset_x = 0
            shadow_offset_y = 0
            shadow_color = None

        opacity = self.EMPHASIS_MAP.get(style.emphasis, 1.0)

        return StyledSegment(
            text=new_text,
            font_size=font_size,
            font_weight=font_weight,
            opacity=opacity,
            color=color,
            role=reference_segment.role,
            font_family=font_family,
            font_style=font_style,
            letter_spacing=letter_spacing,
            text_transform=text_transform,
            background_color=background_color,
            background_padding=background_padding,
            background_radius=background_radius,
            shadow_blur=shadow_blur,
            shadow_offset=(shadow_offset_x, shadow_offset_y),
            shadow_color=shadow_color,
            inline_group=inline_group
        )

    def create_styled_segment_from_typography(
        self,
        text: str,
        typography: Dict,
        role: str = "TEXT",
        inline_group: int = -1
    ) -> StyledSegment:
        """
        Build StyledSegment directly from a typography dict (no reference_segment).

        Used for enhanced typography data from LLM extraction.

        Args:
            text: Text content
            typography: Typography dict
            role: Role id
            inline_group: Same-line group id

        Returns:
            StyledSegment instance
        """
        background = typography.get('background', {})
        background_color = background.get('color') if background else None
        background_padding = background.get('padding', 0) if background else 0
        background_radius = background.get('border_radius', 0) if background else 0

        shadow = typography.get('shadow', {})
        shadow_blur = shadow.get('blur', 0) if shadow else 0
        shadow_offset_x = shadow.get('offset_x', 0) if shadow else 0
        shadow_offset_y = shadow.get('offset_y', 0) if shadow else 0
        shadow_color = shadow.get('color') if shadow else None

        font_family_raw = typography.get('font_family', 'Arial')
        font_family = self._map_font_family(font_family_raw)

        return StyledSegment(
            text=text,
            font_size=typography.get('font_size_px', 32),
            font_weight=self.FONT_WEIGHT_MAP.get(typography.get('weight', 'normal'), 400),
            opacity=1.0,
            color=typography.get('color', '#000000'),
            role=role,
            font_family=font_family,
            font_style=typography.get('font_style', 'normal'),
            letter_spacing=typography.get('letter_spacing', 0.0),
            text_transform=typography.get('text_transform', 'none'),
            background_color=background_color,
            background_padding=background_padding,
            background_radius=background_radius,
            shadow_blur=shadow_blur,
            shadow_offset=(shadow_offset_x, shadow_offset_y),
            shadow_color=shadow_color,
            inline_group=inline_group
        )

    def _map_font_family(self, font_family: str) -> str:
        """
        Map a font family name to a usable CSS font-family stack.

        Args:
            font_family: Detected family (e.g. serif, sans-serif, Georgia)

        Returns:
            CSS font-family string
        """
        font_map = {
            # Category fallbacks
            "serif": "Georgia, 'Times New Roman', Times, serif",
            "sans-serif": "Arial, Helvetica, sans-serif",
            "monospace": "'Courier New', Courier, monospace",
            "decorative": "Impact, 'Arial Black', sans-serif",
            "handwritten": "'Comic Sans MS', 'Bradley Hand', cursive",
            "display": "Impact, 'Arial Black', sans-serif",

            # Descriptive families
            "traditional-serif": "Georgia, 'Times New Roman', Times, serif",
            "modern-serif": "Didot, 'Bodoni MT', serif",
            "slab-serif": "Rockwell, 'Courier New', serif",
            "classic-serif": "Garamond, 'Times New Roman', serif",
            "elegant-serif": "'Playfair Display', Georgia, serif",

            "modern-sans-serif": "Helvetica, Arial, sans-serif",
            "geometric-sans-serif": "Futura, 'Century Gothic', sans-serif",
            "humanist-sans-serif": "'Gill Sans', Optima, sans-serif",
            "condensed-sans-serif": "'Arial Narrow', 'Helvetica Condensed', sans-serif",
            "rounded-sans-serif": "Verdana, 'Arial Rounded MT', sans-serif",

            # Concrete families
            "arial": "Arial, Helvetica, sans-serif",
            "helvetica": "Helvetica, Arial, sans-serif",
            "times": "'Times New Roman', Times, serif",
            "times new roman": "'Times New Roman', Times, serif",
            "georgia": "Georgia, 'Times New Roman', serif",
            "garamond": "Garamond, 'Times New Roman', serif",
            "garamond italic": "Garamond, Georgia, serif",
            "playfair": "'Playfair Display', Georgia, serif",
            "playfair display": "'Playfair Display', Georgia, serif",
            "baskerville": "Baskerville, Georgia, serif",
            "courier": "'Courier New', Courier, monospace",
            "courier new": "'Courier New', Courier, monospace",
            "comic sans": "'Comic Sans MS', cursive",
            "impact": "Impact, 'Arial Black', sans-serif",
            "cooper": "'Cooper Black', 'Arial Black', Impact, sans-serif",
            "cooper black": "'Cooper Black', 'Arial Black', Impact, sans-serif",
            "verdana": "Verdana, Geneva, sans-serif",
            "trebuchet": "'Trebuchet MS', sans-serif",
            "lucida": "'Lucida Sans', 'Lucida Grande', sans-serif",
            "futura": "Futura, 'Century Gothic', sans-serif",
            "roboto": "Roboto, Arial, sans-serif",
            "open sans": "'Open Sans', Arial, sans-serif",
            "gotham": "Gotham, 'Montserrat', sans-serif",
            "bebas": "'Bebas Neue', Impact, sans-serif",
            "bebas neue": "'Bebas Neue', Impact, sans-serif",
            "oswald": "Oswald, Impact, sans-serif",
        }

        font_lower = font_family.lower().strip()

        if font_lower in font_map:
            return font_map[font_lower]

        if 'serif' in font_lower and 'sans' not in font_lower:
            return f"{font_family}, Georgia, serif"
        if 'mono' in font_lower or 'code' in font_lower:
            return f"{font_family}, 'Courier New', monospace"
        if 'script' in font_lower or 'handwritten' in font_lower:
            return f"{font_family}, cursive"
        return f"{font_family}, Arial, sans-serif"

    def migrate_layout(self, structure: TitleStructure) -> LayoutConfig:
        """
        Migrate layout settings from a reference structure.

        Args:
            structure: Reference title structure

        Returns:
            Layout configuration
        """
        root = structure.root

        return LayoutConfig(
            direction=root.direction,
            main_align=root.alignment.main,
            cross_align=root.alignment.cross,
            spacing=root.spacing,
            overall_align=structure.visual_properties.overall_alignment
        )

    def get_background_color(self, structure: TitleStructure) -> str:
        """
        Return background color from structure.

        Args:
            structure: Title structure

        Returns:
            Background color hex
        """
        colors = structure.visual_properties.colors
        return colors.get('background', '#FFFFFF')

    def match_segments(
        self,
        adapted_texts: list,
        reference_segments: list
    ) -> list:
        """
        Pair adapted texts with reference segments by role.

        Args:
            adapted_texts: List of {"role": "...", "text": "..."}
            reference_segments: Reference segment list

        Returns:
            List of (text, ref_segment) pairs
        """
        matches = []

        role_map = {}
        for seg in reference_segments:
            if seg.role not in role_map:
                role_map[seg.role] = []
            role_map[seg.role].append(seg)

        for adapted in adapted_texts:
            role = adapted['role']
            text = adapted['text']

            if role in role_map and role_map[role]:
                ref_seg = role_map[role].pop(0)
                matches.append((text, ref_seg))

        return matches

    def create_inline_segments(
        self,
        full_text: str,
        inline_specs: List[Dict],
        role: str = "TITLE_PRIMARY",
        inline_group: int = 0
    ) -> List[StyledSegment]:
        """
        Create multiple same-line StyledSegments from inline_specs.

        For multi-color text on one line, e.g. "Refugees in the U.S."

        Two formats:
        1. New (from typography): [{role, content, typography}]
           - Split new text using reference word-count ratios
        2. Legacy (text_pattern): [{text_pattern, word_count, typography}]

        Args:
            full_text: Full generated text
            inline_specs: inline_segments spec list
            role: Role id (legacy format)
            inline_group: Same-line group id

        Returns:
            List of StyledSegments
        """
        segments = []

        if inline_specs and 'content' in inline_specs[0]:
            ref_word_counts = []
            for spec in inline_specs:
                ref_content = spec.get('content', '')
                word_count = len(ref_content.split())
                ref_word_counts.append(max(1, word_count))

            total_ref_words = sum(ref_word_counts)

            new_words = full_text.split()
            total_new_words = len(new_words)

            print(f"  [inline] reference word-count ratios: {ref_word_counts}, new text words: {total_new_words}")

            used_words = 0
            for spec_idx, spec in enumerate(inline_specs):
                typography = spec.get('typography', {})
                segment_role = spec.get('role', role)

                if spec_idx == len(inline_specs) - 1:
                    segment_words = new_words[used_words:]
                else:
                    ratio = ref_word_counts[spec_idx] / total_ref_words
                    word_count = max(1, round(total_new_words * ratio))
                    segment_words = new_words[used_words:used_words + word_count]
                    used_words += word_count

                segment_text = ' '.join(segment_words)
                if not segment_text:
                    continue

                styled_seg = self.create_styled_segment_from_typography(
                    text=segment_text,
                    typography=typography,
                    role=segment_role,
                    inline_group=inline_group
                )
                segments.append(styled_seg)
                print(f"  ✓ created inline segment: role={segment_role}, text='{segment_text}', color={typography.get('color')}")
        else:
            words = full_text.split()
            used_words = 0

            for spec_idx, spec in enumerate(inline_specs):
                pattern = spec.get('text_pattern', 'remaining')
                typography = spec.get('typography', {})
                word_count = spec.get('word_count', 1)

                if pattern == 'first_word':
                    segment_words = words[:1]
                    used_words = 1
                elif pattern == 'first_words':
                    segment_words = words[:word_count]
                    used_words = word_count
                elif pattern == 'last_word':
                    segment_words = words[-1:]
                elif pattern == 'last_words':
                    segment_words = words[-word_count:]
                elif pattern == 'remaining':
                    segment_words = words[used_words:]
                elif pattern.startswith('contains:'):
                    keyword = pattern.split(':', 1)[1]
                    segment_words = [w for w in words if keyword.lower() in w.lower()]
                else:
                    segment_words = words[used_words:]

                segment_text = ' '.join(segment_words)
                if not segment_text:
                    continue

                styled_seg = self.create_styled_segment_from_typography(
                    text=segment_text,
                    typography=typography,
                    role=role,
                    inline_group=inline_group
                )
                segments.append(styled_seg)

        return segments
