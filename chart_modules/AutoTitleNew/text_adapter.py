"""
Text Adapter

Uses LLM to adapt new text to the length constraints of a reference structure
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
from openai import OpenAI

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import config
from .structure_loader import TitleStructure


class TextAdapter:
    """Text length adapter"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL
        )
        # self.model = "gemini-3-flash-preview"
        self.model = "gpt-4.1-mini"
    
    def adapt_text(self, new_text: str, reference_structure: TitleStructure) -> List[Dict[str, str]]:
        """
        Rewrite new text to match the length of the reference structure
        
        Args:
            new_text: new title text
            reference_structure: reference title structure
        
        Returns:
            list, each element is {"role": "...", "text": "..."}
        """
        # Extract length constraints from reference structure
        length_constraints = []
        for segment in reference_structure.root.children:
            length_constraints.append({
                "role": segment.role,
                "original_text": segment.content,
                "char_count": len(segment.content),
                "description": self._get_role_description(segment.role)
            })
        
        prompt = f'''You are an expert in title text adaptation. Rewrite the new title text to match the reference structure in length and semantics.

New title text:
{new_text}

Reference structure length constraints:
{json.dumps(length_constraints, ensure_ascii=False, indent=2)}

Requirements:
1. Preserve the core meaning and information of the new title
2. Character count for each segment should be close to the reference (allow ±20% variance)
3. TITLE_PRIMARY should be the main information, concise and impactful
4. SUBTITLE should be detailed explanation or supplementary information
5. TITLE_CONTEXT should be context or temporal background
6. **Must generate text for every role, no empty fields**
7. If reference has multiple TITLE_PRIMARY, display them on separate lines
8. Preserve language characteristics (Chinese is more compact, English may need more characters)

Return JSON format (include all roles):
[
{self._format_expected_roles(reference_structure)}
]

Return only JSON, no other explanation.'''
        
        response = self.call_llm(prompt)
        
        if response:
            try:
                return self._parse_adapted_texts(response)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[WARNING] LLM response parsing failed, using fallback method: {e}")
                print(f"[DEBUG] LLM response content: {response[:500]}")
                return self._fallback_split(new_text, reference_structure)
        else:
            return self._fallback_split(new_text, reference_structure)
    
    def call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """Call the LLM API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2048
        )
        return response.choices[0].message.content.strip()
    
    def _parse_adapted_texts(self, response: str) -> List[Dict[str, str]]:
        """Parse JSON returned by LLM"""
        # Remove possible markdown code blocks
        response = response.replace('```json', '').replace('```', '').strip()
        
        # Extract JSON
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        
        if json_start < 0 or json_end <= json_start:
            raise ValueError(f"No valid JSON array found: {response[:200]}")
        
        json_str = response[json_start:json_end]
        if not json_str:
            raise ValueError(f"Extracted JSON string is empty: {response[:200]}")
        
        data = json.loads(json_str)
        if not isinstance(data, list):
            raise ValueError(f"Parsed result is not a list: {type(data)}")
        
        return data
    
    def _fallback_split(self, text: str, structure: TitleStructure) -> List[Dict[str, str]]:
        """Fallback simple splitting method"""
        segments = []
        
        has_flag = False
        # Return in the order of roles defined in structure
        for seg in structure.root.children:
            if seg.role == "TITLE_PRIMARY":
                # First primary title uses original text, subsequent ones are left empty
                if (not segments or segments[-1]['role'] != 'TITLE_PRIMARY') and not has_flag:
                    segments.append({
                        "role": "TITLE_PRIMARY",
                        "text": text
                    })
                    has_flag = True
                else:
                    segments.append({
                        "role": "TITLE_PRIMARY",
                        "text": ""
                    })
            else:
                # Leave other roles empty
                segments.append({
                    "role": seg.role,
                    "text": ""
                })
        
        return segments
    
    def _format_expected_roles(self, structure: TitleStructure) -> str:
        """Format the expected role list (for LLM prompt)"""
        roles = []
        for seg in structure.root.children:
            desc = self._get_role_description(seg.role)
            roles.append(f'  {{"role": "{seg.role}", "text": "... ({desc})"}}')
        return ',\n'.join(roles)
    
    def adapt_text_to_requirements(
        self,
        text: str,
        content_requirements: str,
        line_count: int = 1,
    ) -> str:
        """Adapt text based on the length description in content_requirements.

        Args:
            text: original text
            content_requirements: requirements extracted from the scene graph, e.g.
                "Subject of the ranking, typically 2-3 words"
            line_count: target number of lines

        Returns:
            adapted text
        """
        print(f"[TextAdapter] adapt_text_to_requirements: text={text}, content_requirements={content_requirements}, line_count={line_count}")
        if not content_requirements:
            return text

        word_count = len(text.split())
        req_lower = content_requirements.lower()

        import re
        range_match = re.search(r'(\d+)\s*[-–]\s*(\d+)\s*words?', req_lower)
        single_match = re.search(r'(\d+)\s*words?', req_lower)
        if range_match:
            lo, hi = int(range_match.group(1)), int(range_match.group(2))
        elif single_match:
            n = int(single_match.group(1))
            lo, hi = max(1, n - 2), n + 2
        else:
            lo, hi = None, None

        if lo is not None and lo <= word_count <= hi:
            if line_count <= 1:
                return text
            return self._split_into_lines(text, line_count)

        line_hint = f"\n- Split into {line_count} roughly equal lines using \\n" if line_count > 1 else ""
        prompt = (
            f"Rewrite the following text to satisfy the content requirements, "
            f"while preserving the core meaning.\n\n"
            f"Original text: {text}\n\n"
            f"Content requirements: {content_requirements}\n"
            f"- The text MUST satisfy the length constraint described above{line_hint}\n"
            f"- Write a coherent, natural phrase — not keyword fragments\n"
            f"- Do NOT cut off words mid-sentence\n\n"
            f"Return only the rewritten text. No explanation."
        )

        response = self.call_llm(prompt, temperature=0.3)
        if response:
            adapted = response.strip().strip('"').strip("'")
            adapted = adapted.replace('\\n', '\n').replace('\\\\n', '\n')
            return adapted
        return text

    def _split_into_lines(self, text: str, line_count: int) -> str:
        """Split text into roughly equal lines at word boundaries."""
        if line_count <= 1:
            return text
        words = text.split()
        if len(words) <= line_count:
            return '\n'.join(words)
        avg = len(words) / line_count
        lines, current = [], []
        for w in words:
            current.append(w)
            if len(current) >= avg and len(lines) < line_count - 1:
                lines.append(' '.join(current))
                current = []
        if current:
            lines.append(' '.join(current))
        return '\n'.join(lines)

    def adapt_text_to_constraints(
        self,
        text: str,
        width: int,
        height: int,
        line_count: int,
        font_size: str = "medium"
    ) -> str:
        """
        Adapt text to fit specified width and line count constraints (pixel estimation, kept for compatibility)
        """
        char_width_map = {
            "large": 24,
            "medium": 16,
            "small": 12
        }
        char_width = char_width_map.get(font_size, 16)

        chars_per_line = int((width / char_width) * 0.9)
        max_chars = int(chars_per_line * line_count)

        if len(text) <= max_chars and line_count == 1:
            return text

        prompt = f'''Rewrite the following text as a concise title that fits within a character limit, while preserving the overall meaning and readability.

Original text: {text}
Constraints:
- Line count: {line_count}
- Approximate characters per line: {chars_per_line} (flexible, prioritize complete words)
- Total character limit: ~{max_chars} characters (not counting \\n)

Rules:
1. The result must read as a coherent, natural title — not a list of fragments
2. Capture the main idea of the original, even if specific details are omitted
3. Use \\n to split into {line_count} roughly equal lines
4. Do NOT cut off words mid-sentence — each line must end at a natural word boundary
5. Do NOT produce keyword fragments — write a proper short title

Return only the rewritten text. No explanation.'''

        response = self.call_llm(prompt, temperature=0.3)
        if response:
            adapted_text = response.strip().strip('"').strip("'")
            adapted_text = adapted_text.replace('\\n', '\n')
            adapted_text = adapted_text.replace('\\\\n', '\n')
            return adapted_text

        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        if line_count > 1:
            lines = []
            remaining = truncated
            for i in range(line_count):
                if len(remaining) <= chars_per_line:
                    lines.append(remaining)
                    break
                line = remaining[:chars_per_line]
                last_space = line.rfind(' ')
                if last_space > chars_per_line * 0.7:
                    line = remaining[:last_space]
                    remaining = remaining[last_space+1:]
                else:
                    remaining = remaining[chars_per_line:]
                lines.append(line)
            return '\n'.join(lines)
        return truncated
    
    def _get_role_description(self, role: str) -> str:
        descriptions = {
            "TITLE_PRIMARY": "main title",
            "TITLE_SECONDARY": "secondary title",
            "TITLE_NUMBER": "number/stat",
            "TITLE_CONTEXT": "context",
            "SUBTITLE": "subtitle"
        }
        return descriptions.get(role, role)

