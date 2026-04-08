"""
SVG to PNG conversion utility.
"""

import os
import tempfile
import shutil
from chart_modules.parse_utils import convert_svg_to_html
from chart_modules.screenshot_utils import get_driver, take_screenshot


def svg_to_png(svg_content: str, output_path: str, background_color: str = None) -> bool:
    """
    Convert SVG content to a PNG file
    using the SVG -> HTML -> Screenshot approach.

    Args:
        svg_content: SVG file content (string)
        output_path: Output PNG file path
        background_color: Background color (hex format), None for transparent background

    Returns:
        bool: Whether the conversion was successful
    """
    driver = None
    temp_dir = None
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_path = os.path.abspath(output_path)

        temp_dir = tempfile.mkdtemp()
        temp_svg_path = os.path.abspath(os.path.join(temp_dir, 'temp.svg'))
        temp_html_path = os.path.abspath(os.path.join(temp_dir, 'temp.html'))

        with open(temp_svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)

        convert_svg_to_html(temp_svg_path, temp_html_path)

        driver = get_driver()
        take_screenshot(driver, temp_html_path)

        temp_png_path = os.path.join(temp_dir, 'temp.png')
        if not os.path.exists(temp_png_path):
            temp_png_path = temp_html_path.replace('.html', '.png')
        
        if os.path.exists(temp_png_path):
            output_path = os.path.abspath(output_path)
            shutil.move(temp_png_path, output_path)
            print(f"SVG to PNG succeeded: {output_path}")
            return True
        else:
            print(f"SVG to PNG failed: generated PNG file not found")
            if temp_dir and os.path.exists(temp_dir):
                files = os.listdir(temp_dir)
                print(f"Files in temp directory: {files}")
            return False

    except Exception as e:
        print(f"SVG to PNG failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if driver:
            driver.quit()
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
