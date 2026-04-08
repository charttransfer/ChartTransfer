import json
import os
import sys
from typing import Dict, Optional, List, Tuple, Set, Union
from logging import getLogger
import logging
import time
import numpy as np
import subprocess
import re
from lxml import etree
from PIL import Image
import base64
import io
import tempfile
import random
import fcntl
import time
import traceback
from pathlib import Path
from bs4 import BeautifulSoup

project_root = Path(__file__).parent

from chart_modules.reference_recognize.generate_color import generate_distinct_palette, rgb_to_hex


def _import_chart_pipeline():
    """Lazy-load ChartPipeline modules (not included in open-source release)."""
    from chart_modules.ChartPipeline.modules.chart_engine.chart_engine import get_template_for_chart_name
    from chart_modules.ChartPipeline.modules.chart_engine.utils.paint_innerchart import render_chart_to_svg
    from chart_modules.ChartPipeline.modules.infographics_generator.svg_utils import extract_svg_content, adjust_and_get_bbox
    from chart_modules.ChartPipeline.modules.infographics_generator.template_utils import select_template
    from chart_modules.ChartPipeline.modules.infographics_generator.data_utils import process_temporal_data, process_numerical_data, deduplicate_combinations
    from chart_modules.ChartPipeline.modules.chart_engine.template.template_registry import get_template_for_chart_type, get_template_for_chart_name
    return {
        'get_template_for_chart_name': get_template_for_chart_name,
        'render_chart_to_svg': render_chart_to_svg,
        'extract_svg_content': extract_svg_content,
        'adjust_and_get_bbox': adjust_and_get_bbox,
        'select_template': select_template,
        'process_temporal_data': process_temporal_data,
        'process_numerical_data': process_numerical_data,
        'deduplicate_combinations': deduplicate_combinations,
        'get_template_for_chart_type': get_template_for_chart_type,
    }
from chart_modules.example_info_extractor.integrate_color_palette import (
    get_field_values_from_data, 
    assign_colors_to_fields,
    create_example_based_colors_section,
    load_color_palettes
)

padding = 50
between_padding = 35


from chart_modules.style_refinement import svg_to_png

def remove_gridlines_from_svg(svg_content: str) -> str:
    """
    Remove all elements with data-tag="gridline" and data-tag="background" from SVG content.
    
    Args:
        svg_content: Original SVG string
    
    Returns:
        SVG string with gridlines and backgrounds removed
    """
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(svg_content.encode('utf-8'), parser)
    
    # Find and remove all elements with data-tag="gridline" or data-tag="background"
    elements_to_remove = root.xpath('//*[@data-tag="gridline" or @data-tag="background"]')
    
    for elem in elements_to_remove:
        parent = elem.getparent()
        if parent is not None:
            parent.remove(elem)
    
    # Convert back to string
    return etree.tostring(root, encoding='unicode', method='xml')

def remove_overlap_elements_from_svg(svg_content: str) -> str:
    """
    Remove elements from SVG content that do not meet retention criteria.
    Retention criteria:
    1. Elements whose data-tag attribute contains "axis" (e.g. data-tag="axis", data-tag="axis-tick")
    2. Elements of type text
    3. Elements of type image
    
    Args:
        svg_content: Original SVG string
    
    Returns:
        SVG string with non-qualifying elements removed
    """
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(svg_content.encode('utf-8'), parser)
    
    def should_keep(elem):
        tag = elem.tag
        if tag is None:
            return False
        
        tag_name = tag.split('}')[-1] if '}' in tag else tag
        
        if tag_name == 'text' or tag_name == 'image':
            return True
        
        data_tag = elem.get('data-tag', '')
        if data_tag and 'axis' in data_tag:
            return True
        
        return False
    
    def has_keepable_descendant(elem):
        if should_keep(elem):
            return True
        for child in elem:
            if has_keepable_descendant(child):
                return True
        return False
    
    def process_element(elem):
        if elem.tag is None:
            return
        
        if should_keep(elem):
            children = list(elem)
            for child in children:
                process_element(child)
        elif has_keepable_descendant(elem):
            children = list(elem)
            for child in children:
                process_element(child)
        else:
            parent = elem.getparent()
            if parent is not None:
                parent.remove(elem)
    
    children = list(root)
    for child in children:
        process_element(child)
    
    return etree.tostring(root, encoding='unicode', method='xml')

def make_infographic(data: Dict, chart_svg_content: str, output_dir: str, bg_color) -> str:
    cp = _import_chart_pipeline()
    adjust_and_get_bbox = cp['adjust_and_get_bbox']
    bg_color = rgb_to_hex(bg_color)
    chart_content, chart_width, chart_height, chart_offset_x, chart_offset_y = adjust_and_get_bbox(chart_svg_content, bg_color)
    # bg_color = "#000001"
    chart_svg_content = f"""<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' width='{chart_width}' height='{chart_height}'>
        {chart_content}</svg>"""
    output_dir_path = os.path.dirname(output_dir)

    # Create directory if it does not exist
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    
    with open(output_dir, 'w', encoding='utf-8') as f:
        f.write(chart_svg_content)
        
    # Convert to PNG
    if output_dir.endswith('.svg'):
        png_path = output_dir.replace('.svg', '.png')
        try:
            print(f"Converting to PNG: {png_path}")
            # Use the SVG -> HTML -> Screenshot method
            success = svg_to_png(chart_svg_content, png_path, background_color=None)
            if success:
                print(f"Converted to PNG: {png_path}")
            else:
                print(f"Error converting to PNG: conversion failed")
        except Exception as e:
            print(f"Error converting to PNG: {e}")
        
        # Generate no-grid version
        png_no_grid_path = output_dir.replace('.svg', '_no_grid.png')
        try:
            print(f"Generating no-grid PNG: {png_no_grid_path}")
            svg_no_grid = remove_gridlines_from_svg(chart_svg_content)
            success_no_grid = svg_to_png(svg_no_grid, png_no_grid_path, background_color=None)
            if success_no_grid:
                print(f"Generated no-grid PNG: {png_no_grid_path}")
            else:
                print(f"Error generating no-grid PNG: conversion failed")
        except Exception as e:
            print(f"Error generating no-grid PNG: {e}")
            traceback.print_exc()
        
        # Generate not-overlap version
        png_not_overlap_path = output_dir.replace('.svg', '_not_overlap.png')
        try:
            print(f"Generating not-overlap PNG: {png_not_overlap_path}")
            svg_not_overlap = remove_overlap_elements_from_svg(chart_svg_content)
            success_not_overlap = svg_to_png(svg_not_overlap, png_not_overlap_path, background_color=None)
            if success_not_overlap:
                print(f"Generated not-overlap PNG: {png_not_overlap_path}")
            else:
                print(f"Error generating not-overlap PNG: conversion failed")
        except Exception as e:
            print(f"Error generating not-overlap PNG: {e}")
            traceback.print_exc()
            
    return output_dir


def generate_variation(input: str, output: str, chart_template, main_colors = None, bg_color = None, text_color = None, data_mark_color = None, example_name = None, pair_id = None, chart_width = None, chart_height = None) -> bool:
    """
    Pipeline entry function for generating an infographic from a single file.

    Args:
        input: Input JSON file path
        output: Output SVG file path
        chart_template: Can be a string (template path) or a list [template_path, field_list]
        main_colors: List of main colors (RGB format)
        bg_color: Background color (RGB format)
        text_color: Text color (RGB format); auto-determined by background luminance if not provided
        data_mark_color: Data mark color (RGB format); used as the primary color if provided
        example_name: Reference example name, used to retrieve a full palette from color_palettes.json

    Returns:
        bool: Whether processing was successful
    """
    try:
        cp = _import_chart_pipeline()
        select_template = cp['select_template']
        get_template_for_chart_name = cp['get_template_for_chart_name']
        render_chart_to_svg = cp['render_chart_to_svg']
        extract_svg_content = cp['extract_svg_content']
        process_temporal_data = cp['process_temporal_data']
        process_numerical_data = cp['process_numerical_data']
        deduplicate_combinations = cp['deduplicate_combinations']

        print(f"[DEBUG generate_variation] Start")
        print(f"[DEBUG generate_variation] input: {input}")
        print(f"[DEBUG generate_variation] output: {output}")
        print(f"[DEBUG generate_variation] chart_template: {chart_template}")
        print(f"[DEBUG generate_variation] main_colors: {main_colors}")
        print(f"[DEBUG generate_variation] bg_color: {bg_color}")
        print(f"[DEBUG generate_variation] text_color: {text_color}")

        # Handle chart_template format
        if isinstance(chart_template, list):
            # Format: [template_path, fields] or [[template_path, fields]]
            if len(chart_template) >= 2 and isinstance(chart_template[1], list):
                template_path = chart_template[0]
                template_fields = chart_template[1]
            else:
                template_path = chart_template[0]
                template_fields = []
            template_for_select = [(template_path, template_fields)]
        else:
            # String format template path
            template_path = chart_template
            template_for_select = [(template_path, [])]

        print("chart_template:", chart_template)
        print("template_for_select:", template_for_select)
        with open(input, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["name"] = input

        if pair_id:
            icons_path = Path("logs/offline_icons") / pair_id / "icons.json"
            if icons_path.exists():
                with open(icons_path, "r", encoding="utf-8") as f:
                    icons_data = json.load(f)
                if "images" not in data:
                    data["images"] = {}
                if "field" not in data["images"]:
                    data["images"]["field"] = {}
                data["images"]["field"].update(icons_data.get("field", {}))

        # Check if template_path is absolute path to JS file (Dynamic Generation)
        if os.path.isabs(template_path) and template_path.endswith('.js') and os.path.exists(template_path):
             print(f"[INFO] Detected absolute template path: {template_path}")
             engine = 'd3-js'
             chart_type = 'custom'
             chart_name = template_path
             # Need ordered_fields. If passed in list, use it.
             if isinstance(chart_template, list) and len(chart_template) >= 2:
                 ordered_fields = chart_template[1]
             else:
                 ordered_fields = []
        else:
             # Select template
             engine, chart_type, chart_name, ordered_fields = select_template(template_for_select)

        # Check if template is blocked (in block_list)
        if engine is None or chart_name is None:
            print(f"[Skipped] Template is in block_list, not generating: {chart_template}")
            return False

        # Process data
        process_data_start = time.time()
        for i, field in enumerate(ordered_fields):
            data["data"]["columns"][i]["role"] = field
        process_temporal_data(data)
        process_numerical_data(data)
        deduplicate_combinations(data)
        print("main_colors:",main_colors)
        # Apply color mapping
        try:
            # If example_name is provided, use the full logic from integrate_color_palette
            if example_name:
                print(f"[INFO] Using example_name '{example_name}' to get palette from color_palettes.json")
                color_palettes = load_color_palettes()
                colors_section = create_example_based_colors_section(example_name, data, color_palettes)
                
                if colors_section:
                    data["colors"] = colors_section
                    print(f"[INFO] Created color config using integrate_color_palette logic")
                    print(f"[INFO] Primary: {colors_section['other']['primary']}")
                    print(f"[INFO] Secondary: {colors_section['other']['secondary']}")
                    print(f"[INFO] Field mappings: {len(colors_section['field'])}")
                    print(f"[INFO] colors_section: {colors_section}")
                else:
                    print(f"[WARNING] Could not create color config from example_name, falling back to main_colors")
                    example_name = None  # Fall back to the logic below
            
            # If no example_name or creation failed, use the provided main_colors
            if not example_name and main_colors and len(main_colors) > 0:
                print(f"[INFO] Applying color mapping, main_colors: {main_colors}")
                # Convert RGB colors to HEX
                def rgb_to_hex_color(rgb):
                    if isinstance(rgb, list) and len(rgb) == 3:
                        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
                    return rgb
                
                hex_colors = [rgb_to_hex_color(c) for c in main_colors]
                
                # Get categorical field values from data
                field_values_dict = get_field_values_from_data(data)
                
                # Assign colors to each categorical field
                field_color_mapping = {}
                for col_name, values in field_values_dict.items():
                    field_color_mapping.update(assign_colors_to_fields(values, hex_colors))
                
                # Create colors field
                if "colors" not in data:
                    data["colors"] = {}
                
                data["colors"]["field"] = field_color_mapping
                
                # Determine primary/secondary/available_colors based on whether data_mark_color is provided
                if data_mark_color:
                    # Prioritize data_mark_color as primary
                    data_mark_hex = rgb_to_hex_color(data_mark_color)
                    primary_color = data_mark_hex
                    secondary_color = hex_colors[0] if hex_colors else "#ee62f5"
                    available_colors = hex_colors[1:] if len(hex_colors) > 1 else []
                    print(f"[INFO] Using data_mark_color as primary: {data_mark_hex}")
                else:
                    # Use original logic
                    primary_color = hex_colors[0] if hex_colors else "#3f8aff"
                    secondary_color = hex_colors[1] if len(hex_colors) > 1 else "#ee62f5"
                    available_colors = hex_colors[2:] if len(hex_colors) > 2 else []
                
                data["colors"]["other"] = {
                    "primary": primary_color,
                    "secondary": secondary_color
                }
                data["colors"]["available_colors"] = available_colors
                
                if bg_color:
                    bg_hex = rgb_to_hex_color(bg_color)
                    data["colors"]["background_color"] = bg_hex
                
                print(f"[INFO] Color mapping complete: {len(field_color_mapping)} field values mapped")
                print(f"[DEBUG] field_color_mapping: {field_color_mapping}")
            
            # Handle text color (independent of color mapping method)
            if text_color:
                def rgb_to_hex_color(rgb):
                    if isinstance(rgb, list) and len(rgb) == 3:
                        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
                    return rgb
                text_hex = rgb_to_hex_color(text_color)
                if "colors" not in data:
                    data["colors"] = {}
                data["colors"]["text_color"] = text_hex
                print(f"[INFO] Using text color: {text_hex}")
            else:
                # If text_color is not provided, auto-determine based on background color
                if bg_color:
                    print("bg_color:",bg_color)
                    # Calculate background color luminance
                    luminance = (0.2126 * bg_color[0] + 0.7152 * bg_color[1] + 0.0722 * bg_color[2]) / 255
                    if "colors" not in data:
                        data["colors"] = {}
                    if luminance > 0.5:
                        data["colors"]["text_color"] = "#000000"  # black
                        print(f"[INFO] Light background detected, auto-using black text")
                    else:
                        data["colors"]["text_color"] = "#FFFFFF"  # white
                        print(f"[INFO] Dark background detected, auto-using white text")
                elif "colors" in data and "text_color" not in data["colors"]:
                    data["colors"]["text_color"] = "#000000"  # default black
            
            if "colors" in data:
                print(f"[DEBUG] text_color: {data['colors'].get('text_color')}")
                
        except Exception as e:
            print(f"[ERROR] Failed to apply color mapping: {e}")
            traceback.print_exc()
        
        # print("data:",time.time())
        data["colors_dark"] = data["colors"]
        
        # Get chart template
        get_template_start = time.time()
        print("chart_name:",chart_name)
        
        # Check if chart_name is absolute path (Dynamic)
        if os.path.isabs(chart_name) and chart_name.endswith('.js') and os.path.exists(chart_name):
             print(f"[INFO] Using dynamic template file: {chart_name}")
             engine_obj = 'd3-js' # Just non-None
             template = chart_name
        else:
             engine_obj, template = get_template_for_chart_name(chart_name)

        if engine_obj is None or template is None:
            logger.error(f"Failed to load template: {engine}/{chart_type}/{chart_name}")
            return False

        # print("template:",time.time())
        
        # Process output filename, replace path separators with underscores
        title_font_family = "Arial"
        if "hand" in chart_name:
            title_font_family = "Comics"
        
        if '-' in engine:
            framework, framework_type = engine.split('-')
        elif '_' in engine:
            framework, framework_type = engine.split('_')
        else:
            framework = engine
            framework_type = None

        # print("start rendering:",time.time())
        _, chart_svg_content = render_chart_to_svg(
            json_data=data,
            js_file=template,
            framework=framework,
            framework_type=framework_type,
            html_output_path=output.replace('.svg', '.html')
        )
        chart_inner_content = extract_svg_content(chart_svg_content)
        
        assemble_start = time.time()
        data["chart_type"] = chart_type
        
        # print("rendering done:",time.time())
        
        print("bg_color:",bg_color)
        return make_infographic(
            data=data,
            chart_svg_content=chart_inner_content,
            output_dir=output,
            bg_color=bg_color
        )
                
    except Exception as e:
        print(f"Error processing infographics: {e} {traceback.format_exc()}")
        return False
    
    

if __name__ == "__main__":
    start = time.time()
    generate_variation(
        input="processed_data/App.json",
        output="buffer/1.svg",
        chart_template = ['d3-js/multiple pie chart/multiple_pie_chart_02', ['x', 'y', 'group']],
        main_colors = [[25, 33, 24], [234, 232, 216], [243, 228, 146], [100, 110, 99], [171, 172, 148]],
        bg_color = [255, 255, 255]
    )
    process_time = time.time() - start
    print(f"Rendering took {process_time:.4f} seconds")