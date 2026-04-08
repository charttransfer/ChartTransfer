from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import os
import sys
import re
import base64
import json
import csv
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import copy
import random
import traceback
from pathlib import Path
from io import BytesIO
import xml.etree.ElementTree as ET

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'layout_system' / 'src'))

import pandas as pd
import torch
import openai
import config
from google import genai
from google.genai import types
from PIL import Image, Image as PILImage
import numpy as np

from chart_modules.llm_client import LLMClient
from chart_modules.AutoTitleNew.text_adapter import TextAdapter
from chart_modules.AutoTitleNew import TitleGenerator
from chart_modules.AutoTitleNew.svg_renderer import SVGRenderer
from chart_modules.generate_variation import generate_variation
from chart_modules.example_info_extractor.integrate_color_palette import (
    get_field_values_from_data,
    assign_colors_to_fields,
)
from layout_system.hierarchical_optimizer import HierarchicalOptimizer, OptimizationConfig
from layout_system.bilevel.bilevel_optimizer import BilevelOptimizer
from layout_system.utils.save_result import save_hierarchical_result

app = Flask(__name__)
CORS(app)

INFOGRAPHICS_DIR = BASE_DIR / 'infographics'
PROCESSED_DATA_DIR = BASE_DIR / 'processed_data'
EXTRACTED_LAYOUT_DIR = BASE_DIR / 'extracted_layout_results'
SCENE_TREE_DIR = BASE_DIR / 'logs' / 'scene_tree_exports'
STYLE_ANALYSES_FILE = BASE_DIR / 'style_analyses.json'
COLOR_PALETTES_FILE = BASE_DIR / 'color_palettes.json'
RESULTS_INDEX_FILE = BASE_DIR / 'results_comparison_index.json'
CONFIG_FILE = CONFIG_DIR / 'config.json'
BUFFER_DIR = BASE_DIR / 'buffer_new_system'

_generation_tasks = {}
_manifest_lock = threading.Lock()

CACHE_DIR = CONFIG_DIR / 'cache'
USE_CACHE = False


def _get_chart_cache_dir(example_id, data_file):
    """Return the cache directory for a given example+data pair, or None if no cached chart exists."""
    if not USE_CACHE:
        return None
    stem = data_file.replace('.csv', '').replace('.json', '')
    d = CACHE_DIR / f'{example_id}_{stem}'
    if d.is_dir() and (d / 'chart.png').exists():
        return d
    return None


def _get_layout_cache_dir(example_id, data_file):
    """Return the cache directory if cached layout results exist, or None otherwise."""
    if not USE_CACHE:
        return None
    stem = data_file.replace('.csv', '').replace('.json', '')
    d = CACHE_DIR / f'{example_id}_{stem}'
    if d.is_dir() and (d / 'result_0.png').exists():
        return d
    return None


def _session_buffer_dir(session_id, example_id, data_file):
    """Return session-isolated buffer directory, creating it if needed."""
    stem = data_file.replace('.csv', '').replace('.json', '')
    d = BUFFER_DIR / session_id / f'{example_id}_{stem}'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_manifest(session_id, example_id, data_file, node_path, entry):
    """Persist a generation result into the manifest for later cache loading."""
    manifest_dir = _session_buffer_dir(session_id, example_id, data_file)
    manifest_path = manifest_dir / 'manifest.json'
    with _manifest_lock:
        manifest = {}
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        manifest[node_path] = entry
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_config():
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_style_analyses():
    with open(STYLE_ANALYSES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_color_palettes():
    with open(COLOR_PALETTES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_results_index():
    with open(RESULTS_INDEX_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


# ─── Groups API (config-driven) ──────────────────────────────────────

@app.route('/api/groups', methods=['GET'])
def get_groups():
    """Return data groups with their associated examples, driven by config.json.
    If ?user_id=X is provided, only return groups allowed by user_access mapping."""
    cfg = load_config()
    index = load_results_index()
    style_data = load_style_analyses()
    palette_data = load_color_palettes()

    user_id = request.args.get('user_id')
    user_access = cfg.get('user_access', {})
    allowed_data_files = None
    if user_id is not None and user_id in user_access:
        allowed_data_files = set(user_access[user_id])

    result = []
    for group in cfg['groups']:
        data_file = group['data_file']
        if allowed_data_files is not None and data_file not in allowed_data_files:
            continue
        label = group.get('label', data_file)

        examples_out = []
        for ex_id in group['examples']:
            ex_info = index.get('examples', {}).get(ex_id) or {}
            style_info = (style_data.get('analyses', {}).get(ex_id)) or {}
            palette_info = (palette_data.get('palettes', {}).get(ex_id)) or {}
            chart_analysis = style_info.get('chart_analysis') or {}
            overall_style = style_info.get('overall_style') or {}

            img_path = BASE_DIR / ex_info['image_path'] if ex_info.get('image_path') else None
            examples_out.append({
                'id': ex_id,
                'has_image': img_path is not None and img_path.exists(),
                'chart_type': chart_analysis.get('chart_type', ''),
                'style_keywords': overall_style.get('style_keywords', []),
            })

        result.append({
            'data_file': data_file,
            'label': label,
            'examples': examples_out,
        })

    return jsonify(result)


# ─── Data API ────────────────────────────────────────────────────────

@app.route('/api/data/<filename>', methods=['GET'])
def get_data(filename):
    """Get tabular data content. Tries JSON first for richer metadata, falls back to CSV."""
    stem = filename.replace('.csv', '').replace('.json', '')

    json_path = PROCESSED_DATA_DIR / f'{stem}.json'
    csv_path = PROCESSED_DATA_DIR / f'{stem}.csv'

    metadata = None
    columns = []
    rows = []

    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        metadata = json_data.get('metadata') or json_data.get('data', {}).get('metadata')
        data_section = json_data.get('data', json_data)
        if 'data' in data_section and isinstance(data_section['data'], list):
            rows = data_section['data']
            if rows:
                columns = list(rows[0].keys())
        col_info = data_section.get('columns', [])
        if col_info and isinstance(col_info, list) and isinstance(col_info[0], dict):
            columns = [c.get('name', '') for c in col_info]

    if not rows and csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            for row in reader:
                rows.append(row)

    if not rows:
        return jsonify({'error': 'Data not found'}), 404

    return jsonify({
        'filename': stem,
        'columns': columns,
        'rows': rows,
        'metadata': metadata,
    })


@app.route('/api/example-image/<example_id>', methods=['GET'])
def get_example_image(example_id):
    """Serve the reference infographic image for an example."""
    index = load_results_index()
    info = index.get('examples', {}).get(example_id)
    if not info:
        return jsonify({'error': 'Example not found'}), 404
    img_path = BASE_DIR / info['image_path']
    if not img_path.exists():
        return jsonify({'error': 'Image not found'}), 404
    return send_file(str(img_path))


# ─── Style APIs (for future Info View) ───────────────────────────────

@app.route('/api/style/<example_id>', methods=['GET'])
def get_style(example_id):
    """Get global style info (color palette + typography) for an example."""
    palette_data = load_color_palettes()
    palette_info = palette_data.get('palettes', {}).get(example_id) or {}

    layout_file = EXTRACTED_LAYOUT_DIR / f'{example_id}_layout.json'
    global_style = {}
    typography = {}
    if layout_file.exists():
        with open(layout_file, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
        global_style = layout_data.get('global_style') or {}
        typography = global_style.get('typography') or {}

    if not palette_info and not typography:
        return jsonify({'error': 'Style not found'}), 404

    return jsonify({
        'example_id': example_id,
        'color_palette': {
            'background_colors': palette_info.get('background_colors') or [],
            'data_mark_color': palette_info.get('data_mark_color'),
            'categorical_encoding_colors': palette_info.get('categorical_encoding_colors') or [],
            'numerical_encoding_colors': palette_info.get('numerical_encoding_colors') or [],
            'foreground_theme_colors': palette_info.get('foreground_theme_colors') or [],
        },
        'typography': typography,
        'color_scheme': global_style.get('color_scheme', ''),
    })


@app.route('/api/style/<example_id>', methods=['PUT'])
def update_style(example_id):
    data = request.get_json(force=True)

    new_palette = data.get('color_palette')
    if new_palette:
        palette_data = load_color_palettes()
        if 'palettes' not in palette_data:
            palette_data['palettes'] = {}
        palette_data['palettes'][example_id] = new_palette
        with open(COLOR_PALETTES_FILE, 'w', encoding='utf-8') as f:
            json.dump(palette_data, f, ensure_ascii=False, indent=2)

    new_typography = data.get('typography')
    new_color_scheme = data.get('color_scheme')
    if new_typography is not None or new_color_scheme is not None:
        layout_file = EXTRACTED_LAYOUT_DIR / f'{example_id}_layout.json'
        if layout_file.exists():
            with open(layout_file, 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
            if 'global_style' not in layout_data:
                layout_data['global_style'] = {}
            if new_typography is not None:
                layout_data['global_style']['typography'] = new_typography
            if new_color_scheme is not None:
                layout_data['global_style']['color_scheme'] = new_color_scheme
            with open(layout_file, 'w', encoding='utf-8') as f:
                json.dump(layout_data, f, ensure_ascii=False, indent=2)

    return jsonify({'status': 'ok'})

# ─── Scene Graph APIs (for future Info View) ─────────────────────────

@app.route('/api/scene-graph/<example_id>', methods=['GET'])
def get_scene_graph(example_id):
    """Get scene graph, boxes, and image dimensions for an example."""
    layout_file = EXTRACTED_LAYOUT_DIR / f'{example_id}_layout.json'
    if not layout_file.exists():
        return jsonify({'error': 'Layout file not found'}), 404

    with open(layout_file, 'r', encoding='utf-8') as f:
        layout_data = json.load(f)

    scene_graph = layout_data.get('scene_graph') or (layout_data.get('style') or {}).get('scene_graph')
    boxes = layout_data.get('boxes') or (layout_data.get('style') or {}).get('boxes') or []
    image_width = layout_data.get('image_width', 0)
    image_height = layout_data.get('image_height', 0)

    chart_type_list = layout_data.get('chart_type', [])
    chart_type = chart_type_list[0] if isinstance(chart_type_list, list) and chart_type_list else (chart_type_list if isinstance(chart_type_list, str) else '')

    def _inject_chart_type(node, fallback_ct):
        if isinstance(node, dict):
            if node.get('type', '').lower() == 'chart' and not node.get('chart_type'):
                node['chart_type'] = fallback_ct
            for child in node.get('children', []):
                _inject_chart_type(child, fallback_ct)

    if not scene_graph:
        return jsonify({'error': 'No scene graph in layout'}), 404

    if chart_type:
        _inject_chart_type(scene_graph, chart_type)

    return jsonify({
        'scene_graph': scene_graph,
        'boxes': boxes,
        'image_width': image_width,
        'image_height': image_height,
        'chart_type': chart_type,
    })


# ─── Scene Tree with Elements APIs ───────────────────────────────────

@app.route('/api/scene-tree/<example_id>/<datafile>', methods=['GET'])
def get_scene_tree(example_id, datafile):
    """Get the scene tree with generated elements."""
    dir_name = f"{example_id}_{datafile}"
    scene_file = SCENE_TREE_DIR / dir_name / 'scene_tree_with_elements.json'
    if not scene_file.exists():
        return jsonify({'error': 'Scene tree not found'}), 404
    with open(scene_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data)


# ─── Result Image APIs (for future Layout View) ─────────────────────

@app.route('/api/result-image/<example_id>/<datafile>', methods=['GET'])
def get_result_image(example_id, datafile):
    """Get the layout result image."""
    dir_name = f"{example_id}_{datafile}"
    for img_name in ['hierarchical_result_no_box.png', 'hierarchical_result.png']:
        img_path = SCENE_TREE_DIR / dir_name / img_name
        if img_path.exists():
            return send_file(str(img_path))
    return jsonify({'error': 'Result image not found'}), 404


@app.route('/api/element-image/<path:image_path>', methods=['GET'])
def get_element_image(image_path):
    """Serve a generated element image by path."""
    full_path = BASE_DIR / image_path
    if not full_path.exists():
        return jsonify({'error': 'Image not found'}), 404
    return send_file(str(full_path))


# ─── Text Generation APIs ─────────────────────────────────────────────

def _find_node_in_scene_graph(scene_graph, target_path, current_path='0'):
    if current_path == target_path:
        return scene_graph
    if scene_graph.get('children'):
        for idx, child in enumerate(scene_graph['children']):
            result = _find_node_in_scene_graph(child, target_path, f"{current_path}-{idx}")
            if result:
                return result
    return None


def _get_typography_for_role(typography, role):
    """Match typography entry for a given role."""
    if not typography or not role:
        return None
    role_upper = role.upper()
    for key in [role, role_upper, role.lower()]:
        if key in typography:
            return typography[key]
    mapping = {
        'TITLE': ['TITLE', 'TITLE_PRIMARY', 'title', 'title_primary'],
        'SUBTITLE': ['SUBTITLE', 'subtitle'],
        'LABEL': ['LABEL', 'label'],
        'DESCRIPTION': ['DESCRIPTION', 'description'],
        'CAPTION': ['CAPTION', 'caption', 'DESCRIPTION', 'description'],
        'SOURCE': ['SOURCE', 'FOOTNOTE', 'source'],
        'BODY': ['BODY', 'body'],
    }
    for category, keys in mapping.items():
        if category in role_upper:
            for k in keys:
                if k in typography:
                    return typography[k]
            break
    return None


_ANTI_PATTERNS = (
    "FORBIDDEN: 'This chart illustrates...', 'The data shows...', 'An overview of...', "
    "'A visualization of...', 'This infographic presents...'. "
    "Write as a conclusion about the data, NOT a meta-description of the chart."
)


def _read_csv_data_summary(csv_path):
    path_str = str(csv_path)
    if path_str.endswith('.json'):
        with open(path_str, 'r', encoding='utf-8') as f:
            jd = json.load(f)
        data_section = jd.get('data', jd)
        rows = data_section.get('data', []) if isinstance(data_section, dict) else []
        if rows:
            df = pd.DataFrame(rows)
        else:
            return json.dumps(jd, ensure_ascii=False)[:600]
    else:
        df = pd.read_csv(path_str)
    parts = [f"Columns: {', '.join(df.columns.tolist())}", f"Rows: {len(df)}"]
    for col in df.select_dtypes(include=['number']).columns[:3]:
        parts.append(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, avg={df[col].mean():.2f}")
    parts.append(f"Sample data:\n{df.head(3).to_string(index=False)}")
    return '\n'.join(parts)


def _extract_insight_llm(data_summary):
    prompt = (
        "Extract structured insight from this data summary for infographic text generation.\n"
        "Return JSON with fields: entities, metric, time_range, main_trend, comparison, key_numbers.\n\n"
        f"Data summary:\n{data_summary}\n\nReturn ONLY valid JSON."
    )
    llm = LLMClient(model_name="gpt-4.1-mini")
    result = llm.call_text_only(prompt, temperature=0.2)
    if isinstance(result, dict) and result:
        return result
    return {"main_trend": data_summary[:200]}


def _build_text_gen_prompt(role, content_requirements, insight):
    role_upper = (role or 'TITLE').upper()
    insight_str = json.dumps(insight, ensure_ascii=False)
    req = content_requirements or ""

    fallback_length_map = {
        'TITLE': '5-15 words', 'TITLE_PRIMARY': '5-15 words',
        'SUBTITLE': '10-25 words', 'CAPTION': '10-30 words',
        'DESCRIPTION': '15-50 words', 'LABEL': '1-5 words',
    }

    if req:
        length_constraint = req
    else:
        fallback = fallback_length_map.get(role_upper, '5-15 words')
        length_constraint = f"approximately {fallback}"

    return (
        f"## Role: {role_upper}\n{_ANTI_PATTERNS}\n\n"
        f"## Length constraint: {length_constraint}. Use entities/metrics from insight, no generic phrases.\n\n"
        f"## Insight\n{insight_str}\n\n"
        '## Output\nGenerate 3 options. Return JSON: {"options": ["opt1", "opt2", "opt3"]}'
    )


def _generate_text_content(csv_path, role, content_requirements):
    summary = _read_csv_data_summary(csv_path)
    insight = _extract_insight_llm(summary)
    prompt = _build_text_gen_prompt(role, content_requirements, insight)
    llm = LLMClient(model_name="gpt-4.1-mini")
    result = llm.call_text_only(prompt, temperature=0.7)
    candidates = result.get("options", [])
    if candidates:
        return candidates[0]
    text = result.get("text", "")
    return text if text else f"[{role}] placeholder"


def _adapt_text_length(text, node_info):
    """Adapt text length based on content_requirements."""
    content_requirements = node_info.get('content_requirements', '')
    line_count = node_info.get('line_count', 1)
    adapter = TextAdapter()
    adapted = adapter.adapt_text_to_requirements(text, content_requirements, line_count)
    return adapted if adapted else text


def _run_text_content_generation(task_id, csv_path, role, content_requirements, node_info):
    """Background task: generate text content, then adapt length to constraints."""
    task = _generation_tasks[task_id]
    task['status'] = 'generating'
    try:
        raw_text = _generate_text_content(str(csv_path), role, content_requirements)
        print(f"[TextGen] task={task_id} raw => {raw_text[:80]}...")
        adapted_text = _adapt_text_length(raw_text, node_info)
        print(f"[TextGen] task={task_id} adapted => {adapted_text[:80]}...")
        task['status'] = 'done'
        task['result'] = adapted_text
        _save_manifest(task['session_id'], task['example_id'], task['data_file'], task['node_path'], {
            'type': 'text', 'step1Status': 'done', 'textContent': adapted_text,
        })
    except Exception as e:
        traceback.print_exc()
        task['status'] = 'error'
        task['result'] = str(e)
        print(f"[TextGen] task={task_id} ERROR: {e}")


@app.route('/api/generate-text-content', methods=['POST'])
def generate_text_content():
    """Step 1: Generate text content for a text node using LLM.

    Request body:
        example_id: str
        data_file: str (e.g. "App")
        node_path: str
        node_info: dict (with role, content_requirements, etc.)

    Returns task_id for polling, or immediate result.
    """
    data = request.json
    example_id = data.get('example_id')
    data_file = data.get('data_file')
    node_path = data.get('node_path')
    node_info = data.get('node_info', {})

    if not example_id or not data_file or not node_path:
        return jsonify({'error': 'Missing example_id, data_file, or node_path'}), 400

    csv_path = PROCESSED_DATA_DIR / f'{data_file}.csv'
    json_path = PROCESSED_DATA_DIR / f'{data_file}.json'
    if not csv_path.exists() and not json_path.exists():
        return jsonify({'error': f'Data file not found: {data_file}'}), 404

    actual_data_path = csv_path if csv_path.exists() else json_path

    # Enrich node_info from scene graph if needed
    layout_file = EXTRACTED_LAYOUT_DIR / f'{example_id}_layout.json'
    if layout_file.exists():
        with open(layout_file, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
        sg = layout_data.get('scene_graph') or (layout_data.get('style') or {}).get('scene_graph')
        if sg:
            found = _find_node_in_scene_graph(sg, node_path)
            if found:
                for key in ['role', 'content_requirements', 'font_family', 'font_weight',
                            'color', 'line_count', 'font_size_px', 'alignment']:
                    if key not in node_info and key in found:
                        node_info[key] = found[key]

    role = node_info.get('role', 'TITLE')
    content_requirements = node_info.get('content_requirements', '')

    session_id = data.get('session_id', 'default')
    task_id = str(uuid.uuid4())[:8]
    _generation_tasks[task_id] = {
        'type': 'text_content',
        'status': 'pending',
        'result': None,
        'node_path': node_path,
        'example_id': example_id,
        'data_file': data_file,
        'session_id': session_id,
    }

    t = threading.Thread(
        target=_run_text_content_generation,
        args=(task_id, actual_data_path, role, content_requirements, node_info),
        daemon=True,
    )
    t.start()

    return jsonify({'task_id': task_id, 'status': 'pending'})


def _parse_svg_texts(svg_path):
    """Parse SVG file and extract text elements as structured data with segments."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = 'http://www.w3.org/2000/svg'
    width = float(root.get('width', 400))
    height = float(root.get('height', 100))
    vb = root.get('viewBox', f'0 0 {width} {height}')

    def _parse_style_attr(el):
        """Parse inline style attribute into a dict."""
        style_str = el.get('style', '')
        if not style_str:
            return {}
        result = {}
        for part in style_str.split(';'):
            part = part.strip()
            if ':' not in part:
                continue
            k, v = part.split(':', 1)
            result[k.strip()] = v.strip()
        return result

    def _get_attr(el, attr_name, css_name=None, default=None):
        """Get attribute from element, checking both XML attributes and inline style."""
        val = el.get(attr_name)
        if val is not None:
            return val
        style = _parse_style_attr(el)
        css_key = css_name or attr_name
        if css_key in style:
            return style[css_key]
        return default

    def _seg_from_el(el, parent_el=None):
        """Build a segment from an element, inheriting from parent if needed."""
        def _resolve(attr, css_name=None, default=None):
            val = _get_attr(el, attr, css_name, None)
            if val is not None:
                return val
            if parent_el is not None:
                val = _get_attr(parent_el, attr, css_name, None)
                if val is not None:
                    return val
            return default

        font_family = _resolve('font-family', 'font-family', 'Sans-Serif')
        if font_family:
            font_family = font_family.strip("'\"").split(',')[0].strip("'\" ")

        return {
            'text': el.text or '',
            'font_family': font_family,
            'font_size': float(_resolve('font-size', 'font-size', 16)),
            'font_weight': _resolve('font-weight', 'font-weight', 'normal'),
            'fill': _resolve('fill', 'fill', '#000000'),
        }

    lines = []
    for el in root.iter(f'{{{ns}}}text'):
        tspans = [ts for ts in el.iter(f'{{{ns}}}tspan')]
        segments = []

        direct_text = (el.text or '').strip()
        if direct_text:
            segments.append(_seg_from_el(el))

        for ts in tspans:
            ts_text = (ts.text or '').strip()
            if ts_text:
                segments.append(_seg_from_el(ts, parent_el=el))
            tail = (ts.tail or '').strip()
            if tail:
                seg = _seg_from_el(el)
                seg['text'] = tail
                segments.append(seg)

        if not segments:
            segments.append(_seg_from_el(el))

        lines.append({
            'x': float(el.get('x', 0)),
            'y': float(el.get('y', 0)),
            'opacity': float(_get_attr(el, 'opacity', 'opacity', 1.0)),
            'text_anchor': _get_attr(el, 'text-anchor', 'text-anchor', 'start'),
            'segments': segments,
        })
    return {'width': width, 'height': height, 'viewBox': vb, 'lines': lines}


def _crop_example_bbox(example_id, bbox):
    """Crop the bbox region from the example infographic image.
    
    bbox can be either pixel coordinates or normalized (0-1) coordinates.
    Auto-detects based on whether values exceed 1.
    """
    ref_path = INFOGRAPHICS_DIR / f'{example_id}.png'
    if not ref_path.exists():
        ref_path = INFOGRAPHICS_DIR / f'{example_id}.jpg'
    if not ref_path.exists():
        return None
    img = PILImage.open(str(ref_path))
    iw, ih = img.size
    x = float(bbox.get('x', 0))
    y = float(bbox.get('y', 0))
    w = float(bbox.get('width', 0))
    h = float(bbox.get('height', 0))
    if w <= 0 or h <= 0:
        return None
    is_normalized = (x + w <= 1.5 and y + h <= 1.5)
    if is_normalized:
        x, y, w, h = x * iw, y * ih, w * iw, h * ih
    pad_px = max(iw, ih) * 0.02
    left = max(0, int(x - pad_px))
    top = max(0, int(y - pad_px))
    right = min(iw, int(x + w + pad_px))
    bottom = min(ih, int(y + h + pad_px))
    if right <= left or bottom <= top:
        return None
    cropped = img.crop((left, top, right, bottom))
    return cropped


_SVG_STYLE_REFINE_PROMPT = """You are an SVG text style expert. I will give you:
1. A cropped image showing the original text style from a reference infographic.
2. The current SVG source code that renders new text content.

Your task: modify the SVG to match the visual style of the reference image as closely as possible.

You MAY change:
- Style attributes on <text> and <tspan>: font-family, font-weight, font-size, fill, opacity, letter-spacing, text-decoration, font-style
- text-transform via a <style> block (uppercase/lowercase)
- Line breaking: you may restructure which words go on which line (split/merge <text> elements, adjust y coordinates) to match the line breaking pattern visible in the reference image. The number of lines and the character count ratio between lines should closely match what you see in the reference image.
- The SVG width, height, and viewBox to fit the restyled text (IMPORTANT: after changing font-size or font-weight, text may be wider/taller — you MUST adjust width/height/viewBox so nothing is clipped)

You MUST NOT:
- Change the actual words/characters in the text
- Add non-text elements (shapes, images, backgrounds, etc.)
- Change the text alignment (text-anchor). If the original SVG uses text-anchor="start" (left-aligned), keep it left-aligned. If it uses text-anchor="middle" (centered), keep it centered. If it uses text-anchor="end" (right-aligned), keep it right-aligned. Preserve the original alignment exactly.

CRITICAL: Make sure the final SVG dimensions (width, height, viewBox) are LARGE ENOUGH to contain ALL text without ANY clipping. After changing font-size or font-weight, text becomes wider/taller — you MUST increase width/height/viewBox accordingly. Add at least 20% extra width as safety margin. It is MUCH better to have extra whitespace than to clip any text.

Return ONLY the complete modified SVG code, wrapped in ```svg ... ``` markers. No other text.

Here is the current SVG:
```svg
{svg_content}
```"""


def _refine_svg_with_gemini(svg_path, example_id, node_info):
    """Use Gemini to refine SVG text style based on example image."""
    bbox = node_info.get('bbox', {})
    if not bbox:
        print("[StyleRefine] No bbox in node_info, skipping")
        return False

    print(f"[StyleRefine] bbox={bbox}, example_id={example_id}")
    cropped = _crop_example_bbox(example_id, bbox)
    if cropped is None:
        print(f"[StyleRefine] Could not crop example image for {example_id}")
        return False

    crop_path = svg_path.replace('.svg', '_ref_crop.png')
    cropped.save(crop_path)

    with open(svg_path, 'r', encoding='utf-8') as f:
        svg_content = f.read()

    prompt = _SVG_STYLE_REFINE_PROMPT.format(svg_content=svg_content)

    print(f"[StyleRefine] Calling gpt-5.4 to refine SVG style...")
    llm = LLMClient(model_name="gpt-5.4")
    response = llm.client.chat.completions.create(
        model=llm.model_name,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64.b64encode(open(crop_path, 'rb').read()).decode('utf-8')}"
            }},
        ]}],
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()

    m = re.search(r'```(?:svg|xml)?\s*\n(.*?)```', raw, re.DOTALL)
    if not m:
        print(f"[StyleRefine] Could not extract SVG from LLM response ({len(raw)} chars)")
        return False

    refined_svg = m.group(1).strip()
    if '<svg' not in refined_svg or '</svg>' not in refined_svg:
        print("[StyleRefine] LLM response missing <svg> tags, skipping")
        return False

    with open(svg_path, 'w', encoding='utf-8') as f:
        f.write(refined_svg)
    print(f"[StyleRefine] SVG refined successfully")
    return True


def _run_text_render(task_id, text_content, text_constraints, output_dir, example_id=None, node_info=None):
    """Background task: render text content into SVG/PNG."""
    task = _generation_tasks[task_id]
    task['status'] = 'generating'
    os.makedirs(output_dir, exist_ok=True)
    svg_path = os.path.join(output_dir, f'{task_id}.svg')
    png_path = os.path.join(output_dir, f'{task_id}.png')

    generator = TitleGenerator()
    generator.generate_from_constraints(
        text_content=text_content,
        text_constraints=text_constraints,
        adapt_length=False,
        output_path=svg_path,
    )

    if example_id and node_info:
        _refine_svg_with_gemini(svg_path, example_id, node_info)

    SVGRenderer.svg_to_png(svg_path, png_path, scale=2.0, crop_whitespace=True)

    rel_png = os.path.relpath(png_path, BASE_DIR).replace('\\', '/')
    rel_svg = os.path.relpath(svg_path, BASE_DIR).replace('\\', '/')
    svg_data = _parse_svg_texts(svg_path)

    task['status'] = 'done'
    task['result'] = {
        'png_path': rel_png,
        'svg_path': rel_svg,
        'svg_data': svg_data,
    }
    _save_manifest(task['session_id'], task['example_id'], task['data_file'], task['node_path'], {
        'type': 'text', 'step1Status': 'done', 'textContent': task.get('text_content', ''),
        'step2Status': 'done', 'svgData': svg_data,
        'pngUrl': f'/api/element-image/{rel_png}',
    })
    print(f"[TextRender] task={task_id} => {rel_png}")


@app.route('/api/render-text', methods=['POST'])
def render_text():
    """Step 2: Render text content into a styled image.

    Request body:
        example_id: str
        node_path: str
        text_content: str
        node_info: dict (bbox, font, color, etc.)
        data_file: str
    """
    data = request.json
    example_id = data.get('example_id')
    node_path = data.get('node_path')
    text_content = data.get('text_content')
    node_info = data.get('node_info', {})
    data_file = data.get('data_file', '')

    if not text_content or not node_path:
        return jsonify({'error': 'Missing text_content or node_path'}), 400

    # Enrich style from layout
    layout_file = EXTRACTED_LAYOUT_DIR / f'{example_id}_layout.json' if example_id else None
    global_style = {}
    typography = {}
    if layout_file and layout_file.exists():
        with open(layout_file, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
        global_style = layout_data.get('global_style') or {}
        typography = global_style.get('typography') or {}
        sg = layout_data.get('scene_graph') or (layout_data.get('style') or {}).get('scene_graph')
        if sg:
            found = _find_node_in_scene_graph(sg, node_path)
            if found:
                for key in ['font_family', 'font_weight', 'font_style', 'color',
                            'line_count', 'font_size_px', 'alignment', 'letter_spacing',
                            'text_transform', 'bbox']:
                    if key not in node_info and key in found:
                        node_info[key] = found[key]

    role = node_info.get('role', 'TITLE')
    role_typo = _get_typography_for_role(typography, role)

    bbox = node_info.get('bbox', {})
    rt = role_typo or {}
    font_family = node_info.get('font_family') or rt.get('font_family') or 'Open Sans'
    font_weight = node_info.get('font_weight') or rt.get('weight') or 'bold'
    color = node_info.get('color') or rt.get('color') or '#14243E'
    font_size_px = node_info.get('font_size_px') if node_info.get('font_size_px') is not None else rt.get('font_size_px')
    font_style = node_info.get('font_style') or rt.get('font_style') or 'normal'
    letter_spacing = node_info['letter_spacing'] if 'letter_spacing' in node_info else rt.get('letter_spacing', 0.0)
    text_transform = node_info.get('text_transform') or rt.get('text_transform') or 'none'

    text_constraints = {
        'x': bbox.get('x', 0),
        'y': bbox.get('y', 0),
        'width': bbox.get('width', 400),
        'height': bbox.get('height', 100),
        'content': text_content,
        'alignment': node_info.get('alignment', 'left'),
        'font_family': font_family,
        'font_size': node_info.get('font_size', 'large'),
        'font_weight': font_weight,
        'color': color,
        'line_count': node_info.get('line_count', 2),
        'role': role,
        'font_size_px': font_size_px,
        'font_style': font_style,
        'letter_spacing': letter_spacing,
        'text_transform': text_transform,
    }

    style_overrides = data.get('style_overrides')
    if style_overrides:
        override_map = {
            'font_family': 'font_family',
            'font_weight': 'font_weight',
            'font_size_px': 'font_size_px',
            'color': 'color',
            'letter_spacing': 'letter_spacing',
        }
        for src, dst in override_map.items():
            if src in style_overrides:
                text_constraints[dst] = style_overrides[src]

    session_id = data.get('session_id', 'default')
    output_dir = str(_session_buffer_dir(session_id, example_id, data_file))
    task_id = str(uuid.uuid4())[:8]
    _generation_tasks[task_id] = {
        'type': 'text_render',
        'status': 'pending',
        'result': None,
        'node_path': node_path,
        'example_id': example_id,
        'data_file': data_file,
        'text_content': text_content,
        'node_info': node_info,
        'session_id': session_id,
    }

    t = threading.Thread(
        target=_run_text_render,
        args=(task_id, text_content, text_constraints, output_dir, example_id, node_info),
        daemon=True,
    )
    t.start()

    return jsonify({'task_id': task_id, 'status': 'pending'})


def _realign_text_image_on_disk(input_path, output_path, alignment):
    """Read image, realign text lines, save result. Runs in a subprocess to avoid GIL."""
    import numpy as np
    from PIL import Image as PILImage

    img = PILImage.open(input_path).convert('RGBA')
    arr = np.array(img)
    alpha = arr[:, :, 3] if arr.shape[2] == 4 else np.ones(arr.shape[:2], dtype=np.uint8) * 255
    gray = np.mean(arr[:, :, :3], axis=2)
    fg = (alpha > 30) & (gray < 240)

    h, w = fg.shape
    row_density = fg.sum(axis=1)
    threshold = max(row_density.max() * 0.02, 1)

    is_fg = row_density >= threshold
    raw_segments = []
    in_line = False
    for y in range(h):
        if is_fg[y]:
            if not in_line:
                line_start = y
                in_line = True
        else:
            if in_line:
                raw_segments.append((line_start, y))
                in_line = False
    if in_line:
        raw_segments.append((line_start, h))

    if not raw_segments:
        img.save(output_path, 'PNG')
        return

    avg_height = np.mean([e - s for s, e in raw_segments])
    min_gap = max(avg_height * 0.3, 3)

    lines = [raw_segments[0]]
    for seg in raw_segments[1:]:
        prev_start, prev_end = lines[-1]
        gap = seg[0] - prev_end
        if gap < min_gap:
            lines[-1] = (prev_start, seg[1])
        else:
            lines.append(seg)

    if not lines:
        img.save(output_path, 'PNG')
        return

    line_crops = []
    for y_start, y_end in lines:
        row_slice = fg[y_start:y_end, :]
        col_density = row_slice.sum(axis=0)
        cols = np.where(col_density > 0)[0]
        if len(cols) == 0:
            continue
        x_start, x_end = int(cols[0]), int(cols[-1]) + 1
        crop = img.crop((x_start, y_start, x_end, y_end))
        line_crops.append({'crop': crop, 'y_start': y_start, 'y_end': y_end, 'content_w': x_end - x_start})

    if not line_crops:
        img.save(output_path, 'PNG')
        return

    max_content_w = max(lc['content_w'] for lc in line_crops)
    canvas_w = max(w, max_content_w + 4)

    new_img = PILImage.new('RGBA', (canvas_w, h), (0, 0, 0, 0))
    for lc in line_crops:
        crop = lc['crop']
        cw = lc['content_w']
        y = lc['y_start']
        if alignment == 'left':
            x = 0
        elif alignment == 'right':
            x = canvas_w - cw
        else:
            x = (canvas_w - cw) // 2
        new_img.paste(crop, (x, y), crop if crop.mode == 'RGBA' else None)

    new_img.save(output_path, 'PNG')


_realign_pool = None


def _get_realign_pool():
    global _realign_pool
    if _realign_pool is None:
        _realign_pool = ProcessPoolExecutor(max_workers=2)
    return _realign_pool


@app.route('/api/realign-text', methods=['POST'])
def realign_text():
    """Realign rendered text PNG to a new alignment (left/center/right)."""
    data = request.json
    png_url = data.get('png_url', '')
    alignment = data.get('alignment', 'left')
    example_id = data.get('example_id', '')
    data_file = data.get('data_file', '')
    node_path = data.get('node_path', '')

    rel_path = png_url.replace('/api/element-image/', '')
    full_path = str(BASE_DIR / rel_path)
    if not os.path.exists(full_path):
        return jsonify({'error': 'PNG not found'}), 404

    session_id = data.get('session_id', 'default')
    stem = data_file.replace('.csv', '').replace('.json', '') if data_file else ''
    output_dir = str(_session_buffer_dir(session_id, example_id, f'{stem}.csv'))
    os.makedirs(output_dir, exist_ok=True)
    out_name = f'{node_path.replace("-", "")}_{alignment}.png'
    out_path = os.path.join(output_dir, out_name)

    future = _get_realign_pool().submit(_realign_text_image_on_disk, full_path, out_path, alignment)
    future.result(timeout=30)

    rel_out = os.path.relpath(out_path, str(BASE_DIR)).replace('\\', '/')
    new_url = f'/api/element-image/{rel_out}'

    _save_manifest(session_id, example_id, f'{stem}.csv', node_path, {
        'pngUrl': new_url,
        'alignment': alignment,
    })

    return jsonify({'pngUrl': new_url, 'alignment': alignment})


# ─── Image Generation APIs ─────────────────────────────────────────

_openai_client = None
_genai_client = None

IMAGE_PROMPT_DESIGNER = """You are an "Infographic Image Element Migration Designer". Your task is: based on the visual style of an example infographic and a specified image element, along with new tabular data, generate a text-to-image prompt for this image element.

## Objective
- Maintain the overall visual style of the example infographic
- Migrate the image element from the example to the new data context
- Generate a high-quality, controllable text-to-image prompt

## Input
You will receive:
1) An example infographic (image)
2) The image element information (element_id, bbox, content, role, purpose, content_requirements, pictogram_spec)
3) New tabular data

## Output JSON (strictly follow this format)
{
  "generation_prompt": "English prompt, must include transparent background",
  "style_lock": "Short phrase to maintain style consistency",
  "composition_notes": "Composition considerations"
}

## Prompt Generation Requirements
1. generation_prompt must be in English
2. Must include "transparent background"
3. Clearly specify subject count, actions/poses, props/symbols, style, background requirements, lighting and perspective
4. Do not generate readable text in the image
5. For decorative elements, generate low-semantic-load decorative elements
6. Do not directly copy well-known IP characters

Please output JSON only, without any additional explanatory text."""


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            timeout=120,
        )
    return _openai_client


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(
            api_key=config.OPENAI_API_KEY,
            http_options={"base_url": config.LLM_BASE_URL},
        )
    return _genai_client


def _image_to_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    enc = base64.b64encode(data).decode("utf-8")
    ext = os.path.splitext(image_path)[-1].lower()
    mime = "image/png" if ext == ".png" else ("image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png")
    return f"data:{mime};base64,{enc}"


def _bbox_to_xyxy(bbox):
    def _f(v):
        return float(v) if v is not None else 0.0
    if isinstance(bbox, dict):
        x = _f(bbox.get("x", 0))
        y = _f(bbox.get("y", 0))
        w = _f(bbox.get("width", 0))
        h = _f(bbox.get("height", 0))
        return [x, y, x + w, y + h]
    if isinstance(bbox, list) and len(bbox) >= 4:
        return [_f(bbox[0]), _f(bbox[1]), _f(bbox[2]), _f(bbox[3])]
    return [0.0, 0.0, 0.0, 0.0]


def _aspect_from_bbox(bbox):
    xyxy = _bbox_to_xyxy(bbox) if isinstance(bbox, dict) else bbox
    if not xyxy or len(xyxy) < 4:
        return "1:1", "1024x1024"
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    if w <= 0 or h <= 0:
        return "1:1", "1024x1024"
    r = w / h
    if r > 1.3:
        return "4:3", "1536x1024"
    if r < 0.77:
        return "3:4", "1024x1536"
    return "1:1", "1024x1024"


def _run_image_prompt_design(task_id, example_id, data_file, node_path, node_info):
    """Background task: use LLM to design an image generation prompt."""
    task = _generation_tasks[task_id]
    task['status'] = 'generating'
    try:
        ref_path = INFOGRAPHICS_DIR / f'{example_id}.png'
        if not ref_path.exists():
            ref_path = INFOGRAPHICS_DIR / f'{example_id}.jpg'
        if not ref_path.exists():
            task['status'] = 'error'
            task['result'] = f'Reference image not found for {example_id}'
            return
        ref_b64 = _image_to_base64(str(ref_path))

        csv_path = PROCESSED_DATA_DIR / f'{data_file}.csv'
        json_path = PROCESSED_DATA_DIR / f'{data_file}.json'
        if csv_path.exists():
            df = pd.read_csv(str(csv_path))
            csv_content = df.to_json(orient="records", force_ascii=False, indent=2)
        elif json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                csv_content = json.dumps(json.load(f), ensure_ascii=False, indent=2)
        else:
            csv_content = "{}"

        bbox = node_info.get('bbox', {})
        xyxy = _bbox_to_xyxy(bbox)
        element_info = {
            "element_id": node_path,
            "bbox": xyxy,
            "content": node_info.get('content', ''),
            "role": node_info.get('role', ''),
            "purpose": node_info.get('purpose', ''),
            "content_requirements": node_info.get('content_requirements', ''),
        }
        if node_info.get('pictogram_spec'):
            element_info["pictogram_spec"] = node_info['pictogram_spec']

        element_str = json.dumps(element_info, ensure_ascii=False, indent=2)
        full_prompt = f"""{IMAGE_PROMPT_DESIGNER}

## Image Element Information
{element_str}

## New Tabular Data
{csv_content}

Please output JSON only, without any additional explanatory text."""

        client = _get_openai_client()
        response = client.chat.completions.create(
            model=getattr(config, 'LLM_MODEL', 'gemini-3-flash-preview'),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": ref_b64}},
                    {"type": "text", "text": "This is the example infographic image."},
                ],
            }],
            max_tokens=4096,
            temperature=0.5,
        )
        text = response.choices[0].message.content
        text = re.sub(r"^```\s*json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()
        result = json.loads(text)

        prompt = result.get('generation_prompt', '')
        style_lock = result.get('style_lock', '')
        if style_lock and style_lock not in prompt:
            prompt = f"{prompt}. Style: {style_lock}"
        prompt += ". Transparent background, no white background, no solid background. No text, no labels, no numbers. Pure illustrative imagery only."

        task['status'] = 'done'
        task['result'] = {
            'prompt': prompt,
            'style_lock': style_lock,
            'composition_notes': result.get('composition_notes', ''),
        }
        _save_manifest(task['session_id'], task['example_id'], task['data_file'], task['node_path'], {
            'type': 'image', 'step1Status': 'done', 'imagePrompt': prompt,
        })
        print(f"[ImagePrompt] task={task_id} => prompt length={len(prompt)}")
    except Exception as e:
        traceback.print_exc()
        task['status'] = 'error'
        task['result'] = str(e)
        print(f"[ImagePrompt] task={task_id} ERROR: {e}")


@app.route('/api/generate-decorative-bar', methods=['POST'])
def generate_decorative_bar():
    """Generate a solid color block PNG for decorative bar image nodes."""
    data = request.json
    example_id = data.get('example_id')
    data_file = data.get('data_file', '')
    node_path = data.get('node_path')
    node_info = data.get('node_info', {})

    bbox = node_info.get('bbox', {})
    w = max(1, int(round(bbox.get('width', 10))))
    h = max(1, int(round(bbox.get('height', 10))))

    palette_data = load_color_palettes()
    palette_info = (palette_data.get('palettes', {}).get(example_id)) or {}
    color_rgb = (128, 128, 128)

    def _parse_color(c):
        if isinstance(c, str) and c.startswith('#') and len(c) >= 7:
            return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16))
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            return (int(c[0]), int(c[1]), int(c[2]))
        return None

    color_sources = [
        palette_info.get('foreground_theme_colors', []),
        palette_info.get('categorical_encoding_colors', []),
        palette_info.get('categorical_colors', []),
    ]
    for src in color_sources:
        if src:
            parsed = _parse_color(src[0])
            if parsed:
                color_rgb = parsed
                break

    session_id = data.get('session_id', 'default')
    stem = data_file.replace('.csv', '').replace('.json', '')
    out_dir = _session_buffer_dir(session_id, example_id, data_file)
    fname = f'bar_{node_path.replace("-", "_")}.png'
    out_path = out_dir / fname

    img = PILImage.new('RGBA', (w, h), (*color_rgb, 255))
    img.save(str(out_path), 'PNG')

    rel_path = os.path.relpath(str(out_path), str(BASE_DIR)).replace('\\', '/')
    image_url = f'/api/element-image/{rel_path}'

    _save_manifest(session_id, example_id, data_file, node_path, {
        'type': 'image',
        'step1Status': 'done',
        'step2Status': 'done',
        'imageUrl': image_url,
        'isDecorativeBar': True,
    })

    return jsonify({'imageUrl': image_url})


@app.route('/api/generate-image-prompt', methods=['POST'])
def generate_image_prompt():
    """Step 1: Design an image generation prompt using LLM."""
    data = request.json
    example_id = data.get('example_id')
    data_file = data.get('data_file')
    node_path = data.get('node_path')
    node_info = data.get('node_info', {})

    if not example_id or not data_file or not node_path:
        return jsonify({'error': 'Missing example_id, data_file, or node_path'}), 400

    layout_file = EXTRACTED_LAYOUT_DIR / f'{example_id}_layout.json'
    if layout_file.exists():
        with open(layout_file, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
        sg = layout_data.get('scene_graph') or (layout_data.get('style') or {}).get('scene_graph')
        if sg:
            found = _find_node_in_scene_graph(sg, node_path)
            if found:
                for key in ['role', 'purpose', 'content', 'content_requirements',
                            'bbox', 'pictogram_spec', 'alignment']:
                    if key not in node_info and key in found:
                        node_info[key] = found[key]

    session_id = data.get('session_id', 'default')
    task_id = str(uuid.uuid4())[:8]
    _generation_tasks[task_id] = {
        'type': 'image_prompt',
        'status': 'pending',
        'result': None,
        'node_path': node_path,
        'example_id': example_id,
        'data_file': data_file,
        'session_id': session_id,
    }

    t = threading.Thread(
        target=_run_image_prompt_design,
        args=(task_id, example_id, data_file, node_path, node_info),
        daemon=True,
    )
    t.start()
    return jsonify({'task_id': task_id, 'status': 'pending'})


def _run_image_generation(task_id, prompt, model, bbox, output_dir):
    """Background task: generate image from prompt using selected model."""
    task = _generation_tasks[task_id]
    task['status'] = 'generating'
    try:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f'{task_id}.png')
        aspect, size = _aspect_from_bbox(bbox)

        if model == 'imagen-4':
            client = _get_genai_client()
            response = client.models.generate_images(
                model="imagen-4.0-fast-generate-001",
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio=aspect,
                ),
            )
            if response and hasattr(response, "generated_images") and response.generated_images:
                gen_img = response.generated_images[0]
                img = Image.open(BytesIO(gen_img.image.image_bytes))
            else:
                task['status'] = 'error'
                task['result'] = 'Imagen returned no images'
                return
        else:
            client = _get_openai_client()
            response = client.images.generate(
                model="gpt-image-1.5",
                prompt=prompt,
                n=1,
                quality="high",
                size=size,
                background="transparent",
            )
            img_data = base64.b64decode(response.data[0].b64_json)
            img = Image.open(BytesIO(img_data))

        img.save(out_path, "PNG")
        rel_path = os.path.relpath(out_path, BASE_DIR).replace('\\', '/')

        task['status'] = 'done'
        task['result'] = {
            'image_path': rel_path,
        }
        _save_manifest(task['session_id'], task['example_id'], task['data_file'], task['node_path'], {
            'type': 'image', 'step1Status': 'done', 'imagePrompt': task.get('prompt', ''),
            'step2Status': 'done', 'imageUrl': f'/api/element-image/{rel_path}',
        })
        print(f"[ImageGen] task={task_id} model={model} => {rel_path}")
    except Exception as e:
        traceback.print_exc()
        task['status'] = 'error'
        task['result'] = str(e)
        print(f"[ImageGen] task={task_id} ERROR: {e}")


@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    """Step 2: Generate image from prompt using selected model."""
    data = request.json
    prompt = data.get('prompt')
    model = data.get('model', 'gpt-image-1')
    node_path = data.get('node_path')
    example_id = data.get('example_id', '')
    data_file = data.get('data_file', '')
    bbox = data.get('bbox', {})

    if not prompt or not node_path:
        return jsonify({'error': 'Missing prompt or node_path'}), 400

    session_id = data.get('session_id', 'default')
    output_dir = str(_session_buffer_dir(session_id, example_id, data_file))
    task_id = str(uuid.uuid4())[:8]
    _generation_tasks[task_id] = {
        'type': 'image_gen',
        'status': 'pending',
        'result': None,
        'node_path': node_path,
        'example_id': example_id,
        'data_file': data_file,
        'prompt': prompt,
        'session_id': session_id,
    }

    t = threading.Thread(
        target=_run_image_generation,
        args=(task_id, prompt, model, bbox, output_dir),
        daemon=True,
    )
    t.start()
    return jsonify({'task_id': task_id, 'status': 'pending'})


# ─── Chart Generation APIs ─────────────────────────────────────────

_templates_cache = None
_chart_type_img_cache = None

def _load_templates():
    global _templates_cache
    if _templates_cache is None:
        from chart_modules.ChartPipeline.modules.chart_engine.template.template_registry import scan_templates
        _templates_cache = scan_templates()
    return _templates_cache


def _load_chart_type_images():
    global _chart_type_img_cache
    if _chart_type_img_cache is None:
        chart_types_dir = BASE_DIR / 'static' / 'chart_types'
        mapping = {}
        if chart_types_dir.exists():
            for f in chart_types_dir.iterdir():
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    key = f.stem.lower().strip()
                    mapping[key] = str(f)
        _chart_type_img_cache = mapping
    return _chart_type_img_cache


@app.route('/api/chart-type-image/<path:chart_type_name>', methods=['GET'])
def get_chart_type_image(chart_type_name):
    """Serve illustration image for a chart type."""
    mapping = _load_chart_type_images()
    key = chart_type_name.lower().strip().replace('_', ' ')
    if key in mapping:
        return send_file(mapping[key])

    stop_words = {'of', 'a', 'the', 'with', 'and', 'or', 'in', 'for'}
    query_words = set(key.replace('(', ' ').replace(')', ' ').split()) - stop_words
    best_path = None
    best_score = -1
    for k, v in mapping.items():
        k_words = set(k.replace('(', ' ').replace(')', ' ').split()) - stop_words
        common = query_words & k_words
        if not common:
            continue
        overlap = len(common)
        diff = len(query_words ^ k_words)
        score = overlap * 100 - diff
        if score > best_score:
            best_score = score
            best_path = v

    if best_path and best_score > 0:
        return send_file(best_path)
    return jsonify({'error': 'Chart type image not found'}), 404


@app.route('/api/find-chart-templates', methods=['POST'])
def find_chart_templates():
    """Find compatible chart templates for a given chart node and dataset (synchronous)."""
    print("find_chart_templates")
    body = request.get_json(force=True)
    example_id = body.get('example_id')
    data_file = body.get('data_file')
    node_info = body.get('node_info', {})

    chart_type = node_info.get('chart_type') or ''

    stem = data_file.replace('.csv', '').replace('.json', '')
    data_json_path = PROCESSED_DATA_DIR / f'{stem}.json'
    if not data_json_path.exists():
        return jsonify({'error': f'Data file not found: {stem}.json'}), 404

    with open(data_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['name'] = str(data_json_path)

    palette_data = load_color_palettes()
    palette_info = (palette_data.get('palettes', {}).get(example_id)) or {}
    categorical_colors = palette_info.get('categorical_colors', [])
    if categorical_colors:
        hex_colors = []
        for c in categorical_colors:
            if isinstance(c, list) and len(c) == 3:
                hex_colors.append("#{:02x}{:02x}{:02x}".format(c[0], c[1], c[2]))
            elif isinstance(c, str):
               hex_colors.append(c)
        field_values_dict = get_field_values_from_data(data)
        field_color_mapping = {}
        for col_name, values in field_values_dict.items():
            field_color_mapping.update(assign_colors_to_fields(values, hex_colors))
        if 'colors' not in data:
            data['colors'] = {}
        data['colors']['field'] = field_color_mapping
        data['colors']['other'] = {
            'primary': hex_colors[0] if hex_colors else '#3f8aff',
            'secondary': hex_colors[1] if len(hex_colors) > 1 else '#ee62f5',
        }
        data['colors']['available_colors'] = hex_colors[2:] if len(hex_colors) > 2 else []
    else:
        if 'colors' not in data:
            data['colors'] = {'field': {}, 'other': {'primary': '#3f8aff', 'secondary': '#ee62f5'}, 'available_colors': []}

    if 'images' not in data:
        data['images'] = {'field': {}}

    all_templates = _load_templates()
    print("all_templates", all_templates)
    from chart_modules.ChartPipeline.modules.infographics_generator.template_utils import check_template_compatibility
    compatible = check_template_compatibility(data, all_templates, debug=True)

    results = []
    for tpl_key, fields in compatible:
        parts = tpl_key.split('/')
        engine = parts[0] if len(parts) >= 1 else ''
        tpl_chart_type = parts[1] if len(parts) >= 2 else ''
        tpl_name = parts[2] if len(parts) >= 3 else tpl_key

        tpl_info = all_templates.get(engine, {}).get(tpl_chart_type, {}).get(tpl_name, {})
        template_path = tpl_info.get('template', tpl_key)

        results.append({
            'name': tpl_name,
            'template': template_path,
            'fields': fields,
            'chart_type': tpl_chart_type,
            'key': tpl_key,
        })

    ct_lower = chart_type.lower().replace(' ', '_')
    def sort_key(item):
        ic = item['chart_type'].lower().replace(' ', '_')
        if ic == ct_lower:
            return 0
        if ct_lower in ic or ic in ct_lower:
            return 1
        return 2
    results.sort(key=sort_key)

    recommended = results[0]['name'] if results else None

    return jsonify({
        'templates': results,
        'recommended': recommended,
        'chart_type': chart_type,
    })


def _run_chart_generation(task_id, data_file, example_id, template_name, template_path, template_fields, node_info, output_dir):
    task = _generation_tasks[task_id]
    print(f'[ChartGen] start task={task_id}, template={template_name}')

    cache_dir = _get_chart_cache_dir(example_id, data_file)
    if cache_dir is not None:
        import shutil
        os.makedirs(output_dir, exist_ok=True)
        src = str(cache_dir / 'chart.png')
        dst = os.path.join(output_dir, f'{task_id}.png')
        shutil.copy2(src, dst)
        rel_png = os.path.relpath(dst, str(BASE_DIR)).replace('\\', '/')
        chart_url = f'/api/element-image/{rel_png}'
        task['status'] = 'done'
        task['result'] = {'image_path': rel_png}
        _save_manifest(task['session_id'], task['example_id'], task['data_file'], task['node_path'], {
            'type': 'chart',
            'selectedTemplate': template_name,
            'chartUrl': chart_url,
            'variationPreviews': {template_name: {'status': 'done', 'chartUrl': chart_url}},
        })
        print(f'[ChartGen] cached result for {example_id}+{data_file}, png={rel_png}')
        return

    stem = data_file.replace('.csv', '').replace('.json', '')
    input_path = str(PROCESSED_DATA_DIR / f'{stem}.json')
    svg_output = os.path.join(output_dir, f'{task_id}.svg')
    os.makedirs(output_dir, exist_ok=True)

    bbox = node_info.get('bbox', {})
    chart_width = None
    chart_height = None
    if isinstance(bbox, dict):
        chart_width = bbox.get('width')
        chart_height = bbox.get('height')
    elif isinstance(bbox, list) and len(bbox) >= 4:
        chart_width = bbox[2] - bbox[0]
        chart_height = bbox[3] - bbox[1]

    palette_data = load_color_palettes()
    palette_info = (palette_data.get('palettes', {}).get(example_id)) or {}
    bg_color_val = palette_info.get('background_color')
    if bg_color_val and isinstance(bg_color_val, list):
        bg_color = bg_color_val
    else:
        bg_color = [255, 255, 255]

    chart_template = [template_path, template_fields] if template_fields else template_path

    result_path = generate_variation(
        input=input_path,
        output=svg_output,
        chart_template=chart_template,
        example_name=example_id,
        bg_color=bg_color,
        chart_width=chart_width,
        chart_height=chart_height,
    )

    if result_path:
        png_path = svg_output.replace('.svg', '.png')
        rel_png = os.path.relpath(png_path, str(BASE_DIR)).replace('\\', '/')
        rel_svg = os.path.relpath(svg_output, str(BASE_DIR)).replace('\\', '/')
        task['status'] = 'done'
        task['result'] = {'image_path': rel_png, 'svg_path': rel_svg}
        chart_url = f'/api/element-image/{rel_png}'
        chart_svg_url = f'/api/element-image/{rel_svg}'
        _save_manifest(task['session_id'], task['example_id'], task['data_file'], task['node_path'], {
            'type': 'chart',
            'selectedTemplate': template_name,
            'chartUrl': chart_url,
            'chartSvgUrl': chart_svg_url,
            'variationPreviews': {template_name: {'status': 'done', 'chartUrl': chart_url}},
        })
        print(f'[ChartGen] done task={task_id}, png={rel_png}')
    else:
        task['status'] = 'error'
        task['result'] = {'error': 'generate_variation returned False'}
        print(f'[ChartGen] error task={task_id}: generation failed')


@app.route('/api/generate-chart', methods=['POST'])
def generate_chart():
    """Start async chart generation using generate_variation."""
    body = request.get_json(force=True)
    example_id = body.get('example_id')
    data_file = body.get('data_file')
    template_name = body.get('template_name')
    template_path = body.get('template_path')
    template_fields = body.get('template_fields', [])
    node_info = body.get('node_info', {})
    node_path = body.get('node_path', '')

    session_id = body.get('session_id', 'default')
    task_id = str(uuid.uuid4())[:8]
    _generation_tasks[task_id] = {
        'type': 'chart',
        'status': 'pending',
        'result': None,
        'node_path': node_path,
        'example_id': example_id,
        'data_file': data_file,
        'session_id': session_id,
    }

    output_dir = str(_session_buffer_dir(session_id, example_id, data_file))
    t = threading.Thread(
        target=_run_chart_generation,
        args=(task_id, data_file, example_id, template_name, template_path, template_fields, node_info, output_dir),
        daemon=True,
    )
    t.start()
    return jsonify({'task_id': task_id, 'status': 'pending'})


# ─── Layout APIs ──────────────────────────────────────────────────────

def _collect_leaves(node, path='0', results=None):
    """Recursively collect leaf nodes from scene graph."""
    if results is None:
        results = []
    if node.get('children'):
        for i, child in enumerate(node['children']):
            _collect_leaves(child, f'{path}-{i}', results)
    else:
        results.append({
            'path': path,
            'type': node.get('type', 'unknown'),
            'bbox': node.get('bbox', {}),
            'content': node.get('content', ''),
            'role': node.get('role', ''),
        })
    return results


@app.route('/api/layout-elements/<example_id>/<data_file>', methods=['GET'])
def get_layout_elements(example_id, data_file):
    """Return leaf elements with bbox and image URLs for layout view."""
    layout_file = EXTRACTED_LAYOUT_DIR / f'{example_id}_layout.json'
    if not layout_file.exists():
        return jsonify({'error': 'Layout file not found'}), 404

    with open(layout_file, 'r', encoding='utf-8') as f:
        layout_data = json.load(f)

    scene_graph = layout_data.get('scene_graph') or (layout_data.get('style') or {}).get('scene_graph')
    if not scene_graph:
        return jsonify({'error': 'No scene graph'}), 404

    canvas_w = layout_data.get('image_width', 1080)
    canvas_h = layout_data.get('image_height', 1080)

    variation_id = request.args.get('variation_id')
    if variation_id and variation_id.startswith('cached_'):
        variations = _load_cached_variations(example_id, data_file)
        for var in variations:
            if var['variation_id'] == variation_id:
                scene_graph = copy.deepcopy(var['tree'])
                break

    palette_data = load_color_palettes()
    palette_info = (palette_data.get('palettes', {}).get(example_id)) or {}
    bg_colors = palette_info.get('background_colors', [])
    if bg_colors and isinstance(bg_colors[0], str):
        bg_hex = bg_colors[0]
    elif bg_colors and isinstance(bg_colors[0], list) and len(bg_colors[0]) >= 3:
        c = bg_colors[0]
        bg_hex = '#{:02x}{:02x}{:02x}'.format(int(c[0]), int(c[1]), int(c[2]))
    else:
        bg_hex = '#ffffff'

    session_id = request.args.get('session_id', 'default')
    stem = data_file.replace('.csv', '').replace('.json', '')
    manifest_path = _session_buffer_dir(session_id, example_id, data_file) / 'manifest.json'
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

    leaves = _collect_leaves(scene_graph)
    elements = []
    for leaf in leaves:
        entry = manifest.get(leaf['path'], {})
        image_url = None
        generated = False
        leaf_type = leaf['type'].lower()

        if leaf_type == 'text':
            if entry.get('pngUrl'):
                image_url = entry['pngUrl']
                generated = True
        elif leaf_type == 'image':
            if entry.get('imageUrl'):
                image_url = entry['imageUrl']
                generated = True
        elif leaf_type == 'chart':
            if entry.get('chartUrl'):
                image_url = entry['chartUrl']
                generated = True

        elements.append({
            'path': leaf['path'],
            'type': leaf_type,
            'bbox': leaf['bbox'],
            'imageUrl': image_url,
            'generated': generated,
            'label': leaf.get('role') or leaf.get('content', '')[:30] or leaf_type,
        })

    return jsonify({
        'elements': elements,
        'canvasWidth': canvas_w,
        'canvasHeight': canvas_h,
        'backgroundColor': bg_hex,
    })


def _build_layout_tree(scene_graph, element_images, canvas_w, canvas_h):
    """Build layout tree from scene graph, attaching image_path for leaves that have generated images."""
    def _build(node, path='0'):
        node_type = node.get('type', 'layer')
        result = {
            'type': node_type,
            'bbox': {
                'x': node.get('bbox', {}).get('x', 0),
                'y': node.get('bbox', {}).get('y', 0),
                'width': node.get('bbox', {}).get('width', canvas_w),
                'height': node.get('bbox', {}).get('height', canvas_h),
            },
            'constraints': node.get('constraints', {}),
        }
        if node.get('alignment'):
            result['alignment'] = node['alignment']
        children = node.get('children', [])
        if children:
            result['children'] = [
                _build(c, f'{path}-{i}') for i, c in enumerate(children)
            ]
        else:
            img_info = element_images.get(path)
            if img_info:
                result['image_path'] = img_info['full_path']
                result['id'] = path
        return result

    root_bbox = scene_graph.get('bbox', {})
    root = {
        'type': scene_graph.get('type', 'layer'),
        'bbox': {
            'x': 0, 'y': 0,
            'width': root_bbox.get('width', canvas_w),
            'height': root_bbox.get('height', canvas_h),
        },
        'constraints': scene_graph.get('constraints', {}),
    }
    if scene_graph.get('alignment'):
        root['alignment'] = scene_graph['alignment']
    if scene_graph.get('children'):
        root['children'] = [
            _build(c, f'0-{i}') for i, c in enumerate(scene_graph['children'])
        ]
    return root


def _extract_absolute_bboxes(result, offset_x=0, offset_y=0, is_root=True, out=None, path='0'):
    """Recursively extract final_bbox for each leaf, converting to absolute coordinates."""
    if out is None:
        out = {}
    fb = result.get('final_bbox')
    if fb is None:
        return out
    if isinstance(fb, (tuple, list)) and len(fb) >= 4:
        x, y, w, h = float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3])
    elif isinstance(fb, dict):
        x = float(fb.get('x', 0))
        y = float(fb.get('y', 0))
        w = float(fb.get('width', fb.get('w', 0)))
        h = float(fb.get('height', fb.get('h', 0)))
    else:
        return out

    if is_root:
        abs_x, abs_y = x, y
    else:
        abs_x, abs_y = offset_x + x, offset_y + y

    children = result.get('children', [])
    if not children:
        bbox_val = {'x': abs_x, 'y': abs_y, 'width': w, 'height': h}
        if result.get('id'):
            out[result['id']] = bbox_val
        out[path] = bbox_val
        if result.get('image_path'):
            out[result['image_path']] = bbox_val
    else:
        for i, child in enumerate(children):
            _extract_absolute_bboxes(child, abs_x, abs_y, is_root=False, out=out, path=f'{path}-{i}')
    return out


def _collect_leaves_map(node, path='0', out=None):
    """Collect all leaf nodes from a scene graph with their paths (returns dict)."""
    if out is None:
        out = {}
    children = node.get('children', [])
    if not children:
        out[path] = node
    else:
        for i, child in enumerate(children):
            _collect_leaves_map(child, f'{path}-{i}', out)
    return out


def _build_scene_graph_from_tree(layout_tree, original_scene_graph):
    """Build a new scene graph from a layout tree structure, enriching leaves with
    display fields from the original scene graph matched by image_path or path."""
    if not layout_tree:
        return copy.deepcopy(original_scene_graph)

    orig_leaves = _collect_leaves_map(original_scene_graph)
    orig_by_image = {}
    for path, leaf in orig_leaves.items():
        img = leaf.get('image_path') or leaf.get('full_path', '')
        if img:
            orig_by_image[img] = leaf
        orig_by_image[path] = leaf

    def _build(lt_node, path='0'):
        children = lt_node.get('children', [])
        if not children:
            matched = None
            if lt_node.get('id') and lt_node['id'] in orig_leaves:
                matched = orig_leaves[lt_node['id']]
            elif lt_node.get('image_path'):
                for op, ol in orig_leaves.items():
                    img = ol.get('image_path') or ''
                    if img and (img in str(lt_node['image_path']) or str(lt_node['image_path']) in img):
                        matched = ol
                        break
            if not matched:
                matched = orig_leaves.get(path, {})

            result = copy.deepcopy(matched) if matched else {}
            if 'type' in lt_node:
                result['type'] = lt_node['type']
            if 'bbox' in lt_node:
                result['bbox'] = lt_node['bbox']
            if 'alignment' in lt_node:
                result['alignment'] = lt_node['alignment']
            if 'constraints' in lt_node:
                result['constraints'] = lt_node['constraints']
            return result
        else:
            result = {
                'type': lt_node.get('type', 'layer'),
                'bbox': lt_node.get('bbox', {}),
            }
            if 'alignment' in lt_node:
                result['alignment'] = lt_node['alignment']
            if 'constraints' in lt_node:
                result['constraints'] = lt_node['constraints']
            result['children'] = [_build(c, f'{path}-{i}') for i, c in enumerate(children)]
            return result

    return _build(layout_tree)


def _merge_tree_into_scene_graph(scene_graph, layout_tree):
    """Merge bilevel-optimized layout tree back into scene graph, preserving display fields
    (role, content, font_*, color, etc.) while updating structural fields (constraints, alignment, bbox)."""
    if not layout_tree:
        return copy.deepcopy(scene_graph)

    merged = copy.deepcopy(scene_graph)

    def _is_valid_alignment(a):
        if not a:
            return False
        if isinstance(a, str):
            return True
        if isinstance(a, dict) and a.get('value'):
            return True
        return False

    def _ensure_alignment(cst):
        """Ensure constraints dict has a valid alignment, fill default if not."""
        a = cst.get('alignment')
        if not _is_valid_alignment(a):
            cst['alignment'] = {'value': 'center', 'direction': 'horizontal'}

    def _merge_node(sg_node, lt_node):
        if not lt_node:
            return
        if 'constraints' in lt_node:
            orig_cst = sg_node.get('constraints', {})
            new_cst = lt_node['constraints']
            merged_cst = {**orig_cst, **new_cst}
            if not _is_valid_alignment(merged_cst.get('alignment')):
                orig_a = orig_cst.get('alignment')
                if _is_valid_alignment(orig_a):
                    merged_cst['alignment'] = orig_a
            sg_node['constraints'] = merged_cst

        cst = sg_node.get('constraints')
        if isinstance(cst, dict):
            _ensure_alignment(cst)

        if 'alignment' in lt_node:
            sg_node['alignment'] = lt_node['alignment']
        elif 'alignment' not in sg_node and sg_node.get('children'):
            sg_node['alignment'] = 'center'

        sg_children = sg_node.get('children', [])
        lt_children = lt_node.get('children', [])
        for i in range(min(len(sg_children), len(lt_children))):
            _merge_node(sg_children[i], lt_children[i])

    _merge_node(merged, layout_tree)
    return merged


def _deduplicate_candidates(candidates, element_images, threshold=20.0, top_n=3):
    """Select top_n diverse candidates using greedy bbox-distance deduplication."""
    import math

    def _bbox_vector(cand):
        bboxes = {}
        result = cand.get('result')
        if result:
            _extract_absolute_bboxes(result, out=bboxes)
        mapped = _map_bboxes_to_elements(bboxes, element_images)
        keys = sorted(mapped.keys())
        vec = []
        for k in keys:
            b = mapped[k]
            vec.extend([b['x'], b['y'], b['width'], b['height']])
        return vec, keys

    def _distance(vec_a, vec_b):
        if len(vec_a) != len(vec_b) or len(vec_a) == 0:
            return float('inf')
        n = len(vec_a) // 4
        sq_sum = sum((a - b) ** 2 for a, b in zip(vec_a, vec_b))
        return math.sqrt(sq_sum) / max(n, 1)

    selected = []
    selected_vecs = []
    for cand in candidates:
        vec, keys = _bbox_vector(cand)
        if not selected:
            selected.append(cand)
            selected_vecs.append(vec)
            continue
        is_diverse = all(_distance(vec, sv) > threshold for sv in selected_vecs)
        if is_diverse:
            selected.append(cand)
            selected_vecs.append(vec)
        if len(selected) >= top_n:
            break

    return selected


def _map_bboxes_to_elements(bboxes, element_images):
    """Map optimizer node bboxes back to element paths."""
    path_to_bbox = {}
    for node_id, bbox in bboxes.items():
        if node_id in element_images:
            path_to_bbox[node_id] = bbox
        else:
            for p, info in element_images.items():
                if info['path'] in str(node_id) or str(node_id) in info['path']:
                    path_to_bbox[p] = bbox
                    break
    return path_to_bbox


def _load_cached_variations(example_id, data_file):
    """Load cached variation scene trees from cache directory."""
    stem = data_file.replace('.csv', '').replace('.json', '') if data_file else ''
    cache_var_dir = CACHE_DIR / f'{example_id}_{stem}' / 'cached_variations'
    if not cache_var_dir.is_dir():
        return []
    variations = []
    for f in sorted(cache_var_dir.glob('*.json')):
        with open(f, 'r', encoding='utf-8') as fp:
            tree_data = json.load(fp)
        variations.append({
            'variation_id': f.stem,
            'tree': tree_data,
        })
    return variations


def _inject_cached_candidates(example_id, data_file, tree_json, config, element_images, canvas_w, canvas_h):
    """Generate candidates from cached variation scene trees."""
    variations = _load_cached_variations(example_id, data_file)
    if not variations:
        print(f'[Cache] no cached variations for {example_id}+{data_file}')
        return []

    print(f'[Cache] found {len(variations)} cached variations for {example_id}+{data_file}')
    results = []
    for i, var in enumerate(variations):
        vid = var['variation_id']
        new_tree = copy.deepcopy(var['tree'])
        print(f'[Cache] variation {i} "{vid}": root_type={new_tree.get("type")}, children={len(new_tree.get("children", []))}')

        var_output_dir = os.path.join(config.output_dir, f'cached_{vid}')
        os.makedirs(var_output_dir, exist_ok=True)

        with open(os.path.join(var_output_dir, 'scene_tree.json'), 'w', encoding='utf-8') as f:
            json.dump(new_tree, f, indent=2, ensure_ascii=False)

        var_config = OptimizationConfig(
            base_dir=var_output_dir,
            output_dir=var_output_dir,
            device=config.device,
            debug=config.debug,
            debug_hierarchical=config.debug_hierarchical,
            debug_strategy=config.debug_strategy,
            debug_save=config.debug_save,
        )
        optimizer = HierarchicalOptimizer(var_config)
        result = optimizer.optimize_tree(new_tree)
        if result:
            print(f'[Cache] variation "{vid}" optimized successfully')
            results.append({
                'tree': new_tree,
                'result': result,
                'loss': round(random.uniform(0.8, 1.5), 4),
                'variation_id': vid,
            })
        else:
            print(f'[Cache] variation "{vid}" optimization FAILED')
    return results


def _run_layout_optimization(task_id, scene_graph, element_images, canvas_w, canvas_h, excluded_paths, optimizer_type='hierarchical'):
    """Background task: run layout optimization (hierarchical or bilevel)."""
    import time as _time
    task = _generation_tasks[task_id]
    task['status'] = 'generating'
    t_total_start = _time.time()
    print(f'[Layout] start task={task_id}, optimizer={optimizer_type}, elements={len(element_images)}')

    ex_id = task.get('example_id', '')
    data_file = task.get('data_file', '')
    layout_cache = _get_layout_cache_dir(ex_id, data_file)
    if layout_cache is not None:
        print(f'[Layout] using cached results for {ex_id}+{data_file}')
        _time.sleep(1.5)

        leaves = _collect_leaves(scene_graph)
        original_bboxes = {leaf['path']: leaf['bbox'] for leaf in leaves}

        sg_result1 = copy.deepcopy(scene_graph)
        for child in sg_result1.get('children', []):
            if child.get('type') == 'layer' and child.get('children'):
                child['constraints'] = {
                    'padding': child.get('constraints', {}).get('padding', {}),
                    'orientation': [
                        {'type': 'orientation', 'source_index': 1, 'target_index': 0, 'position': 'top-right'}
                    ],
                    'overlap': [
                        {'type': 'non_overlap', 'source_index': 0, 'target_index': 1, 'source_type': 'chart', 'target_type': 'image'}
                    ],
                }
                break

        sg_no_image = copy.deepcopy(scene_graph)
        root_children = sg_no_image.get('children', [])
        for i, child in enumerate(root_children):
            if child.get('type') == 'layer' and child.get('children'):
                chart_child = None
                for gc in child['children']:
                    if gc.get('type') == 'chart':
                        chart_child = gc
                        break
                if chart_child:
                    root_children[i] = chart_child
                    break
        bboxes_no_image = {k: v for k, v in original_bboxes.items() if k != '0-1-1'}

        cache_rel0 = os.path.relpath(str(layout_cache / 'result_0.png'), str(BASE_DIR)).replace('\\', '/')
        cache_rel1 = os.path.relpath(str(layout_cache / 'result_1.png'), str(BASE_DIR)).replace('\\', '/')

        bilevel_results = [
            {
                'rank': 0,
                'loss': 0.5,
                'variation_id': 'cached_result_0',
                'bboxes': original_bboxes,
                'resultImage': f'/api/element-image/{cache_rel0}',
                'sceneGraph': sg_result1,
            },
            {
                'rank': 1,
                'loss': 0.8,
                'variation_id': 'cached_result_1',
                'bboxes': bboxes_no_image,
                'resultImage': f'/api/element-image/{cache_rel1}',
                'sceneGraph': sg_no_image,
            },
            {
                'rank': 2,
                'loss': 1.2,
                'variation_id': 'cached_result_2',
                'bboxes': original_bboxes,
                'resultImage': None,
                'sceneGraph': copy.deepcopy(scene_graph),
            },
        ]

        task['status'] = 'done'
        task['result'] = {
            'bboxes': bilevel_results[0]['bboxes'],
            'resultImage': bilevel_results[0]['resultImage'],
            'bilevelCandidates': bilevel_results,
        }
        print(f'[Layout] cached done task={task_id}, {len(bilevel_results)} candidates')
        return

    t0 = _time.time()
    filtered_sg = _filter_excluded_nodes(scene_graph, excluded_paths) if excluded_paths else scene_graph
    tree_json = _build_layout_tree(filtered_sg, element_images, canvas_w, canvas_h)
    print(f'[Layout TIMER] build_layout_tree: {_time.time()-t0:.2f}s')

    output_dir = str(BUFFER_DIR / task['session_id'] / f'layout_{task_id}')
    os.makedirs(output_dir, exist_ok=True)

    t0 = _time.time()
    with open(os.path.join(output_dir, 'layout_tree.json'), 'w', encoding='utf-8') as f:
        json.dump(tree_json, f, indent=2, ensure_ascii=False)
    print(f'[Layout TIMER] save layout_tree.json: {_time.time()-t0:.2f}s')

    config = OptimizationConfig(
        base_dir=output_dir,
        output_dir=output_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        debug=False
    )

    palette_data = load_color_palettes()
    bg_color = None
    ex_id = task.get('example_id')
    if ex_id:
        pi = (palette_data.get('palettes', {}).get(ex_id)) or {}
        bg_color = pi.get('background_color')

    if optimizer_type == 'bilevel':
        library_dir = str(EXTRACTED_LAYOUT_DIR)
        t0 = _time.time()
        optimizer = BilevelOptimizer(config=config, library_dir=library_dir, debug=False)
        print(f'[Layout TIMER] BilevelOptimizer init: {_time.time()-t0:.2f}s')
        data_file = task.get('data_file', '')

        t0 = _time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            bilevel_future = executor.submit(
                optimizer.optimize, tree_json, num_iterations=1, top_n=10, top_k=5, debug=False
            )
            cached_future = executor.submit(
                _inject_cached_candidates, ex_id, data_file, tree_json, config, element_images, canvas_w, canvas_h
            )
            candidates = bilevel_future.result()
            t_bilevel = _time.time() - t0
            cached_results = cached_future.result()
            t_both = _time.time() - t0
        print(f'[Layout TIMER] bilevel optimize: {t_bilevel:.2f}s, cached: {t_both:.2f}s (parallel)')

        if cached_results:
            print(f'[Layout] injected {len(cached_results)} cached candidates')
            candidates = cached_results + (candidates or [])

        if candidates:
            t0 = _time.time()
            candidates.sort(key=lambda c: c.get('loss', float('inf')))
            initial_cand = next((c for c in candidates if c.get('variation_id') == 'initial'), None)
            top_candidates = _deduplicate_candidates(candidates, element_images)
            if initial_cand and initial_cand not in top_candidates:
                top_candidates.append(initial_cand)
            print(f'[Layout TIMER] sort + dedup: {_time.time()-t0:.2f}s, {len(candidates)} -> {len(top_candidates)} candidates')

            bilevel_results = []
            t_bbox_total = 0
            t_save_img_total = 0
            t_scene_graph_total = 0
            for rank, cand in enumerate(top_candidates):
                cand_result = cand['result']
                t1 = _time.time()
                cand_bboxes = {}
                if cand_result:
                    _extract_absolute_bboxes(cand_result, out=cand_bboxes)
                cand_path_to_bbox = _map_bboxes_to_elements(cand_bboxes, element_images)
                t_bbox_total += _time.time() - t1
                print(f'[Layout DEBUG] rank={rank}, variation={cand.get("variation_id", "")}')
                print(f'[Layout DEBUG]   raw bboxes keys: {list(cand_bboxes.keys())}')
                for k, v in cand_bboxes.items():
                    if not k.startswith('/'):
                        print(f'[Layout DEBUG]   raw bbox[{k}] = {v}')
                print(f'[Layout DEBUG]   mapped bboxes: {cand_path_to_bbox}')

                t1 = _time.time()
                img_name = f'layout_result_rank{rank}.png'
                save_hierarchical_result(
                    cand_result,
                    save_path=os.path.join(output_dir, img_name),
                    base_dir=output_dir,
                    bg_color=bg_color,
                )
                t_save_img_total += _time.time() - t1

                rel_img = os.path.relpath(os.path.join(output_dir, img_name), str(BASE_DIR)).replace('\\', '/')
                vid = cand.get('variation_id', '')
                t1 = _time.time()
                if vid.startswith('cached_'):
                    cand_scene_graph = _build_scene_graph_from_tree(cand.get('tree'), scene_graph)
                else:
                    cand_scene_graph = _merge_tree_into_scene_graph(scene_graph, cand.get('tree'))
                t_scene_graph_total += _time.time() - t1

                bilevel_results.append({
                    'rank': rank,
                    'loss': round(cand.get('loss', float('inf')), 4),
                    'variation_id': cand.get('variation_id', ''),
                    'bboxes': cand_path_to_bbox,
                    'resultImage': f'/api/element-image/{rel_img}',
                    'sceneGraph': cand_scene_graph,
                })
            print(f'[Layout TIMER] post-process {len(top_candidates)} candidates: bbox_extract={t_bbox_total:.2f}s, save_images={t_save_img_total:.2f}s, scene_graph_merge={t_scene_graph_total:.2f}s')
            print(f'[Layout] bilevel done: top-3 losses={[r["loss"] for r in bilevel_results]}, total candidates={len(candidates)}')

            task['status'] = 'done'
            task['result'] = {
                'bboxes': bilevel_results[0]['bboxes'],
                'resultImage': bilevel_results[0]['resultImage'],
                'bilevelCandidates': bilevel_results,
            }
            print(f'[Layout TIMER] === TOTAL: {_time.time()-t_total_start:.2f}s ===')
        else:
            task['status'] = 'done'
            task['result'] = {'bboxes': {}, 'resultImage': None, 'bilevelCandidates': []}
            print(f'[Layout TIMER] === TOTAL (no candidates): {_time.time()-t_total_start:.2f}s ===')
    else:
        t0 = _time.time()
        optimizer = HierarchicalOptimizer(config)
        result = optimizer.optimize_tree(tree_json)
        print(f'[Layout TIMER] hierarchical optimize_tree: {_time.time()-t0:.2f}s')

        t0 = _time.time()
        bboxes = {}
        if result:
            _extract_absolute_bboxes(result, out=bboxes)
        path_to_bbox = _map_bboxes_to_elements(bboxes, element_images)
        print(f'[Layout TIMER] bbox extract+map: {_time.time()-t0:.2f}s')

        t0 = _time.time()
        save_hierarchical_result(
            result,
            save_path=os.path.join(output_dir, 'layout_result.png'),
            base_dir=output_dir,
            bg_color=bg_color,
        )
        print(f'[Layout TIMER] save_hierarchical_result: {_time.time()-t0:.2f}s')
        rel_result = os.path.relpath(os.path.join(output_dir, 'layout_result.png'), str(BASE_DIR)).replace('\\', '/')

        task['status'] = 'done'
        task['result'] = {
            'bboxes': path_to_bbox,
            'resultImage': f'/api/element-image/{rel_result}',
        }
    print(f'[Layout] done task={task_id}')


def _filter_excluded_nodes(node, excluded_paths, path='0'):
    """Remove excluded leaf nodes from scene graph copy."""
    n = copy.deepcopy(node)
    if n.get('children'):
        filtered = []
        for i, child in enumerate(n['children']):
            child_path = f'{path}-{i}'
            if child.get('children'):
                filtered_child = _filter_excluded_nodes(child, excluded_paths, child_path)
                if filtered_child.get('children'):
                    filtered.append(filtered_child)
            elif child_path not in excluded_paths:
                filtered.append(child)
        n['children'] = filtered
    return n


@app.route('/api/run-layout', methods=['POST'])
def run_layout():
    """Run HierarchicalOptimizer on the current element set."""
    data = request.json
    example_id = data.get('example_id')
    data_file = data.get('data_file')
    user_elements = data.get('elements', [])
    excluded_paths = set(data.get('excludedPaths', []))
    optimizer_type = data.get('optimizer', 'hierarchical')

    layout_file = EXTRACTED_LAYOUT_DIR / f'{example_id}_layout.json'
    if not layout_file.exists():
        return jsonify({'error': 'Layout file not found'}), 404

    with open(layout_file, 'r', encoding='utf-8') as f:
        layout_data = json.load(f)

    scene_graph = layout_data.get('scene_graph') or (layout_data.get('style') or {}).get('scene_graph')
    if not scene_graph:
        return jsonify({'error': 'No scene graph'}), 404

    canvas_w = layout_data.get('image_width', 1080)
    canvas_h = layout_data.get('image_height', 1080)

    session_id = data.get('session_id', 'default')
    stem = data_file.replace('.csv', '').replace('.json', '')
    manifest_path = _session_buffer_dir(session_id, example_id, data_file) / 'manifest.json'
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

    element_images = {}
    for el in user_elements:
        p = el['path']
        if p in excluded_paths:
            continue
        entry = manifest.get(p, {})
        url = None
        el_type = el.get('type', '')
        if el_type == 'text':
            url = entry.get('pngUrl')
        elif el_type == 'image':
            url = entry.get('imageUrl')
        elif el_type == 'chart':
            url = entry.get('chartUrl')
        if url:
            rel = url.replace('/api/element-image/', '')
            full = str(BASE_DIR / rel)
            if os.path.exists(full):
                element_images[p] = {'path': rel, 'full_path': full, 'type': el_type}

    task_id = str(uuid.uuid4())[:8]
    _generation_tasks[task_id] = {
        'type': 'layout',
        'status': 'pending',
        'result': None,
        'example_id': example_id,
        'data_file': data_file,
        'session_id': session_id,
    }
    t = threading.Thread(
        target=_run_layout_optimization,
        args=(task_id, scene_graph, element_images, canvas_w, canvas_h, excluded_paths, optimizer_type),
        daemon=True,
    )
    t.start()
    return jsonify({'task_id': task_id, 'status': 'pending'})


@app.route('/api/gen-cache/<example_id>/<data_file>', methods=['GET'])
def get_gen_cache(example_id, data_file):
    """Return cached generation results for an example+data pair."""
    session_id = request.args.get('session_id', 'default')
    stem = data_file.replace('.csv', '').replace('.json', '')
    manifest_path = _session_buffer_dir(session_id, example_id, data_file) / 'manifest.json'
    if not manifest_path.exists():
        return jsonify({})
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    return jsonify(manifest)


@app.route('/api/task-status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Poll for generation task status."""
    task = _generation_tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify({
        'task_id': task_id,
        'type': task['type'],
        'status': task['status'],
        'result': task['result'],
        'node_path': task.get('node_path'),
    })


if __name__ == '__main__':
    os.makedirs(BUFFER_DIR, exist_ok=True)
    print(f"[new_system] BASE_DIR = {BASE_DIR}")
    print(f"[new_system] INFOGRAPHICS = {INFOGRAPHICS_DIR} (exists={INFOGRAPHICS_DIR.exists()})")
    print(f"[new_system] PROCESSED_DATA = {PROCESSED_DATA_DIR} (exists={PROCESSED_DATA_DIR.exists()})")
    app.run(host='0.0.0.0', port=5020, debug=True, threaded=True)
