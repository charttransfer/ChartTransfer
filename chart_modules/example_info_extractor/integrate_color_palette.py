#!/usr/bin/env python3
"""
Integrate color palettes from examples into the chart generation workflow.

This script provides utilities to:
1. Map categorical_encoding_colors to field values
2. Generate color assignments for data fields based on example palettes
3. Update processed_data JSON files with palette-based colors
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

COLOR_PALETTES_FILE = project_root / "color_palettes.json"
PROCESSED_DATA_DIR = project_root / "processed_data"


def load_color_palettes():
    """Load all color palettes from color_palettes.json."""
    with open(COLOR_PALETTES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_categorical_colors_for_example(example_name, color_palettes=None):
    """
    Get categorical encoding colors for a specific example.
    
    Args:
        example_name: Name/ID of the example
        color_palettes: Pre-loaded color palettes dict (optional)
        
    Returns:
        List of categorical encoding colors, or empty list if not found
    """
    if color_palettes is None:
        color_palettes = load_color_palettes()
    
    # Color palettes structure: {"palettes": {...}, "stats": {...}}
    palettes = color_palettes.get("palettes", {})
    
    if example_name in palettes:
        return palettes[example_name].get("categorical_encoding_colors", [])
    
    return []


def assign_colors_to_fields(field_values, categorical_colors):
    """
    Assign colors from categorical_encoding_colors to field values.
    
    Args:
        field_values: List of unique field values (e.g., ['iOS App Store', 'Google Play Store'])
        categorical_colors: List of colors from example palette
        
    Returns:
        Dictionary mapping field values to colors
    """
    if not categorical_colors:
        return {}
    
    color_mapping = {}
    for i, value in enumerate(field_values):
        # Cycle through colors if there are more values than colors
        color_index = i % len(categorical_colors)
        color_mapping[value] = categorical_colors[color_index]
    
    return color_mapping


def get_field_values_from_data(data_json):
    """
    Extract unique values from categorical columns in the data.
    
    Args:
        data_json: Parsed JSON data
        
    Returns:
        Dictionary mapping column names to lists of unique values
    """
    columns = data_json.get("data", {}).get("columns", [])
    data_rows = data_json.get("data", {}).get("data", [])
    
    field_values = {}
    
    for col in columns:
        if col.get("data_type") == "categorical":
            col_name = col["name"]
            unique_values = set()
            
            for row in data_rows:
                if col_name in row:
                    unique_values.add(row[col_name])
            
            field_values[col_name] = sorted(list(unique_values))
    
    return field_values


def create_example_based_colors_section(example_name, data_json, color_palettes=None):
    """
    Create a colors section based on example palette.
    
    Args:
        example_name: Name/ID of the example to use
        data_json: Parsed JSON data
        color_palettes: Pre-loaded color palettes dict (optional)
        
    Returns:
        Dictionary with colors configuration
    """
    # Get full palette for other color information
    if color_palettes is None:
        color_palettes = load_color_palettes()
    
    palettes = color_palettes.get("palettes", {})
    example_palette = palettes.get(example_name, {})
    
    # Get categorical colors from example
    categorical_colors = get_categorical_colors_for_example(example_name, color_palettes)
    
    # Get data_mark_color
    data_mark_color = example_palette.get("data_mark_color")
    
    # If neither categorical_colors nor data_mark_color is available, return None
    if not categorical_colors and not data_mark_color:
        print(f"Warning: No categorical colors or data_mark_color found for example '{example_name}'")
        return None
    
    # Get field values from data
    field_values_dict = get_field_values_from_data(data_json)
    
    # Create color mappings for each categorical field
    field_color_mapping = {}
    if categorical_colors:
        for col_name, values in field_values_dict.items():
            field_color_mapping.update(assign_colors_to_fields(values, categorical_colors))
    
    print("example_palette", example_palette)
    # Determine primary color: prioritize data_mark_color if available
    if data_mark_color:
        primary_color = data_mark_color
        secondary_color = categorical_colors[0] if categorical_colors else "#ee62f5"
        available_colors = categorical_colors[1:] if len(categorical_colors) > 1 else []
    else:
        primary_color = categorical_colors[0] if categorical_colors else "#3f8aff"
        secondary_color = categorical_colors[1] if len(categorical_colors) > 1 else "#ee62f5"
        available_colors = categorical_colors[2:] if len(categorical_colors) > 2 else []
    
    # Create colors section
    colors_section = {
        "field": field_color_mapping,
        "other": {
            "primary": primary_color,
            "secondary": secondary_color
        },
        "available_colors": available_colors,
        "background_color": example_palette.get("background_colors", ["#FFFFFF"])[0],
        "text_color": example_palette.get("foreground_theme_colors", ["#000000"])[0]
    }
    
    return colors_section


def update_data_file_with_example_colors(json_path, example_name, save_backup=True):
    """
    Update a data JSON file with colors from a specific example.
    
    Args:
        json_path: Path to the data JSON file
        example_name: Name/ID of the example to use
        save_backup: Whether to save a backup of the original file
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\nUpdating: {json_path.name}")
    print(f"  Using example: {example_name}")
    
    # Load data JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    
    # Create backup if requested
    if save_backup:
        backup_path = json_path.with_suffix('.json.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data_json, f, indent=2)
        print(f"  Backup saved: {backup_path.name}")
    
    # Load color palettes
    color_palettes = load_color_palettes()
    
    # Create new colors section
    new_colors = create_example_based_colors_section(example_name, data_json, color_palettes)
    
    if new_colors is None:
        print(f"  Failed: Could not create colors from example")
        return False
    
    # Update colors section
    data_json["colors"] = new_colors
    
    # Add metadata about color source
    if "color_metadata" not in data_json:
        data_json["color_metadata"] = {}
    
    data_json["color_metadata"]["source"] = "example_palette"
    data_json["color_metadata"]["example_name"] = example_name
    
    # Save updated JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data_json, f, indent=2)
    
    print(f"  ✓ Updated successfully")
    print(f"    Field colors: {len(new_colors['field'])} mappings")
    print(f"    Background: {new_colors['background_color']}")
    
    return True


def demo_color_assignment():
    """Demonstrate color assignment for all data files."""
    print("=" * 80)
    print("Color Palette Integration Demo")
    print("=" * 80)
    
    # Load color palettes
    color_palettes = load_color_palettes()
    palettes = color_palettes.get("palettes", {})
    
    # Get a sample example
    if not palettes:
        print("No color palettes found!")
        return
    
    sample_example = list(palettes.keys())[0]
    print(f"\nUsing sample example: {sample_example}")
    
    # Load a sample data file
    json_files = list(PROCESSED_DATA_DIR.glob("*.json"))
    if not json_files:
        print("No data files found!")
        return
    
    sample_data_file = json_files[0]
    print(f"Sample data file: {sample_data_file.name}\n")
    
    # Load data
    with open(sample_data_file, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    
    # Get field values
    field_values_dict = get_field_values_from_data(data_json)
    print(f"Categorical fields found: {len(field_values_dict)}")
    for col_name, values in field_values_dict.items():
        print(f"  {col_name}: {values}")
    
    # Get categorical colors
    categorical_colors = get_categorical_colors_for_example(sample_example, color_palettes)
    print(f"\nCategorical colors from example: {categorical_colors}")
    
    # Create color assignments
    print("\nColor assignments:")
    for col_name, values in field_values_dict.items():
        color_mapping = assign_colors_to_fields(values, categorical_colors)
        for value, color in color_mapping.items():
            print(f"  {value}: {color}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Integrate example color palettes into data files"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a demonstration of color assignment"
    )
    parser.add_argument(
        "--update",
        type=str,
        metavar="DATA_FILE",
        help="Update a specific data file (e.g., App.json)"
    )
    parser.add_argument(
        "--example",
        type=str,
        metavar="EXAMPLE_NAME",
        help="Example name/ID to use for colors"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        demo_color_assignment()
    elif args.update:
        if not args.example:
            print("Error: --example is required when using --update")
            return
        
        json_path = PROCESSED_DATA_DIR / args.update
        if not json_path.exists():
            print(f"Error: File not found: {json_path}")
            return
        
        update_data_file_with_example_colors(
            json_path,
            args.example,
            save_backup=not args.no_backup
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

