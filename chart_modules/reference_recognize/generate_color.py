import numpy as np
if not hasattr(np, "asscalar"):
    np.asscalar = lambda x: x.item()

import random
from sklearn.cluster import KMeans
from colorsys import rgb_to_hls, hls_to_rgb
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_conversions import convert_color
import matplotlib.pyplot as plt

def visualize_palette(palette, bg_color=None, title="Color Palette"):
    """
    Display a color palette with an optional background color.
    
    Args:
        palette (List[List[int]]): List of colors in RGB format
        bg_color (List[int] or None): Background color (for canvas background)
        title (str): Image title
    """
    n = len(palette)
    fig, ax = plt.subplots(figsize=(n * 1.2, 1.8))
    
    # Set background color
    if bg_color:
        fig.patch.set_facecolor(np.array(bg_color) / 255)
        ax.set_facecolor(np.array(bg_color) / 255)

    for i, color in enumerate(palette):
        rgb = np.array(color) / 255
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=rgb))
        ax.text(i + 0.5, -0.3, f"{color}", ha="center", va="center", fontsize=9)

    ax.set_xlim(0, n)
    ax.set_ylim(-0.5, 1)
    ax.axis('off')
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig("palette_preview.png")


def rgb_to_lab(rgb):
    """Convert RGB to Lab color."""
    color_rgb = sRGBColor(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    return convert_color(color_rgb, LabColor)


def color_distance(c1, c2):
    """Calculate perceptual distance between two colors (Lab space)."""
    return delta_e_cie2000(rgb_to_lab(c1), rgb_to_lab(c2))



def is_distinct(color, others, threshold=20, bg_color=None):
    """Check if a color is sufficiently distinct from other colors and the background."""
    for other in others:
        if color_distance(color, other) < threshold:
            return False
    if bg_color and color_distance(color, bg_color) < threshold:
        return False
    return True


def perturb_color(color, amount=15):
    """Slightly perturb a color."""
    return [min(255, max(0, c + random.randint(-amount, amount))) for c in color]


def generate_new_colors(n, avoid_colors, bg_color):
    """Generate new colors that are highly distinct from existing colors and the background."""
    result = []
    attempts = 0
    while len(result) < n and attempts < 500:
        candidate = [random.randint(0, 255) for _ in range(3)]
        if is_distinct(candidate, result + avoid_colors, bg_color=bg_color):
            result.append(candidate)
        attempts += 1
    return result

def rgb_to_hex(rgb_list):
    """
    Convert an RGB list to a hexadecimal color code.
    :param rgb_list: List of three integers [R, G, B], e.g. [202, 150, 229]
    :return: Hexadecimal color string, e.g. "#ca96e5"
    """
    if len(rgb_list) != 3:
        raise ValueError("RGB list must contain exactly 3 elements [R, G, B]")
    
    r, g, b = rgb_list
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def generate_distinct_palette(data, main_colors, bg_color, num_colors=5):
    """
    Generate a color palette ensuring mutual distinction, overall harmony, and clear contrast with the background.
    
    Args:
        main_colors (List[List[int]]): List of main colors, each as an RGB list
        bg_color (List[int]): Background color in RGB
        num_colors (int): Target palette size
        
    Returns:
        List[List[int]]: List of highly distinct colors
    """
    main_colors = [list(map(int, c)) for c in main_colors]
    selected = []
    num_colors = len(data["colors"]["field"])

    # 1. Prioritize sufficiently distinct main colors
    for color in main_colors:
        if is_distinct(color, selected, bg_color=bg_color):
            selected.append(color)
        else:
            adjusted = perturb_color(color)
            if is_distinct(adjusted, selected, bg_color=bg_color):
                selected.append(adjusted)

        if len(selected) >= num_colors:
            selected =  selected[:num_colors]
            
            for i, field in enumerate(data["colors"]["field"].keys()):
                data["colors"]["field"][field] = rgb_to_hex(selected[i])
            
            for i, field in enumerate(data["colors_dark"]["field"].keys()):
                data["colors_dark"]["field"][field] = rgb_to_hex(selected[i])
            
            return data

    # 2. If not enough, generate additional colors
    remaining = num_colors - len(selected)
    new_colors = generate_new_colors(remaining, selected, bg_color)
    selected.extend(new_colors)

    for i, field in enumerate(data["colors"]["field"].keys()):
        data["colors"]["field"][field] = rgb_to_hex(selected[i])
    
    for i, field in enumerate(data["colors_dark"]["field"].keys()):
        data["colors_dark"]["field"][field] = rgb_to_hex(selected[i])
    print("data:",data)
    return data


# Example usage
if __name__ == "__main__":
    main_colors = [[25, 33, 24], [234, 232, 216], [243, 228, 146], [100, 110, 99], [171, 172, 148]]
    bg_color = [255, 255, 255]
    palette = generate_distinct_palette(main_colors, bg_color, num_colors=6)

    print("Generated palette:", palette)
    visualize_palette({}, palette, bg_color, title="Generated Distinct Color Palette")