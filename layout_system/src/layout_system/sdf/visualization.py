"""Visualization functions for SDF-based layout optimization."""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import torch

from .core import make_container_grid, sdf_to_softmask


# -----------------------------
# Visualize SDF normalization and softmask conversion
# -----------------------------
def visualize_sdf_norm_and_softmask(sdf_norm: np.ndarray, mask: np.ndarray,
                                    bbox: tuple = None, container_size: tuple = None,
                                    tau_values: list = [0.5, 1.0, 1.5, 2.0],
                                    save_path: str = None):
    """
    Visualize SDF normalization and sdf_to_softmask conversion process.
    
    Args:
        sdf_norm: Normalized SDF array [H, W]
        mask: Original binary mask [H, W]
        bbox: Optional bounding box (x, y, w, h) for softmask visualization
        container_size: Optional container size (W, H) for softmask visualization
        tau_values: List of tau values to visualize for softmask conversion
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Row 1: SDF normalization visualization
    # Original mask
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(mask, cmap='gray', interpolation='bilinear')
    ax1.set_title('Original Binary Mask', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Width')
    ax1.set_ylabel('Height')
    
    # SDF normalized - full range
    ax2 = plt.subplot(3, 4, 2)
    im2 = ax2.imshow(sdf_norm, cmap='RdYlBu', interpolation='bilinear')
    ax2.contour(sdf_norm, levels=[0], colors='black', linewidths=2)
    ax2.set_title(f'SDF Normalized (Full Range)\n[{sdf_norm.min():.3f}, {sdf_norm.max():.3f}]', 
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Width')
    ax2.set_ylabel('Height')
    plt.colorbar(im2, ax=ax2, label='SDF Value')
    
    # SDF normalized - zoomed range around zero
    ax3 = plt.subplot(3, 4, 3)
    sdf_range = 0.1
    vmin, vmax = -sdf_range, sdf_range
    im3 = ax3.imshow(sdf_norm, cmap='RdYlBu', interpolation='bilinear', vmin=vmin, vmax=vmax)
    ax3.contour(sdf_norm, levels=[0], colors='black', linewidths=2)
    ax3.contour(sdf_norm, levels=np.linspace(-sdf_range, sdf_range, 11), 
                colors='gray', linewidths=0.5, alpha=0.3)
    ax3.set_title(f'SDF Normalized (Zoomed)\nRange: [-{sdf_range}, {sdf_range}]', 
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel('Width')
    ax3.set_ylabel('Height')
    plt.colorbar(im3, ax=ax3, label='SDF Value')
    
    # SDF histogram
    ax4 = plt.subplot(3, 4, 4)
    ax4.hist(sdf_norm.flatten(), bins=100, alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero level')
    ax4.set_title('SDF Value Distribution', fontsize=11, fontweight='bold')
    ax4.set_xlabel('SDF Value')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Row 2-3: Softmask conversion with different tau values
    if bbox is not None and container_size is not None:
        x, y, w, h = bbox
        Wc, Hc = container_size
        
        # Convert to torch tensors
        sdf_norm_t = torch.from_numpy(sdf_norm)[None, None].float()
        device = sdf_norm_t.device
        
        # Create container grid
        X, Y = make_container_grid(Hc, Wc, device=device)
        
        # Visualize softmask for different tau values
        # Layout: Row 2-3, each row has 2 tau values, each tau has 2 subplots (mask + distance)
        for idx, tau_px in enumerate(tau_values):
            # Row: 1 (idx 0-1) or 2 (idx 2-3)
            row = 1 + idx // 2
            # Column: 1-2 (idx 0) or 3-4 (idx 1) for row 1, 1-2 (idx 2) or 3-4 (idx 3) for row 2
            col_offset = (idx % 2) * 2  # 0 or 2
            
            # Convert bbox values to torch tensors for sdf_to_softmask
            x_t = torch.tensor(x, device=device, dtype=torch.float32)
            y_t = torch.tensor(y, device=device, dtype=torch.float32)
            w_t = torch.tensor(w, device=device, dtype=torch.float32)
            h_t = torch.tensor(h, device=device, dtype=torch.float32)
            
            # Compute softmask
            m, d_px = sdf_to_softmask(sdf_norm_t, x_t, y_t, w_t, h_t, X, Y, tau_px=tau_px)
            m_np = m.squeeze().cpu().numpy()
            d_px_np = d_px.squeeze().cpu().numpy()
            
            # Softmask visualization - position: row*4 + col_offset + 1
            ax_mask = plt.subplot(3, 4, row * 4 + col_offset + 1)
            im_mask = ax_mask.imshow(m_np, cmap='viridis', interpolation='bilinear', vmin=0, vmax=1)
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax_mask.add_patch(rect)
            ax_mask.set_title(f'Softmask (τ={tau_px:.1f}px)', fontsize=11, fontweight='bold')
            ax_mask.set_xlabel('Width')
            ax_mask.set_ylabel('Height')
            ax_mask.set_xlim(0, Wc)
            ax_mask.set_ylim(Hc, 0)
            plt.colorbar(im_mask, ax=ax_mask, label='Mask Value')
            
            # Distance visualization - position: row*4 + col_offset + 2
            ax_dist = plt.subplot(3, 4, row * 4 + col_offset + 2)
            d_range = 5.0  # Show distance range
            im_dist = ax_dist.imshow(d_px_np, cmap='coolwarm', interpolation='bilinear', 
                                     vmin=-d_range, vmax=d_range)
            ax_dist.contour(d_px_np, levels=[0], colors='black', linewidths=2)
            rect_dist = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax_dist.add_patch(rect_dist)
            ax_dist.set_title(f'Distance (τ={tau_px:.1f}px)\nRange: [-{d_range}, {d_range}]px', 
                             fontsize=11, fontweight='bold')
            ax_dist.set_xlabel('Width')
            ax_dist.set_ylabel('Height')
            ax_dist.set_xlim(0, Wc)
            ax_dist.set_ylim(Hc, 0)
            plt.colorbar(im_dist, ax=ax_dist, label='Distance (px)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"SDF norm and softmask visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# -----------------------------
# Visualize SDF
# -----------------------------
def visualize_sdf(sdf1: np.ndarray, sdf2: np.ndarray, 
                  mask1: np.ndarray = None, mask2: np.ndarray = None,
                  save_path: str = None, sdf_range: float = 0.1):
    """
    Visualize two SDFs side by side with heatmaps and zero contours.
    
    Args:
        sdf1: First SDF array [H, W]
        sdf2: Second SDF array [H, W]
        mask1: Optional original mask for first image
        mask2: Optional original mask for second image
        save_path: Optional path to save the figure
        sdf_range: Range of SDF values to display around zero (default 0.1)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # SDF 1 visualization - limit range for finer detail
    ax1 = axes[0, 0]
    vmin1, vmax1 = -sdf_range, sdf_range
    im1 = ax1.imshow(sdf1, cmap='RdYlBu', interpolation='bilinear', 
                     vmin=vmin1, vmax=vmax1)
    # Add multiple contour lines for detail
    ax1.contour(sdf1, levels=[0], colors='black', linewidths=2)
    ax1.contour(sdf1, levels=np.linspace(-sdf_range, sdf_range, 11), 
                colors='gray', linewidths=0.5, alpha=0.3)
    ax1.set_title('SDF 1 (chart.png)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Width')
    ax1.set_ylabel('Height')
    plt.colorbar(im1, ax=ax1, label='SDF Value')
    
    # SDF 2 visualization - limit range for finer detail
    ax2 = axes[0, 1]
    vmin2, vmax2 = -sdf_range, sdf_range
    im2 = ax2.imshow(sdf2, cmap='RdYlBu', interpolation='bilinear',
                     vmin=vmin2, vmax=vmax2)
    # Add multiple contour lines for detail
    ax2.contour(sdf2, levels=[0], colors='black', linewidths=2)
    ax2.contour(sdf2, levels=np.linspace(-sdf_range, sdf_range, 11), 
                colors='gray', linewidths=0.5, alpha=0.3)
    ax2.set_title('SDF 2 (pictogram.png)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Width')
    ax2.set_ylabel('Height')
    plt.colorbar(im2, ax=ax2, label='SDF Value')
    
    # Original masks if provided
    if mask1 is not None:
        ax3 = axes[1, 0]
        ax3.imshow(mask1, cmap='gray', interpolation='bilinear')
        ax3.set_title('Original Mask 1', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Width')
        ax3.set_ylabel('Height')
    
    if mask2 is not None:
        ax4 = axes[1, 1]
        ax4.imshow(mask2, cmap='gray', interpolation='bilinear')
        ax4.set_title('Original Mask 2', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Width')
        ax4.set_ylabel('Height')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"SDF visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# -----------------------------
# Visualize optimization progress
# -----------------------------
def visualize_optimization_progress(
    softmasks: List[torch.Tensor],
    bboxes: List[Tuple[float, float, float, float]],
    Wc: int, Hc: int,
    epoch: int = -1,
    loss_info: Dict[str, float] = None,
    save_path: str = None,
    gradients: List[Dict[str, float]] = None,
    debug: bool = False
):
    """
    Visualize optimization progress showing current layout and loss information.
    
    Args:
        softmasks: List of soft masks for each node [1,1,H,W]
        bboxes: List of bounding boxes (x, y, w, h) for each node
        Wc: Container width
        Hc: Container height
        epoch: Epoch number (-1 for initial, >=0 for epoch number)
        loss_info: Dictionary with loss values
        save_path: Path to save the figure
        gradients: List of gradient dicts for each node (keys: 'tx', 'ty', 'ts')
    """
    num_nodes = len(softmasks)
    
    # Convert tensors to numpy
    masks_np = []
    for m in softmasks:
        masks_np.append(m.squeeze().cpu().numpy())
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Main layout visualization (left side)
    ax_main = plt.subplot(1, 2, 1)
    
    # Create combined visualization
    combined = np.zeros((Hc, Wc, 3))
    colors = plt.cm.tab10(np.linspace(0, 1, num_nodes))
    
    # Compute union mask for visual balance calculation
    union_mask = np.ones((Hc, Wc))
    for mask_np in masks_np:
        union_mask = union_mask * (1.0 - mask_np)
    union_mask = 1.0 - union_mask
    
    for i, (mask_np, bbox) in enumerate(zip(masks_np, bboxes)):
        x, y, w, h = bbox
        # Use different colors for each node
        combined[:, :, 0] += mask_np * colors[i][0]  # Red channel
        combined[:, :, 1] += mask_np * colors[i][1]  # Green channel
        combined[:, :, 2] += mask_np * colors[i][2]  # Blue channel
        
        # Draw bbox rectangle
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=colors[i], 
                            facecolor='none', linestyle='--')
        ax_main.add_patch(rect)
    
    # Normalize combined image
    combined = np.clip(combined, 0, 1)
    ax_main.imshow(combined, interpolation='bilinear')
    ax_main.set_xlim(0, Wc)
    ax_main.set_ylim(Hc, 0)
    ax_main.set_xlabel('Width (px)', fontsize=12)
    ax_main.set_ylabel('Height (px)', fontsize=12)
    
    title = "Initial Layout" if epoch < 0 else f"Epoch {epoch}"
    ax_main.set_title(title, fontsize=14, fontweight='bold')
    
    # Add bbox labels
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        ax_main.text(x + w/2, y + h/2, f"Node {i+1}", 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # Visual balance visualization: show centroid and container center
    total_mass = union_mask.sum()
    if total_mass > 1e-8:
        # Calculate centroid
        x_coords = np.arange(Wc)
        y_coords = np.arange(Hc)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        centroid_x = (union_mask * x_grid).sum() / total_mass
        centroid_y = (union_mask * y_grid).sum() / total_mass
        
        # Container center
        center_x = Wc / 2.0
        center_y = Hc / 2.0
        
        # Draw container center (green cross)
        ax_main.plot(center_x, center_y, 'g+', markersize=15, markeredgewidth=3, 
                    label='Container Center', zorder=10)
        
        # Draw centroid (red circle)
        ax_main.plot(centroid_x, centroid_y, 'ro', markersize=10, markeredgewidth=2,
                    label='Centroid', zorder=10)
        
        # Draw line connecting centroid to center
        ax_main.plot([centroid_x, center_x], [centroid_y, center_y], 
                    'r--', linewidth=2, alpha=0.7, label='Balance Distance', zorder=9)
        
        # Add distance annotation
        distance = np.sqrt((centroid_x - center_x)**2 + (centroid_y - center_y)**2)
        mid_x = (centroid_x + center_x) / 2
        mid_y = (centroid_y + center_y) / 2
        ax_main.annotate(f'd={distance:.1f}px', 
                        xy=(mid_x, mid_y), xytext=(5, 5), textcoords='offset points',
                        fontsize=9, color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax_main.legend(loc='upper right', fontsize=9)
    
    # Loss information (right side)
    ax_info = plt.subplot(1, 2, 2)
    ax_info.axis('off')
    
    # Build loss info text
    info_lines = []
    info_lines.append("Optimization Progress")
    info_lines.append("=" * 30)
    info_lines.append("")
    
    if epoch >= 0:
        info_lines.append(f"Epoch: {epoch}")
    else:
        info_lines.append("Stage: Initial")
    
    info_lines.append("")
    info_lines.append("Layout Information:")
    info_lines.append(f"  Container: {Wc} x {Hc} px")
    info_lines.append(f"  Number of nodes: {num_nodes}")
    info_lines.append("")
    info_lines.append("Node Bounding Boxes:")
    info_lines.append("-" * 30)
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        info_lines.append(f"  Node {i+1}:")
        info_lines.append(f"    x: {x:.2f} px")
        info_lines.append(f"    y: {y:.2f} px")
        info_lines.append(f"    w: {w:.2f} px")
        info_lines.append(f"    h: {h:.2f} px")
        info_lines.append("")
    
    if loss_info:
        info_lines.append("Loss Values:")
        info_lines.append("-" * 30)
        
        if 'total' in loss_info:
            info_lines.append(f"  Total Loss: {loss_info['total']:.6f}")
        
        if 'A_union' in loss_info:
            info_lines.append(f"  Union Area: {loss_info['A_union']:.2f} px²")
        
        if 'A_inter' in loss_info:
            info_lines.append(f"  Intersection: {loss_info['A_inter']:.6f} px²")
        
        info_lines.append("")
        info_lines.append("Loss Components:")
        info_lines.append("-" * 30)
        
        if 'pen' in loss_info:
            info_lines.append(f"  Penetration: {loss_info['pen']:.6f}")
        
        if 'similarity' in loss_info:
            info_lines.append(f"  Similarity: {loss_info['similarity']:.6f}")
        
        if 'readability' in loss_info:
            info_lines.append(f"  Readability: {loss_info['readability']:.6f}")
        
        if 'alignment' in loss_info:
            info_lines.append(f"  Alignment: {loss_info['alignment']:.6f}")
        
        if 'proximity' in loss_info:
            info_lines.append(f"  Proximity: {loss_info['proximity']:.6f}")
        
        if 'data_ink' in loss_info:
            info_lines.append(f"  Data Ink: {loss_info['data_ink']:.6f}")
        
        if 'visual_balance' in loss_info:
            info_lines.append(f"  Visual Balance: {loss_info['visual_balance']:.6f}")
    
    if gradients:
        info_lines.append("")
        info_lines.append("Gradients:")
        info_lines.append("-" * 30)
        for i, grad in enumerate(gradients):
            info_lines.append(f"  Node {i+1}:")
            info_lines.append(f"    ∂L/∂tx: {grad['tx']:+.4e}")
            info_lines.append(f"    ∂L/∂ty: {grad['ty']:+.4e}")
            info_lines.append(f"    ∂L/∂ts: {grad['ts']:+.4e}")
            grad_mag = (grad['tx']**2 + grad['ty']**2 + grad['ts']**2)**0.5
            info_lines.append(f"    |∇L|: {grad_mag:.4e}")
    
    # Display text
    info_text = "\n".join(info_lines)
    ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if debug:
            print(f"Optimization progress visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# -----------------------------
# Visualize final layout result
# -----------------------------
def visualize_final_result(m1f: torch.Tensor, m2f: torch.Tensor, 
                           d1f_px: torch.Tensor, d2f_px: torch.Tensor,
                           bbox1: tuple, bbox2: tuple,
                           mask1: np.ndarray, mask2: np.ndarray,
                           Wc: int, Hc: int,
                           save_path: str = None, debug: bool = False):
    """
    Visualize the final optimization result showing layout, masks, and overlap.
    
    Args:
        m1f: Final soft mask for object 1 [1,1,H,W]
        m2f: Final soft mask for object 2 [1,1,H,W]
        d1f_px: Final SDF distance for object 1 [1,1,H,W]
        d2f_px: Final SDF distance for object 2 [1,1,H,W]
        bbox1: Bounding box (x, y, w, h) for object 1
        bbox2: Bounding box (x, y, w, h) for object 2
        mask1: Original mask for object 1
        mask2: Original mask for object 2
        Wc: Container width
        Hc: Container height
        save_path: Optional path to save the figure
    """
    # Convert tensors to numpy
    m1_np = m1f.squeeze().cpu().numpy()
    m2_np = m2f.squeeze().cpu().numpy()
    d1_np = d1f_px.squeeze().cpu().numpy()
    d2_np = d2f_px.squeeze().cpu().numpy()
    
    # Compute union and intersection
    union = 1.0 - (1.0 - m1_np) * (1.0 - m2_np)
    inter = m1_np * m2_np
    hard_overlap = ((d1_np < 0.0) & (d2_np < 0.0)).astype(np.float32)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Individual masks
    ax1 = axes[0, 0]
    ax1.imshow(m1_np, cmap='gray', interpolation='bilinear')
    x1, y1, w1, h1 = bbox1
    rect1 = plt.Rectangle((x1, y1), w1, h1, linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect1)
    ax1.set_title(f'Object 1 Mask\nbbox: ({x1:.1f}, {y1:.1f}, {w1:.1f}, {h1:.1f})', 
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('Width')
    ax1.set_ylabel('Height')
    ax1.set_xlim(0, Wc)
    ax1.set_ylim(Hc, 0)
    
    ax2 = axes[0, 1]
    ax2.imshow(m2_np, cmap='gray', interpolation='bilinear')
    x2, y2, w2, h2 = bbox2
    rect2 = plt.Rectangle((x2, y2), w2, h2, linewidth=2, edgecolor='blue', facecolor='none')
    ax2.add_patch(rect2)
    ax2.set_title(f'Object 2 Mask\nbbox: ({x2:.1f}, {y2:.1f}, {w2:.1f}, {h2:.1f})', 
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Width')
    ax2.set_ylabel('Height')
    ax2.set_xlim(0, Wc)
    ax2.set_ylim(Hc, 0)
    
    # Combined view
    ax3 = axes[0, 2]
    combined = np.zeros((Hc, Wc, 3))
    combined[:, :, 0] = m1_np  # Red channel for object 1
    combined[:, :, 2] = m2_np  # Blue channel for object 2
    combined[:, :, 1] = inter  # Green channel for intersection
    ax3.imshow(combined, interpolation='bilinear')
    rect1_comb = plt.Rectangle((x1, y1), w1, h1, linewidth=2, edgecolor='red', 
                               facecolor='none', linestyle='--')
    rect2_comb = plt.Rectangle((x2, y2), w2, h2, linewidth=2, edgecolor='blue', 
                               facecolor='none', linestyle='--')
    ax3.add_patch(rect1_comb)
    ax3.add_patch(rect2_comb)
    ax3.set_title('Combined Layout\n(Red: Obj1, Blue: Obj2, Green: Overlap)', 
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel('Width')
    ax3.set_ylabel('Height')
    ax3.set_xlim(0, Wc)
    ax3.set_ylim(Hc, 0)
    
    # Row 2: Union, Intersection, and Hard Overlap
    ax4 = axes[1, 0]
    im4 = ax4.imshow(union, cmap='viridis', interpolation='bilinear')
    ax4.set_title('Union Area', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Width')
    ax4.set_ylabel('Height')
    plt.colorbar(im4, ax=ax4, label='Union Value')
    
    ax5 = axes[1, 1]
    im5 = ax5.imshow(inter, cmap='hot', interpolation='bilinear')
    ax5.set_title('Intersection Area (Soft)', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Width')
    ax5.set_ylabel('Height')
    plt.colorbar(im5, ax=ax5, label='Intersection Value')
    
    ax6 = axes[1, 2]
    im6 = ax6.imshow(hard_overlap, cmap='Reds', interpolation='bilinear')
    ax6.set_title('Hard Overlap (SDF < 0)', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Width')
    ax6.set_ylabel('Height')
    plt.colorbar(im6, ax=ax6, label='Overlap')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if debug:
            print(f"Final result visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# -----------------------------
# Save composite image with original images placed according to optimized layout
# -----------------------------
def save_composite_image(png1: str = None, png2: str = None, bbox1: tuple = None, bbox2: tuple = None,
                        png_list: List[str] = None, bbox_list: List[tuple] = None,
                        Wc: int = 1000, Hc: int = 1000, save_path: str = "composite_result.png", debug: bool = False):
    """
    Load original PNG images and composite them according to optimized bbox positions.
    
    Supports both legacy 2-node format and new N-node format.
    
    Args:
        png1: Legacy - Path to first image (for backward compatibility)
        png2: Legacy - Path to second image (for backward compatibility)
        bbox1: Legacy - Bounding box (x, y, w, h) for first image
        bbox2: Legacy - Bounding box (x, y, w, h) for second image
        png_list: List of image paths for N nodes
        bbox_list: List of bounding boxes (x, y, w, h) for N nodes
        Wc: Container width
        Hc: Container height
        save_path: Path to save the composite image
    """
    import os
    
    # Handle backward compatibility: convert png1/png2 to png_list
    if png_list is None:
        if png1 is not None and png2 is not None:
            png_list = [png1, png2]
            bbox_list = [bbox1, bbox2]
        else:
            raise ValueError("Either (png_list, bbox_list) or (png1, png2, bbox1, bbox2) must be provided")
    
    if bbox_list is None:
        raise ValueError("bbox_list must be provided")
    
    if len(png_list) != len(bbox_list):
        raise ValueError(f"Mismatch: {len(png_list)} images but {len(bbox_list)} bboxes")
    
    num_nodes = len(png_list)
    if num_nodes == 0:
        if debug:
            print("Warning: No images to composite")
        return
    
    for i, png_path in enumerate(png_list):
        if png_path is None:
            if debug:
                print(f"Warning: Image path at index {i} is None")
            return
        if not os.path.exists(png_path):
            if debug:
                print(f"Warning: Image file not found: {png_path}")
            return
    
    if debug:
        print(f"Loading {num_nodes} images for composite")
        print(f"Container size: {Wc}x{Hc}")
    
    # Create output canvas
    canvas = Image.new("RGBA", (Wc, Hc), (255, 255, 255, 255))
    
    # Place all images
    for i, (png_path, bbox) in enumerate(zip(png_list, bbox_list)):
        x, y, w, h = bbox
        x_int = int(x)
        y_int = int(y)
        w_int = max(1, int(w))
        h_int = max(1, int(h))
        
        if w_int > 0 and h_int > 0:
            # Clip coordinates to canvas bounds
            x_clip = max(0, min(x_int, Wc - 1))
            y_clip = max(0, min(y_int, Hc - 1))
            
            # Calculate how much of the image fits in the canvas
            x_end = min(x_clip + w_int, Wc)
            y_end = min(y_clip + h_int, Hc)
            w_fit = x_end - x_clip
            h_fit = y_end - y_clip
            
            if w_fit > 0 and h_fit > 0:
                img = Image.open(png_path).convert("RGBA")
                orig_w, orig_h = img.size
                scale_w = w_int / orig_w if orig_w > 0 else 1.0
                scale_h = h_int / orig_h if orig_h > 0 else 1.0
                scale = min(scale_w, scale_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                x_offset = (w_int - new_w) // 2
                y_offset = (h_int - new_h) // 2
                paste_x = x_clip + x_offset
                paste_y = y_clip + y_offset
                paste_x = max(0, min(paste_x, Wc - 1))
                paste_y = max(0, min(paste_y, Hc - 1))
                x_end_paste = min(paste_x + new_w, Wc)
                y_end_paste = min(paste_y + new_h, Hc)
                w_paste = x_end_paste - paste_x
                h_paste = y_end_paste - paste_y
                if w_paste < new_w or h_paste < new_h:
                    img_resized = img_resized.crop((0, 0, w_paste, h_paste))
                canvas.paste(img_resized, (paste_x, paste_y), img_resized)
                if debug:
                    print(f"Placed image{i+1} ({png_path}) at ({paste_x}, {paste_y}) with size ({w_paste}, {h_paste})")
                if debug:
                    print(f"Placed image{i+1} ({png_path}) at ({x_clip}, {y_clip}) with size ({w_fit}, {h_fit})")
            else:
                if debug:
                    print(f"Warning: Image{i+1} has no valid area to place: ({x_clip}, {y_clip}, {w_fit}, {h_fit})")
        else:
            if debug:
                print(f"Warning: Skipping image{i+1} - invalid size: ({w_int}, {h_int})")
    
    # Convert to RGB for saving (remove alpha channel)
    canvas_rgb = Image.new("RGB", canvas.size, (255, 255, 255))
    canvas_rgb.paste(canvas, mask=canvas.split()[3])  # Use alpha channel as mask
    
    # Save result
    canvas_rgb.save(save_path, "PNG")
    if debug:
        print(f"Composite image saved to: {save_path}")

