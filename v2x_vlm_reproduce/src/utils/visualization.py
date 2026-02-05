"""
可视化工具

用于:
1. 轨迹预测可视化
2. 训练曲线绘制
3. 注意力图可视化
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, visualization disabled")


def visualize_trajectory(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: str,
    num_samples: int = 10,
    ego_size: tuple = (4.5, 1.8),
    show_error: bool = True
):
    """
    可视化轨迹预测结果
    
    Args:
        predictions: 预测轨迹 [N, T, 2]
        targets: GT轨迹 [N, T, 2]
        output_dir: 输出目录
        num_samples: 可视化样本数
        ego_size: 自车尺寸 (长, 宽)
        show_error: 是否显示误差统计
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 随机选择样本
    n_total = len(predictions)
    indices = np.random.choice(n_total, min(num_samples, n_total), replace=False)
    
    for i, idx in enumerate(indices):
        pred = predictions[idx]  # [T, 2]
        target = targets[idx]    # [T, 2]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 绘制GT轨迹
        ax.plot(target[:, 0], target[:, 1], 
               'g-', linewidth=3, label='Ground Truth', zorder=2)
        ax.scatter(target[0, 0], target[0, 1], 
                  c='green', s=150, marker='o', label='Start', zorder=3)
        ax.scatter(target[-1, 0], target[-1, 1], 
                  c='green', s=150, marker='*', zorder=3)
        
        # 绘制预测轨迹
        ax.plot(pred[:, 0], pred[:, 1], 
               'r--', linewidth=2, label='Prediction', zorder=2)
        ax.scatter(pred[-1, 0], pred[-1, 1], 
                  c='red', s=150, marker='*', zorder=3)
        
        # 绘制误差线
        for t in range(0, len(pred), 5):
            ax.plot([pred[t, 0], target[t, 0]], 
                   [pred[t, 1], target[t, 1]], 
                   'b:', alpha=0.5, linewidth=1)
        
        # 绘制时间标记
        time_points = [0, 25, 35, 44]  # 0s, 2.5s, 3.5s, 4.5s
        for tp in time_points:
            if tp < len(target):
                ax.annotate(f'{tp/10:.1f}s', 
                           xy=(target[tp, 0], target[tp, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='green')
        
        # 计算误差统计
        if show_error:
            l2_error = np.sqrt(((pred - target) ** 2).sum(axis=-1))
            
            # 添加误差统计文本
            stats_text = (
                f"L2 Error Stats:\n"
                f"  Mean: {l2_error.mean():.3f}m\n"
                f"  @2.5s: {l2_error[:25].mean():.3f}m\n"
                f"  @3.5s: {l2_error[:35].mean():.3f}m\n"
                f"  @4.5s: {l2_error.mean():.3f}m\n"
                f"  Max: {l2_error.max():.3f}m"
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'Trajectory Prediction - Sample {idx}', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'trajectory_{i:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {len(indices)} trajectory visualizations to {output_dir}")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    metrics: Dict[str, List[float]] = None,
    output_path: str = "training_curves.png"
):
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        metrics: 其他指标字典
        output_path: 输出路径
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return
    
    n_plots = 2 if metrics is None else 2 + len(metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    # 损失曲线
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 验证损失单独显示
    axes[1].plot(epochs, val_losses, 'r-o', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Loss')
    axes[1].set_title('Validation Loss')
    axes[1].grid(True, alpha=0.3)
    
    # 标记最低点
    best_epoch = np.argmin(val_losses) + 1
    best_val = min(val_losses)
    axes[1].annotate(f'Best: {best_val:.4f}', 
                    xy=(best_epoch, best_val),
                    xytext=(best_epoch + 0.5, best_val + 0.01),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='gray'))
    
    # 其他指标
    if metrics:
        for i, (name, values) in enumerate(metrics.items()):
            ax = axes[2 + i]
            ax.plot(range(1, len(values) + 1), values, 'g-o', markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {output_path}")


def visualize_attention(
    attention_weights: np.ndarray,
    image: np.ndarray,
    output_path: str,
    alpha: float = 0.5
):
    """
    可视化注意力权重
    
    Args:
        attention_weights: 注意力权重 [H, W]
        image: 原始图像 [H, W, 3]
        output_path: 输出路径
        alpha: 叠加透明度
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 注意力热图
    im = axes[1].imshow(attention_weights, cmap='hot')
    axes[1].set_title('Attention Weights')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # 叠加图像
    axes[2].imshow(image)
    
    # 调整注意力大小以匹配图像
    from scipy.ndimage import zoom
    scale_h = image.shape[0] / attention_weights.shape[0]
    scale_w = image.shape[1] / attention_weights.shape[1]
    attention_resized = zoom(attention_weights, (scale_h, scale_w))
    
    axes[2].imshow(attention_resized, cmap='hot', alpha=alpha)
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_trajectory_comparison(
    predictions_list: List[np.ndarray],
    labels: List[str],
    target: np.ndarray,
    output_path: str
):
    """
    比较多个模型的轨迹预测
    
    Args:
        predictions_list: 多个预测结果列表
        labels: 模型标签
        target: GT轨迹
        output_path: 输出路径
    """
    if not HAS_MATPLOTLIB:
        return
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_list)))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # GT
    ax.plot(target[:, 0], target[:, 1], 'k-', linewidth=3, label='Ground Truth')
    ax.scatter(target[0, 0], target[0, 1], c='black', s=100, marker='o')
    
    # 各模型预测
    for pred, label, color in zip(predictions_list, labels, colors):
        ax.plot(pred[:, 0], pred[:, 1], '--', linewidth=2, 
               color=color, label=label)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Trajectory Comparison', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_video_from_trajectories(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    fps: int = 10
):
    """
    从轨迹创建动画视频
    
    Args:
        predictions: [T, 2]
        targets: [T, 2]
        output_path: 输出视频路径
        fps: 帧率
    """
    if not HAS_MATPLOTLIB:
        return
    
    try:
        from matplotlib.animation import FuncAnimation, FFMpegWriter
    except ImportError:
        print("FFmpeg not available for video creation")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # 设置坐标范围
    all_points = np.concatenate([predictions, targets], axis=0)
    margin = 5
    ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
    ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
    
    # 初始化线条
    gt_line, = ax.plot([], [], 'g-', linewidth=2, label='Ground Truth')
    pred_line, = ax.plot([], [], 'r--', linewidth=2, label='Prediction')
    gt_point, = ax.plot([], [], 'go', markersize=10)
    pred_point, = ax.plot([], [], 'ro', markersize=10)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    def init():
        gt_line.set_data([], [])
        pred_line.set_data([], [])
        gt_point.set_data([], [])
        pred_point.set_data([], [])
        return gt_line, pred_line, gt_point, pred_point
    
    def animate(frame):
        gt_line.set_data(targets[:frame+1, 0], targets[:frame+1, 1])
        pred_line.set_data(predictions[:frame+1, 0], predictions[:frame+1, 1])
        gt_point.set_data([targets[frame, 0]], [targets[frame, 1]])
        pred_point.set_data([predictions[frame, 0]], [predictions[frame, 1]])
        ax.set_title(f'Time: {frame/fps:.1f}s')
        return gt_line, pred_line, gt_point, pred_point
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(predictions), interval=1000/fps, blit=True)
    
    try:
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='V2X-VLM'))
        anim.save(output_path, writer=writer)
        print(f"Video saved to {output_path}")
    except Exception as e:
        print(f"Failed to save video: {e}")
    
    plt.close()
