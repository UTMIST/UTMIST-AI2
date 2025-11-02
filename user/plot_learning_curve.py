"""
Generate learning curve visualization from training data.
Run this after training to analyze progress and identify areas for improvement.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

def plot_learning_curve(experiment_path):
    """
    Plot learning curve with improved visualization and analysis.
    
    Args:
        experiment_path: Path to checkpoint folder (e.g., 'checkpoints/experiment_aggressive_v3')
    """
    if not os.path.exists(experiment_path):
        print(f"Error: Path {experiment_path} does not exist!")
        return
    
    monitor_file = os.path.join(experiment_path, 'monitor.csv')
    if not os.path.exists(monitor_file):
        print(f"Error: monitor.csv not found in {experiment_path}!")
        return
    
    # Load data
    x, y = ts2xy(load_results(experiment_path), "timesteps")
    
    if len(x) == 0 or len(y) == 0:
        print("Error: No data found in monitor.csv!")
        return
    
    # Smoothing window
    window_size = 100
    weights = np.repeat(1.0, window_size) / window_size
    y_smoothed = np.convolve(y, weights, "valid")
    x_smoothed = x[len(x) - len(y_smoothed):]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Main learning curve
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(x_smoothed, y_smoothed, linewidth=2, color='blue', label='Smoothed Reward')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel("Timesteps", fontsize=11)
    ax1.set_ylabel("Episode Reward", fontsize=11)
    ax1.set_title("Learning Curve (Smoothed, Window=100)", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Recent performance (last 1M steps)
    ax2 = plt.subplot(2, 2, 2)
    if len(x_smoothed) > 0:
        last_1m = x_smoothed[-1] - 1_000_000
        mask = x_smoothed >= max(0, last_1m)
        if np.any(mask):
            ax2.plot(x_smoothed[mask], y_smoothed[mask], linewidth=2, color='green')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_xlabel("Timesteps", fontsize=11)
            ax2.set_ylabel("Episode Reward", fontsize=11)
            ax2.set_title("Recent Performance (Last 1M Steps)", fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Statistics
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')
    
    # Calculate statistics
    total_steps = x[-1] if len(x) > 0 else 0
    final_avg_reward = np.mean(y_smoothed[-1000:]) if len(y_smoothed) >= 1000 else np.mean(y_smoothed)
    max_reward = np.max(y_smoothed)
    min_reward = np.min(y_smoothed)
    recent_trend = np.mean(y_smoothed[-1000:]) - np.mean(y_smoothed[-5000:-1000]) if len(y_smoothed) >= 5000 else 0
    
    stats_text = f"""
TRAINING STATISTICS
═══════════════════════════════════════
Total Timesteps: {total_steps/1e6:.2f}M
Final Average Reward: {final_avg_reward:.2f}
Maximum Reward: {max_reward:.2f}
Minimum Reward: {min_reward:.2f}
Recent Trend: {'+' if recent_trend > 0 else ''}{recent_trend:.2f}

INTERPRETATION:
═══════════════════════════════════════
• Final reward > 0: Agent winning more than losing
• Final reward < 0: Agent needs improvement
• Upward trend: Learning successfully
• Downward trend: May need reward tuning

CURRENT STATUS:
═══════════════════════════════════════
"""
    if final_avg_reward > 0:
        stats_text += "✓ Agent is performing well (positive rewards)"
    elif final_avg_reward > -100:
        stats_text += "⚠ Agent is improving but still negative rewards"
    else:
        stats_text += "✗ Agent struggling (highly negative rewards)"
    
    if recent_trend > 5:
        stats_text += "\n✓ Strong upward trend - keep training!"
    elif recent_trend > 0:
        stats_text += "\n✓ Slow improvement - patience needed"
    elif recent_trend > -5:
        stats_text += "\n⚠ Plateauing - may need hyperparameter tuning"
    else:
        stats_text += "\n✗ Declining performance - check reward function"
    
    ax3.text(0.1, 0.5, stats_text, fontsize=10, fontfamily='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Reward distribution histogram
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(y_smoothed, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Reward')
    ax4.set_xlabel("Episode Reward", fontsize=11)
    ax4.set_ylabel("Frequency", fontsize=11)
    ax4.set_title("Reward Distribution", fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f"Training Analysis: {os.path.basename(experiment_path)}", 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save
    output_path = os.path.join(experiment_path, "Learning Curve.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Learning curve saved to: {output_path}")
    
    # Also save full analysis
    analysis_path = os.path.join(experiment_path, "Training Analysis.png")
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    print(f"✓ Full analysis saved to: {analysis_path}")
    
    plt.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY for {os.path.basename(experiment_path)}")
    print(f"{'='*60}")
    print(f"Total Steps: {total_steps/1e6:.2f}M")
    print(f"Final Avg Reward: {final_avg_reward:.2f}")
    print(f"Recent Trend: {recent_trend:+.2f} per 1000 episodes")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Default to latest experiment
    if len(sys.argv) > 1:
        experiment_path = sys.argv[1]
    else:
        # Find latest experiment
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            experiments = [d for d in os.listdir(checkpoint_dir) 
                          if os.path.isdir(os.path.join(checkpoint_dir, d)) 
                          and d.startswith('experiment')]
            if experiments:
                # Get most recently modified
                experiments.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
                experiment_path = os.path.join(checkpoint_dir, experiments[0])
                print(f"Using latest experiment: {experiment_path}")
            else:
                print("No experiments found! Please specify path:")
                print(f"  python user/plot_learning_curve.py checkpoints/experiment_name")
                sys.exit(1)
        else:
            print("checkpoints directory not found!")
            sys.exit(1)
    
    plot_learning_curve(experiment_path)

