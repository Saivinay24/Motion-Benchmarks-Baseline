import argparse
import matplotlib.pyplot as plt
import numpy as np
from motion_metric import MotionCritic

def main():
    # 1. Setup Arguments (So the user can provide any video)
    parser = argparse.ArgumentParser(description="GenAI Motion Critic: Audit Video Physics")
    parser.add_argument("--video", type=str, required=True, help="Path to the .mp4 file")
    parser.add_argument("--output", type=str, default="physics_report.png", help="Output filename for the graph")
    args = parser.parse_args()

    print(f"Loading Motion Critic...")
    critic = MotionCritic()
    
    print(f"Processing video: {args.video} (This may take a moment)...")
    try:
        scores = critic.compute_jitter_score(args.video)
    except Exception as e:
        print(e)
        return

    if not scores:
        print("Error: No frames analyzed.")
        return

    # 2. Generate Professional Report
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    
    print(f"Analysis Complete.")
    print(f" > Average Jitter: {avg_score:.2f}")
    print(f" > Peak Instability: {max_score:.2f}")

    plt.figure(figsize=(12, 6))
    
    # Plot the raw metric
    plt.plot(scores, color='#007acc', linewidth=2, label='Optical Flow Variance (Physics Violation)')
    
    # Draw a "Threshold" line (Scientific Baseline)
    # We assume arbitrarily that > 50 is visually disturbing
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Human Perceptual Threshold (Est.)')

    plt.title(f"Physics Compliance Report: {args.video}", fontsize=14, fontweight='bold')
    plt.xlabel("Frame Number", fontsize=12)
    plt.ylabel("Motion Instability Score", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add stats box
    stats_text = f"Mean Jitter: {avg_score:.1f}\nMax Jitter: {max_score:.1f}"
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.savefig(args.output)
    print(f"SUCCESS: Report saved to {args.output}")
    plt.show()

if __name__ == "__main__":
    main()