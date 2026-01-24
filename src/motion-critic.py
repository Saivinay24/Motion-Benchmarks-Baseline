import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_and_analyze_motion():
    print("1. Generating Synthetic Test Video (Smooth vs. Jittery)...")
    
    # Video settings
    width, height = 640, 480
    frames = 60
    
    # OUTPUT: Initialize Video Writer to save the file for the CLI tool
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('synthetic_test.mp4', fourcc, 30, (width, height))
    
    # Track metrics
    smoothness_scores = []
    
    # Initialize previous frame for Optical Flow
    prev_gray = np.zeros((height, width), dtype=np.uint8)
    
    print("2. Running Optical Flow Analysis...")
    
    for i in range(frames):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # LOGIC: Move a white circle across the screen
        # Frames 0-30: Perfect Physics (Smooth)
        # Frames 30-60: Bad Physics (Random Jitter)
        x_pos = int(50 + (i * 8)) # Constant velocity
        y_pos = 240
        
        if i > 30: # Inject "Bad Physics"
            noise_x = np.random.randint(-15, 15)
            noise_y = np.random.randint(-15, 15)
            x_pos += noise_x
            y_pos += noise_y
            status = "BAD PHYSICS (Jitter)"
            color = (0, 0, 255) # Red for bad
        else:
            status = "NATURAL MOTION"
            color = (0, 255, 0) # Green for good
            
        # Draw the ball
        cv2.circle(frame, (x_pos, y_pos), 30, (255, 255, 255), -1)
        
        # Add Text Label
        cv2.putText(frame, f"Frame {i}: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # SAVE FRAME TO DISK (Crucial Step)
        out.write(frame)

        # Convert to grayscale for Optical Flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate Optical Flow (Farneback)
        if i > 0:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Calculate Magnitude (Speed of motion per pixel)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # METRIC: Calculate the variance of motion
            if np.count_nonzero(mag > 1) > 0:
                motion_variance = np.var(mag[mag > 1])
            else:
                motion_variance = 0
                
            smoothness_scores.append(motion_variance)
        else:
            smoothness_scores.append(0)
            
        prev_gray = gray

    # Release the video file so other tools can use it
    out.release()
    print("   > Video saved as 'synthetic_test.mp4'")

    print("3. Plotting the 'Physics Violation' Graph...")
    
    # Plotting results
    plt.figure(figsize=(10, 6))
    
    # Plot the metric
    plt.plot(smoothness_scores, linewidth=2, color='blue', label='Motion Variance (Jitter)')
    
    # Add zones
    plt.axvspan(0, 30, color='green', alpha=0.1, label='Phase 1: Natural Motion')
    plt.axvspan(30, 60, color='red', alpha=0.1, label='Phase 2: Glitchy Motion')
    
    plt.title("Automated Detection of 'Unnatural' Animation", fontsize=14)
    plt.xlabel("Video Frame", fontsize=12)
    plt.ylabel("Detected Motion Instability (Optical Flow)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save as png to match README
    output_filename = 'sensitivity_curve.png'
    plt.savefig(output_filename)
    print(f"SUCCESS: Evidence saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    create_and_analyze_motion()