import cv2
import numpy as np

class MotionCritic:
    def __init__(self):
        # Parameters for Farneback Optical Flow
        self.pyr_scale = 0.5
        self.levels = 3
        self.winsize = 15
        self.iterations = 3
        self.poly_n = 5
        self.poly_sigma = 1.2
        self.flags = 0

    def compute_jitter_score(self, video_path):
        """
        Analyzes a video file and returns a frame-by-frame 'Jitter Score' (Variance).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file at {video_path}")

        scores = []
        ret, prev_frame = cap.read()
        if not ret:
            return []

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Calculate Dense Optical Flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                self.pyr_scale, self.levels, self.winsize,
                self.iterations, self.poly_n, self.poly_sigma, self.flags
            )

            # 2. Extract Motion Magnitude
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # 3. Filter: Ignore static background (only check moving pixels)
            # Threshold: We assume 'real' motion is > 1.0 pixel per frame
            active_motion = mag[mag > 1.0]

            # 4. Metric: Variance of the motion magnitude
            # High Variance = Jittery/Chaotic (Bad Physics)
            # Low Variance = Smooth/Consistent (Good Physics)
            if len(active_motion) > 0:
                score = np.var(active_motion)
            else:
                score = 0.0

            scores.append(score)
            prev_gray = gray

        cap.release()
        return scores