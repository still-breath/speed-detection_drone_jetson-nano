import cv2
import os
import time
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from filterpy.kalman import KalmanFilter

# Limit CPU usage for Jetson Nano
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker

class KalmanFilterWrapper:
    """Enhanced Kalman Filter for tracking and smoothing object positions"""
    def __init__(self):
        # Initialize Kalman Filter with 4 state variables (x, y, vx, vy)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        dt = 0.033  # Assuming 30 FPS
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement function
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Initial state uncertainty
        self.kf.P = np.eye(4) * 50
        
        # Measurement noise (lower = more trust in measurements)
        self.kf.R = np.eye(2) * 2.5
        
        # Process noise - optimized for vehicle tracking from drone
        q = 0.02  # Process noise magnitude
        self.kf.Q = np.array([
            [q/4*dt**4, 0, q/2*dt**3, 0],
            [0, q/4*dt**4, 0, q/2*dt**3],
            [q/2*dt**3, 0, q*dt**2, 0],
            [0, q/2*dt**3, 0, q*dt**2]
        ])
        
        # Initialize state with zeros
        self.kf.x = np.array([0, 0, 0, 0])
        
        # Flag to check if filter has been initialized
        self.initialized = False

    def predict(self):
        """Predict next position."""
        if self.initialized:
            self.kf.predict()
        return self.kf.x[:2]  # Return the position part of state

    def update(self, measurement):
        """Update filter with new measurement."""
        if not self.initialized:
            # If first measurement, initialize the state
            self.kf.x = np.array([measurement[0], measurement[1], 0, 0])
            self.initialized = True
        else:
            self.kf.update(measurement)
        return self.kf.x  # Return updated state

    def get_position(self):
        """Get current position from filter."""
        return self.kf.x[:2]  # Take x and y from state vector
    
    def get_velocity(self):
        """Get current velocity from filter."""
        return self.kf.x[2:4]  # Take vx and vy from state vector

class ImprovedVideoMosaic:
    """Enhanced video mosaic class for panorama creation from moving drone footage"""
    def __init__(self, first_image, output_height_times=2, output_width_times=3, detector_type="orb", debug=False):
        """Initialize the video stitching mosaic.
        
        Args:
            first_image: First frame to initialize panorama
            output_height_times: Height multiplier for panorama
            output_width_times: Width multiplier for panorama
            detector_type: Feature detector type ('orb' or 'sift')
            debug: Enable debug visualizations
        """
        self.detector_type = detector_type
        self.debug = debug
        
        # Initialize feature detector based on type
        if detector_type == "sift":
            self.detector = cv2.SIFT_create(600)  # Optimized for jetson nano
            self.bf = cv2.BFMatcher()
        elif detector_type == "orb":
            self.detector = cv2.ORB_create(600)  # Higher number for better feature detection
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
        # Process first frame
        self.process_first_frame(first_image)
        
        # Initialize output image
        self.output_img = np.zeros(shape=(int(output_height_times * first_image.shape[0]), 
                                        int(output_width_times * first_image.shape[1]), 
                                        first_image.shape[2]), dtype=np.uint8)
        
        # Offset for initial placement (center the first frame)
        self.w_offset = int(self.output_img.shape[0] / 2 - first_image.shape[0] / 2)
        self.h_offset = int(self.output_img.shape[1] / 2 - first_image.shape[1] / 2)
        
        # Place first frame in panorama
        self.output_img[self.w_offset:self.w_offset + first_image.shape[0],
                        self.h_offset:self.h_offset + first_image.shape[1], :] = first_image
        
        # Initialize homography matrices
        self.H_old = np.eye(3)
        self.H_old[0, 2] = self.h_offset
        self.H_old[1, 2] = self.w_offset
        
        # Keep track of the cumulative homography for coordinate transformation
        self.H_cumulative = self.H_old.copy()
        
        # Movement tracking
        self.last_good_H = self.H_old.copy()
        self.consecutive_failures = 0
        self.max_failures = 3
        
        # Motion smoothing
        self.prev_translations = []
        self.prev_rotations = []
        self.smoothing_window = 5

    def process_first_frame(self, first_image):
        """Process the first frame for feature detection."""
        self.frame_prev = first_image
        frame_gray_prev = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        self.kp_prev, self.des_prev = self.detector.detectAndCompute(frame_gray_prev, None)

    def match(self, des_cur, des_prev):
        """Match descriptors between frames with improved filtering."""
        if self.detector_type == "sift":
            # For SIFT, use ratio test
            pair_matches = self.bf.knnMatch(des_cur, des_prev, k=2)
            matches = []
            for m, n in pair_matches:
                if m.distance < 0.75 * n.distance:  # Slightly relaxed ratio test
                    matches.append(m)
        elif self.detector_type == "orb":
            # For ORB, use simple matching with distance threshold
            matches = self.bf.match(des_cur, des_prev)
            # Filter matches based on distance
            matches = [m for m in matches if m.distance < 70]  # Adjusted threshold
            
        # Sort by distance and select best ones
        matches = sorted(matches, key=lambda x: x.distance)
        return matches[:min(len(matches), 30)]  # Keep more matches for better homography

    def calculate_optical_flow(self, prev_frame, curr_frame):
        """Calculate optical flow for homography stabilization."""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Optical flow parameters tuned for Jetson Nano performance
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 
            pyr_scale=0.5,     # Pyramid scale
            levels=3,          # Pyramid levels (reduced for performance)
            winsize=15,        # Window size
            iterations=3,      # Iterations
            poly_n=5,          # Pixel neighborhood size
            poly_sigma=1.2,    # Standard deviation
            flags=0
        )
        return flow

    def process_frame(self, frame_cur):
        """Process each frame for stitching with enhanced reliability."""
        self.frame_cur = frame_cur
        frame_gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
        self.kp_cur, self.des_cur = self.detector.detectAndCompute(frame_gray_cur, None)
        
        # Match features
        self.matches = self.match(self.des_cur, self.des_prev)
        
        # Skip if not enough matches
        if len(self.matches) < 8:  # Require more matches for better homography
            self.consecutive_failures += 1
            if self.consecutive_failures <= self.max_failures:
                # Use last good homography with small adjustment
                self.H = self.last_good_H.copy()
                # Add small drift to simulate motion
                self.H[0, 2] += 1.0  # Small x drift
                self.H_cumulative = np.matmul(self.H_old, self.H)
            return
        
        # Reset failure counter
        self.consecutive_failures = 0
        
        # Calculate homography
        self.H = self.findHomography(self.kp_cur, self.kp_prev, self.matches)
        if self.H is None:
            return
            
        # Store as last good homography
        self.last_good_H = self.H.copy()
        
        # Enhance with optical flow for better stabilization
        flow = self.calculate_optical_flow(self.frame_prev, self.frame_cur)
        flow_translation = np.mean(flow, axis=(0, 1))
        
        # Create stabilization matrix from flow
        stabilization_matrix = np.eye(3)
        stabilization_matrix[0, 2] = -flow_translation[0] * 0.3
        stabilization_matrix[1, 2] = -flow_translation[1] * 0.3
        
        # Combine stabilization with homography
        self.H = np.matmul(stabilization_matrix, self.H)
        
        # Apply advanced smoothing to homography
        self.H = self.smooth_homography(self.H, self.H_old)
        
        # Compute the cumulative homography for coordinate transformation
        self.H_cumulative = np.matmul(self.H_old, self.H)
        
        # Check for unrealistic transformation
        if self.is_reasonable_homography(self.H_cumulative):
            # Warp the frame to the panorama
            self.warp(frame_cur, self.H_cumulative)
            
            # Prepare for next iteration
            self.H_old = self.H_cumulative.copy()
            self.kp_prev = self.kp_cur
            self.des_prev = self.des_cur
            self.frame_prev = self.frame_cur
        else:
            # If homography seems unreasonable, use previous one
            self.H_cumulative = self.H_old.copy()
            
    def is_reasonable_homography(self, H):
        """Check if homography is reasonable (no extreme distortion)"""
        # Extract rotation component (SVD)
        try:
            u, s, vh = np.linalg.svd(H[:2, :2])
            
            # Check condition number (ratio of singular values)
            condition = s[0] / s[1]
            if condition > 15:  # Too much distortion
                return False
                
            # Check for extreme scaling
            scale_factor = np.sqrt(s[0] * s[1])
            if scale_factor < 0.5 or scale_factor > 2.0:
                return False
                
            # Check translation magnitude
            translation_magnitude = np.sqrt(H[0, 2]**2 + H[1, 2]**2)
            if translation_magnitude > 200:  # Too large movement
                return False
                
            return True
        except:
            return False
        
    def smooth_homography(self, H_new, H_old, alpha=0.8):
        """Apply adaptive smoothing to homography matrix."""
        if H_new is None or H_old is None:
            return H_old if H_old is not None else np.eye(3)
            
        # Decompose homographies
        # Extract translation components
        t_new = np.array([H_new[0, 2], H_new[1, 2]])
        t_old = np.array([H_old[0, 2], H_old[1, 2]])
        
        # Extract rotation + scale components (2x2 upper left)
        rs_new = H_new[:2, :2]
        rs_old = H_old[:2, :2]
        
        # Store history for smoothing
        self.prev_translations.append(t_new)
        if len(self.prev_translations) > self.smoothing_window:
            self.prev_translations.pop(0)
            
        # Apply smoothing
        if len(self.prev_translations) > 1:
            # More weight to recent translations, less to older ones
            weights = np.linspace(0.5, 1.0, len(self.prev_translations))
            weights = weights / np.sum(weights)  # Normalize
            
            # Weighted average of recent translations
            t_smooth = np.zeros(2)
            for i, trans in enumerate(self.prev_translations):
                t_smooth += trans * weights[i]
        else:
            # Not enough history, use simple smoothing
            t_smooth = alpha * t_old + (1 - alpha) * t_new
            
        # Smoother transition for rotation/scaling
        rs_smooth = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                rs_smooth[i, j] = alpha * rs_old[i, j] + (1 - alpha) * rs_new[i, j]
                
        # Reconstruct smoothed homography
        H_smooth = np.eye(3)
        H_smooth[:2, :2] = rs_smooth
        H_smooth[0, 2] = t_smooth[0]
        H_smooth[1, 2] = t_smooth[1]
        
        return H_smooth

    @staticmethod
    def findHomography(image_1_kp, image_2_kp, matches):
        """Calculate robust homography matrix using RANSAC."""
        # Extract matched points
        image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        
        for i in range(len(matches)):
            image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
            image_2_points[i] = image_2_kp[matches[i].trainIdx].pt
            
        # Find homography with RANSAC
        homography, mask = cv2.findHomography(
            image_1_points, image_2_points, 
            method=cv2.RANSAC, 
            ransacReprojThreshold=3.0,  # Slightly increased threshold
            maxIters=2000  # More iterations for better result
        )
        
        # Validate homography quality
        if mask is not None:
            inliers = np.sum(mask)
            if inliers < 8:  # Require more inliers for reliability
                return None
                
            # Check inlier ratio
            inlier_ratio = inliers / len(matches)
            if inlier_ratio < 0.4:  # At least 40% should be inliers
                return None
        
        return homography

    def warp(self, frame_cur, H):
        """Warp the current frame into the panorama using the calculated homography."""
        warped_img = cv2.warpPerspective(
            frame_cur, 
            H, 
            (self.output_img.shape[1], self.output_img.shape[0]), 
            flags=cv2.INTER_LINEAR
        )
        
        # Combine only valid areas to the panorama
        mask = warped_img > 0  # Mask for valid (non-black) areas
        self.output_img[mask] = warped_img[mask]
        
    def transform_coordinates(self, points):
        """Transform coordinates from original frame to panorama coordinates.
        
        Args:
            points: List of (x, y) points in the original frame
            
        Returns:
            List of transformed (x, y) points in the panorama
        """
        if len(points) == 0:
            return []
            
        # Convert to homogeneous coordinates
        homogeneous_points = np.ones((len(points), 3))
        for i, point in enumerate(points):
            homogeneous_points[i, 0:2] = point
            
        # Apply transformation
        transformed_points = []
        for point in homogeneous_points:
            transformed = np.dot(self.H_cumulative, point)
            # Convert back from homogeneous coordinates
            if transformed[2] != 0:
                transformed_points.append((transformed[0]/transformed[2], transformed[1]/transformed[2]))
            else:
                transformed_points.append((transformed[0], transformed[1]))
                
        return transformed_points
        
    def get_motion_vector(self):
        """Calculate camera motion vector from homography.
        Returns:
            (dx, dy): Estimated camera motion
        """
        if hasattr(self, 'H') and self.H is not None:
            dx = self.H[0, 2]
            dy = self.H[1, 2]
            return (dx, dy)
        return (0, 0)
        
    def get_inverse_transform(self):
        """Get inverse transformation to convert from panorama to original frame coords."""
        if hasattr(self, 'H_cumulative'):
            try:
                return np.linalg.inv(self.H_cumulative)
            except np.linalg.LinAlgError:
                return np.eye(3)
        return np.eye(3)

class VehicleSpeedEstimator:
    """Enhanced speed estimator that accounts for camera motion"""
    def __init__(self, save_dir=None, pixels_per_meter=38.0, max_track_points=30, speed_window=5):
        self.tracks = {}
        self.pixels_per_meter = pixels_per_meter
        self.save_dir = save_dir
        self.max_track_points = max_track_points
        self.speed_window = speed_window
        self.speed_correction_factor = 0.6  # Calibration factor
        self.alpha = 0.15  # Smoothing factor
        self.median_window = 5
        
        # Create log file for speed data
        if save_dir:
            self.speed_log_file = open(save_dir / 'speed_log.txt', 'w')
            # Write header
            self.speed_log_file.write("ID,Class,Time(s),X,Y,Speed(km/h),Distance(m),RelativeSpeed(km/h)\n")
        else:
            self.speed_log_file = None
            
        # Constants for speed calculation
        self.dt_threshold = 0.2  # Minimum time difference for speed calculation
        self.min_points_for_speed = 3  # Minimum points required for speed estimation
        
        # Camera motion compensation
        self.camera_motion = (0, 0)  # Initialize camera motion vector
        self.motion_history = []
        self.motion_history_size = 10

    def update_camera_motion(self, motion_vector):
        """Update camera motion vector from mosaic."""
        self.camera_motion = motion_vector
        
        # Store in history for smoothing
        self.motion_history.append(motion_vector)
        if len(self.motion_history) > self.motion_history_size:
            self.motion_history.pop(0)
            
        # Calculate smoothed camera motion
        if len(self.motion_history) > 0:
            dx_sum, dy_sum = 0, 0
            for dx, dy in self.motion_history:
                dx_sum += dx
                dy_sum += dy
            self.camera_motion = (dx_sum / len(self.motion_history), 
                                 dy_sum / len(self.motion_history))

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def update(self, object_id, x, y, timestamp_ms, class_name=None):
        """Update tracking information for an object with camera motion compensation.
        
        Args:
            object_id: Unique ID for the tracked object
            x, y: Current position
            timestamp_ms: Current timestamp in milliseconds
            class_name: Object class name
            
        Returns:
            (speed, track_points): Current speed in km/h and list of track points
        """
        current_time = timestamp_ms / 1000.0  # Convert from milliseconds to seconds
        
        if object_id not in self.tracks:
            # Initialize a new track with Kalman Filter
            self.tracks[object_id] = {
                'points': [],          # Track all (x,y) positions
                'filtered_points': [], # Track filtered (x,y) positions
                'times': [],           # Track time for each position (seconds)
                'velocities': [],      # Instantaneous velocities
                'speeds': [],          # Speed history
                'current_speed': None, # Current smoothed speed
                'total_distance': 0.0, # Total distance traveled
                'initial_time': current_time,
                'class_name': class_name,
                'kalman_filter': KalmanFilterWrapper()
            }
            
        track = self.tracks[object_id]
        
        # Store raw position and time
        track['points'].append((float(x), float(y)))
        track['times'].append(current_time)
        
        # Use Kalman filter to smooth position
        track['kalman_filter'].predict()
        measurement = np.array([x, y])
        filtered_state = track['kalman_filter'].update(measurement)
        
        # Extract filtered position
        x_filtered, y_filtered = filtered_state[:2]
        track['filtered_points'].append((float(x_filtered), float(y_filtered)))
        
        # Calculate speed only if we have enough points
        if len(track['filtered_points']) >= 2:
            # Calculate distance between the last two filtered points
            last_point = track['filtered_points'][-2]
            current_point = track['filtered_points'][-1]
            
            # Calculate raw distance
            distance = self.calculate_distance(last_point, current_point)
            
            # Calculate time difference
            dt = track['times'][-1] - track['times'][-2]
            
            # Calculate camera motion compensation
            # How much the camera moved during this time interval
            camera_dx = self.camera_motion[0] * dt
            camera_dy = self.camera_motion[1] * dt
            
            # Calculate relative velocity considering camera motion
            relative_dx = current_point[0] - last_point[0] - camera_dx
            relative_dy = current_point[1] - last_point[1] - camera_dy
            
            # Compensated distance
            compensated_distance = np.sqrt(relative_dx**2 + relative_dy**2)
            
            # Convert distance from pixels to meters
            distance_meters = distance / self.pixels_per_meter
            compensated_distance_meters = compensated_distance / self.pixels_per_meter
            
            # Update total distance
            track['total_distance'] += distance_meters
            
            # Calculate instantaneous velocity if time difference is significant
            if dt > 0.001:  # Avoid division by zero
                # Calculate both raw and compensated velocities
                velocity = distance_meters / dt  # m/s
                compensated_velocity = compensated_distance_meters / dt  # m/s
                
                # Apply correction factor
                velocity *= self.speed_correction_factor
                compensated_velocity *= self.speed_correction_factor
                
                # Convert to km/h
                speed_kmh = velocity * 3.6
                compensated_speed_kmh = compensated_velocity * 3.6
                
                # Store compensated velocity (the one considering camera motion)
                track['velocities'].append(compensated_speed_kmh)
                
                # Apply median filtering for robustness
                if len(track['velocities']) >= self.median_window:
                    recent_speeds = track['velocities'][-self.median_window:]
                    median_speed = np.median(recent_speeds)
                else:
                    median_speed = compensated_speed_kmh

                # Apply additional smoothing for the speed
                if track['current_speed'] is None:
                    track['current_speed'] = median_speed
                else:
                    # Exponential moving average for smoothing
                    track['current_speed'] = (1 - self.alpha) * track['current_speed'] + self.alpha * median_speed
                
                track['speeds'].append(track['current_speed'])
                
                # Log speed information
                if self.speed_log_file and len(track['speeds']) % 5 == 0:  # Log every 5 updates
                    self.speed_log_file.write(
                        f"{object_id},{track['class_name']},{current_time:.3f},"
                        f"{x_filtered:.1f},{y_filtered:.1f},{track['current_speed']:.1f},"
                        f"{track['total_distance']:.2f},{compensated_speed_kmh:.1f}\n"
                    )
                    self.speed_log_file.flush()
        
        # Limit the number of tracking points for memory efficiency
        if len(track['points']) > self.max_track_points:
            track['points'].pop(0)
            track['times'].pop(0)
            track['filtered_points'].pop(0)
            
            if len(track['velocities']) > self.max_track_points:
                track['velocities'].pop(0)
            if len(track['speeds']) > self.max_track_points:
                track['speeds'].pop(0)
                
        return track['current_speed'], track['filtered_points']

    def draw_track(self, frame, track_points, color=(0, 255, 0), thickness=2):
        """Draw the tracking path on the frame."""
        if len(track_points) > 1:
            # Convert points to integer
            points = np.array(track_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=thickness)

    def end(self):
        """Clean up resources."""
        if self.speed_log_file:
            self.speed_log_file.close()

# FPSMonitor class for calculating and displaying real-time FPS
class FPSMonitor:
    def __init__(self, avg_frames=30):
        self.prev_time = time.time()
        self.curr_time = self.prev_time
        self.frame_times = []
        self.avg_frames = avg_frames
        self.fps = 0

    def update(self):
        self.curr_time = time.time()
        dt = self.curr_time - self.prev_time
        self.prev_time = self.curr_time
        
        # Store frame time for average calculation
        self.frame_times.append(dt)
        if len(self.frame_times) > self.avg_frames:
            self.frame_times.pop(0)
            
        # Calculate average FPS from recent frames
        avg_time = sum(self.frame_times) / len(self.frame_times)
        self.fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return self.fps
    
    def draw(self, frame, height=20):
        """Display FPS in top-left corner with background for readability."""
        fps_text = f"FPS: {self.fps:.1f}"
        # Add dark background for text
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (10, 10), (10 + text_size[0] + 10, height + 15), (0, 0, 0), -1)
        # Draw text
        cv2.putText(frame, fps_text, (15, height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def get_fps(self):
        return self.fps

@torch.no_grad()
def run(
        source='moving20_20km.mp4',
        yolo_weights='caraerial-320_fp16.engine',  # Optimized engine file for Jetson Nano
        imgsz=(320, 320),
        conf_thres=0.35,
        iou_thres=0.45,
        device='0',
        show_vid=True,
        line_thickness=1,
        hide_labels=False,
        hide_conf=True,
        vid_stride=1,
        pixels_per_meter=23,  # Initial calibration: pixels per meter
        draw_tracks=True,     # Draw tracking paths
        save_output=True      # Save output video
):
    # Directories and initialization
    source = str(source)
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    project = ROOT / 'runs' / 'track'
    name = 'exp'
    exist_ok = False
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = AutoBackend(yolo_weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)

    # Dataloader
    dataset = LoadImages(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_cap = dataset.cap if hasattr(dataset, 'cap') else None
    
    # Get video properties
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output video writer if saving
    out = None
    if save_output:
        output_path = str(save_dir / 'output.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 480))

    # Initialize Video Stitching
    video_mosaic = True
    is_first_frame = True
    
    # Initialize FPS monitor
    fps_monitor = FPSMonitor(avg_frames=10)

    # Initialize SpeedEstimator with calibration
    def calculate_pixels_per_meter(base_pixels_per_meter, current_height, base_height=20):
        """Recalculate pixels_per_meter based on drone height."""
        base_pixels_per_meter = 38.0
        return base_pixels_per_meter * (base_height / current_height)  # Inverted ratio

    # Current drone height (in meters)
    drone_height = 20  # Replace with actual value if using sensors
    adjusted_pixels_per_meter = calculate_pixels_per_meter(38, drone_height)
    
    # Create speed estimator with adjusted calibration
    speed_estimator = VehicleSpeedEstimator(
        save_dir=save_dir, 
        pixels_per_meter=adjusted_pixels_per_meter,
        max_track_points=30,  # Keep 30 track points
        speed_window=5        # Use 5 frames for speed averaging
    )

    # Verify that ByteTrack config file exists
    tracking_config = ROOT / 'trackers' / 'bytetrack' / 'configs' / 'bytetrack.yaml'
    
    # Initialize the tracker (without ReID)
    tracker = create_tracker('bytetrack', tracking_config, None, device, False)

    # Run tracking
    seen, dt = 0, (Profile(), Profile(), Profile())
    frame_count = 0
    
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        timestamp_ms = vid_cap.get(cv2.CAP_PROP_POS_MSEC) if vid_cap is not None else (frame_idx * 1000 / fps)
        frame_count += 1
        
        # Update FPS monitor
        current_fps = fps_monitor.update()

        # Resize frame to save resources
        im0s_resized = cv2.resize(im0s, (640, 360))  # Low resolution for Jetson Nano
        # Process frame with Video Stitching
        if is_first_frame:
            video_mosaic = ImprovedVideoMosaic(im0s_resized, output_height_times=2, output_width_times=3, detector_type="orb")
            is_first_frame = False
        else:
            video_mosaic.process_frame(im0s_resized)
            im0s_resized = video_mosaic.output_img  # Use panorama as new frame

        # Preprocessing and inference
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.float() / 255.0
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            preds = model(im)

        with dt[2]:
            preds = non_max_suppression(preds, conf_thres, iou_thres)

        # Process detections
        for det in preds:
            seen += 1
            im0 = im0s_resized.copy()
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                outputs = tracker.update(det.cpu(), im0)  # Use tracker for object tracking
                if len(outputs) > 0:
                    for output in outputs:
                        bbox = output[0:4]
                        id = int(output[4])
                        cls = int(output[5])

                        # Speed calculation
                        x, y, x2, y2 = bbox
                        center_x = (x + x2) / 2
                        center_y = (y + y2) / 2

                        # Update speed estimator
                        speed, track_points = speed_estimator.update(id, center_x, center_y, timestamp_ms, class_name=names[cls])

                        # Create label with ID, class and speed if available
                        label = f"{id} {names[cls]}"
                        if speed is not None:
                            label += f" {speed:.1f} km/h"
                        
                        # Draw bounding box and label
                        color = colors(cls, True)
                        annotator.box_label(bbox, label, color=color)

                        # Draw tracking path if enabled
                        if draw_tracks and track_points and len(track_points) > 1:
                            speed_estimator.draw_track(im0, track_points, color=color, thickness=2)

            # Add FPS information
            fps_monitor.draw(im0)
            
            # Show frame
            if show_vid:
                cv2.imshow("Panorama with Speed Estimation", im0)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    if out is not None:
                        out.release()
                    exit()
                
            # Save to output video if enabled
            if save_output and out is not None:
                # Resize to output size if needed
                output_frame = cv2.resize(im0, (640, 480))
                out.write(output_frame)

        # Log progress
        if frame_count % 30 == 0:  # Log every 30 frames
            LOGGER.info(f"Frame {frame_count}, FPS: {current_fps:.1f}, Tracking {len(tracker.tracked_objects) if hasattr(tracker, 'tracked_objects') else 0} objects")

    # Clean up
    speed_estimator.end()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % tuple(x.t / seen * 1E3 for x in dt))
    LOGGER.info(f'Results saved to {save_dir}')

def main():
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run()

if __name__ == "__main__":
    main()
