import cv2
import os
import time
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

# Limit CPU usage for Jetson Nano
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
from pathlib import Path
from filterpy.kalman import KalmanFilter

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
from scipy.stats import iqr
from scipy.ndimage import gaussian_filter

class RobustKalmanFilter:
    """Enhanced Kalman Filter with outlier rejection and adaptive process noise"""
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
        
        # Measurement function (we only observe x, y positions)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Initial state uncertainty
        self.kf.P = np.eye(4) * 100
        
        # Measurement noise - initially generous
        self.kf.R = np.eye(2) * 5
        
        # Process noise - higher values allow faster changes in velocity
        q = 0.05  # Process noise magnitude (will be adapted dynamically)
        self.kf.Q = np.array([
            [q/4*dt**4, 0, q/2*dt**3, 0],
            [0, q/4*dt**4, 0, q/2*dt**3],
            [q/2*dt**3, 0, q*dt**2, 0],
            [0, q/2*dt**3, 0, q*dt**2]
        ])
        
        # Initialize state with zeros
        self.kf.x = np.array([0, 0, 0, 0])
        
        # Flag to check if the filter has been initialized
        self.initialized = False
        
        # Store history for outlier detection
        self.measurement_history = []
        self.max_history = 10
        
        # Adaptive noise parameters
        self.base_process_noise = q
        self.min_process_noise = 0.01
        self.max_process_noise = 0.2

    def predict(self, dt=None):
        """Predict next position."""
        if self.initialized:
            dt = dt if dt is not None else 0.033
            self.kf.F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            q = self.base_process_noise
            self.kf.Q = np.array([
                [q/4*dt**4, 0, q/2*dt**3, 0],
                [0, q/4*dt**4, 0, q/2*dt**3],
                [q/2*dt**3, 0, q*dt**2, 0],
                [0, q/2*dt**3, 0, q*dt**2]
            ])
            self.kf.predict()
        return self.kf.x[:2]

    def update(self, measurement):
        """Update filter with new measurement with outlier rejection."""
        if not self.initialized:
            # If first measurement, initialize the state
            self.kf.x = np.array([measurement[0], measurement[1], 0, 0])
            self.initialized = True
            self.measurement_history.append(measurement)
            return self.kf.x
        
        # Check if measurement is an outlier
        if len(self.measurement_history) >= 3:
            # Calculate prediction
            predicted_pos = self.kf.x[:2]
            
            # Calculate Mahalanobis distance
            innovation = measurement - predicted_pos
            S = self.kf.S  # Innovation covariance
            if S is not None and S.size > 0:
                try:
                    mahalanobis_dist = np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation)
                    
                    # Reject outliers (adjust threshold as needed)
                    if mahalanobis_dist > 5.0:  # Chi-square threshold for 95% confidence
                        # Use prediction instead of measurement for extreme outliers
                        return self.kf.x
                except np.linalg.LinAlgError:
                    pass  # Fall back to normal update if inversion fails
        
        # Store measurement history
        self.measurement_history.append(measurement)
        if len(self.measurement_history) > self.max_history:
            self.measurement_history.pop(0)
        
        # Adapt process noise based on recent measurements variability
        if len(self.measurement_history) >= 3:
            measurements = np.array(self.measurement_history)
            # Calculate variance in recent positions
            pos_variance = np.mean(np.var(measurements, axis=0))
            
            # Scale process noise accordingly (more variance = more process noise)
            adaptive_q = max(min(self.base_process_noise * (1 + pos_variance / 10), 
                                self.max_process_noise), 
                            self.min_process_noise)
            
            # Update Q matrix with adaptive noise
            dt = 0.033  # Same as initialization
            self.kf.Q = np.array([
                [adaptive_q/4*dt**4, 0, adaptive_q/2*dt**3, 0],
                [0, adaptive_q/4*dt**4, 0, adaptive_q/2*dt**3],
                [adaptive_q/2*dt**3, 0, adaptive_q*dt**2, 0],
                [0, adaptive_q/2*dt**3, 0, adaptive_q*dt**2]
            ])
        
        # Perform the regular Kalman update
        self.kf.update(measurement)
        return self.kf.x

    def get_position(self):
        """Get current position from filter."""
        return self.kf.x[:2]
    
    def get_velocity(self):
        """Get current velocity from filter."""
        return self.kf.x[2:4]

class NCCFrameRegistration:
    """Frame registration using Normalized Cross-Correlation and feature matching"""
    def __init__(self, first_frame, max_features=500, template_size=100):
        self.prev_frame = first_frame.copy()
        self.prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.template_size = template_size
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Extract keypoints and descriptors from first frame
        self.prev_kp, self.prev_des = self.orb.detectAndCompute(self.prev_gray, None)
        
        # Initialize template regions for NCC
        h, w = self.prev_gray.shape
        self.templates = []
        self.template_points = []
        
        # Create templates from different regions of the image
        regions = [(0, 0), (w//3, 0), (2*w//3, 0), 
                   (0, h//3), (w//3, h//3), (2*w//3, h//3),
                   (0, 2*h//3), (w//3, 2*h//3), (2*w//3, 2*h//3)]
        
        for x, y in regions:
            # Ensure template is within image bounds
            if x + template_size <= w and y + template_size <= h:
                self.templates.append(self.prev_gray[y:y+template_size, x:x+template_size])
                self.template_points.append((x + template_size//2, y + template_size//2))
        
        # Motion history for smoothing
        self.motion_history = []
        self.max_history = 5
        
        # Homography matrix for coordinate transformation
        self.H_cumulative = np.eye(3)
    
    def register_frame(self, current_frame):
        """Register current frame with previous frame using combined NCC and feature matching"""
        # Convert to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # 1. NCC-based registration
        template_shifts = []
        
        for i, template in enumerate(self.templates):
            # Define search region (larger than template)
            tx, ty = self.template_points[i]
            search_size = self.template_size * 2
            sx = max(0, tx - self.template_size//2)
            sy = max(0, ty - self.template_size//2)
            
            # Ensure search region is within image bounds
            search_w = min(search_size, current_gray.shape[1] - sx)
            search_h = min(search_size, current_gray.shape[0] - sy)
            
            if search_w > template.shape[1] and search_h > template.shape[0]:
                search_region = current_gray[sy:sy+search_h, sx:sx+search_w]
                
                # Perform template matching
                try:
                    result = cv2.matchTemplate(search_region, template, cv2.TM_CCORR_NORMED)
                    _, _, _, max_loc = cv2.minMaxLoc(result)
                    
                    # Calculate shift
                    shift_x = (sx + max_loc[0]) - (tx - self.template_size//2)
                    shift_y = (sy + max_loc[1]) - (ty - self.template_size//2)
                    template_shifts.append((shift_x, shift_y))
                except:
                    continue  # Skip this template if there's an error
        
        # 2. Feature-based registration (ORB + BFMatcher)
        current_kp, current_des = self.orb.detectAndCompute(current_gray, None)
        feature_shifts = []
        
        if current_des is not None and self.prev_des is not None and len(current_des) > 0 and len(self.prev_des) > 0:
            try:
                matches = self.bf.match(self.prev_des, current_des)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Use only good matches
                good_matches = matches[:min(20, len(matches))]
                
                if len(good_matches) >= 4:  # Need at least 4 points for homography
                    prev_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches])
                    curr_pts = np.float32([current_kp[m.trainIdx].pt for m in good_matches])
                    
                    # Calculate shifts between matched points
                    for i in range(len(prev_pts)):
                        dx = curr_pts[i][0] - prev_pts[i][0]
                        dy = curr_pts[i][1] - prev_pts[i][1]
                        feature_shifts.append((dx, dy))
                    
                    # Try to compute homography matrix
                    H, status = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)
                    if H is not None:
                        # Update cumulative homography
                        self.H_cumulative = np.matmul(H, self.H_cumulative)
            except:
                pass  # Continue if feature matching fails
        
        # Combine shifts from both methods
        all_shifts = template_shifts + feature_shifts
        
        if len(all_shifts) > 0:
            # Remove outliers using IQR
            dx_values = [shift[0] for shift in all_shifts]
            dy_values = [shift[1] for shift in all_shifts]
            
            # Calculate median and IQR
            median_dx = np.median(dx_values)
            median_dy = np.median(dy_values)
            iqr_dx = iqr(dx_values) if len(dx_values) > 2 else 10
            iqr_dy = iqr(dy_values) if len(dy_values) > 2 else 10
            
            # Filter out outliers
            filtered_shifts = [(dx, dy) for dx, dy in all_shifts 
                              if abs(dx - median_dx) < 1.5 * iqr_dx and abs(dy - median_dy) < 1.5 * iqr_dy]
            
            if len(filtered_shifts) > 0:
                # Calculate average shift
                avg_dx = sum(shift[0] for shift in filtered_shifts) / len(filtered_shifts)
                avg_dy = sum(shift[1] for shift in filtered_shifts) / len(filtered_shifts)
                
                # Store in motion history
                self.motion_history.append((avg_dx, avg_dy))
                if len(self.motion_history) > self.max_history:
                    self.motion_history.pop(0)
                
                # Calculate smoothed motion
                smoothed_dx = sum(motion[0] for motion in self.motion_history) / len(self.motion_history)
                smoothed_dy = sum(motion[1] for motion in self.motion_history) / len(self.motion_history)
                
                # Create translation matrix
                translation_matrix = np.eye(3)
                translation_matrix[0, 2] = -smoothed_dx
                translation_matrix[1, 2] = -smoothed_dy
                
                # If homography was computed, blend it with translation
                if hasattr(self, 'H_cumulative') and self.H_cumulative is not None:
                    # Use translation for minor corrections
                    correction_matrix = np.eye(3)
                    correction_matrix[0, 2] = -smoothed_dx * 0.5  # Half weight
                    correction_matrix[1, 2] = -smoothed_dy * 0.5
                    
                    # Apply correction to cumulative homography
                    self.H_cumulative = np.matmul(correction_matrix, self.H_cumulative)
        
        # Update for next iteration
        self.prev_frame = current_frame.copy()
        self.prev_gray = current_gray
        self.prev_kp, self.prev_des = current_kp, current_des
        
        return self.H_cumulative

    def transform_coordinates(self, points):
        """Transform coordinates from original frame to stabilized frame coordinates"""
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
        
    def warp_image(self, frame):
        """Warp the image using the current homography"""
        return cv2.warpPerspective(frame, self.H_cumulative, 
                                  (frame.shape[1], frame.shape[0]), 
                                  flags=cv2.INTER_LINEAR)

class AutoCalibration:
    """Automatic calibration for pixels_per_meter based on vehicle sizes"""
    def __init__(self, initial_pixels_per_meter=26.0):
        self.base_pixels_per_meter = initial_pixels_per_meter
        self.vehicle_lengths = {  # Average vehicle lengths in meters
            'car': 4.5,
            'truck': 7.5,
            'bus': 12.0,
            'motorcycle': 2.0,
            'bicycle': 1.7,
            'person': 0.5  # For scale reference
        }
        
        # Store observed vehicle sizes
        self.observed_vehicles = {cls: [] for cls in self.vehicle_lengths.keys()}
        self.min_samples = 5  # Minimum samples before recalibration
        self.current_pixels_per_meter = initial_pixels_per_meter
        self.calibration_weights = {  # Weights for different classes (higher = more reliable)
            'car': 1.0,
            'truck': 0.7,
            'bus': 0.6,
            'motorcycle': 0.5,
            'bicycle': 0.4,
            'person': 0.3
        }
        
        # For altitude-based adjustment
        self.reference_altitude = 20.0  # meters
        self.last_calibration_time = 0
        self.calibration_interval = 5.0  # seconds
    
    def add_vehicle_observation(self, class_name, bbox_width, bbox_height, timestamp):
        """Add observed vehicle dimensions for calibration"""
        if class_name.lower() in self.vehicle_lengths:
            # Store the longer dimension (usually vehicle length)
            size = max(bbox_width, bbox_height)
            self.observed_vehicles[class_name.lower()].append((size, timestamp))
            
            # Limit history size
            if len(self.observed_vehicles[class_name.lower()]) > 30:
                self.observed_vehicles[class_name.lower()].pop(0)
            
            # Update calibration if we have enough data and enough time has passed
            if timestamp - self.last_calibration_time > self.calibration_interval:
                self.update_calibration(timestamp)
                
    def update_calibration(self, timestamp):
        """Update pixels_per_meter based on observed vehicle sizes"""
        calibration_values = []
        total_weight = 0
        
        for cls in self.observed_vehicles:
            observations = self.observed_vehicles[cls]
            if len(observations) >= self.min_samples:
                # Filter recent observations (last 10 seconds)
                recent_obs = [obs[0] for obs in observations 
                             if timestamp - obs[1] < 10.0]
                
                if len(recent_obs) >= self.min_samples:
                    # Remove outliers - keep values within 1.5 IQR
                    q1 = np.percentile(recent_obs, 25)
                    q3 = np.percentile(recent_obs, 75)
                    iqr_val = q3 - q1
                    filtered_sizes = [size for size in recent_obs 
                                     if q1 - 1.5*iqr_val <= size <= q3 + 1.5*iqr_val]
                    
                    if len(filtered_sizes) >= 3:
                        # Calculate median size
                        median_size = np.median(filtered_sizes)
                        
                        # Convert to pixels_per_meter
                        expected_length = self.vehicle_lengths[cls]
                        pixels_per_meter = median_size / expected_length
                        
                        # Add to calibration values with appropriate weight
                        weight = self.calibration_weights.get(cls, 0.5)
                        calibration_values.append((pixels_per_meter, weight))
                        total_weight += weight
        
        # Calculate weighted average if we have values
        if total_weight > 0:
            weighted_sum = sum(val * weight for val, weight in calibration_values)
            new_pixels_per_meter = weighted_sum / total_weight
            
            # Apply smoothing (blend with previous value)
            alpha = 0.3  # Weight for new value (0.3 = 30% new, 70% old)
            self.current_pixels_per_meter = (1 - alpha) * self.current_pixels_per_meter + alpha * new_pixels_per_meter
            self.last_calibration_time = timestamp
            
            LOGGER.info(f"Updated calibration: {self.current_pixels_per_meter:.2f} pixels per meter")
            
        return self.current_pixels_per_meter
    
    def adjust_for_altitude(self, altitude_meters):
        """Adjust pixels_per_meter based on drone altitude"""
        if altitude_meters > 0:
            altitude_factor = self.reference_altitude / altitude_meters
            adjusted = self.current_pixels_per_meter * altitude_factor
            return adjusted
        return self.current_pixels_per_meter

class SpeedEstimator:
    def __init__(self, save_dir=None, pixels_per_meter=26.0, max_track_points=30, speed_window=5):
        self.tracks = {}
        self.calibration = AutoCalibration(initial_pixels_per_meter=pixels_per_meter)
        self.save_dir = save_dir
        self.max_track_points = max_track_points  # Maximum number of tracking points to keep
        self.speed_window = speed_window  # Number of frames to average for speed calculation
        
        # Tunable parameters
        self.speed_correction_factor = 0.95  # Adjust based on validation
        self.alpha = 0.15  # Smoothing factor
        self.median_window = 5  # Window for median filtering
        
        # Create a log file for speed data
        if save_dir:
            self.speed_log_file = open(save_dir / 'speed_log.txt', 'w')
            # Write header
            self.speed_log_file.write("ID,Class,Time(s),X,Y,Speed(km/h),Distance(m),Confidence\n")
        else:
            self.speed_log_file = None
            
        # Constants for speed calculation
        self.dt_threshold = 0.2  # Minimum time difference for speed calculation (seconds)
        self.min_points_for_speed = 3  # Minimum points required for speed estimation
        
        # Speed validation thresholds
        self.max_plausible_speed = 160.0  # km/h
        self.min_plausible_speed = 1.0    # km/h
        
        # Confidence tracking
        self.confidence_threshold = 0.6  # Minimum confidence to report speed

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def update(self, object_id, x, y, timestamp_ms, class_name=None, bbox_width=None, bbox_height=None, altitude=None, background_motion=(0.0, 0.0)):
        """Update tracking information for an object."""
        pixels_per_meter = self.calibration.adjust_for_altitude(altitude) if altitude else self.calibration.current_pixels_per_meter

        current_time = timestamp_ms / 1000.0  # Convert from milliseconds to seconds
        
        # Update calibration if we have bbox dimensions
        if bbox_width is not None and bbox_height is not None and class_name is not None:
            self.calibration.add_vehicle_observation(class_name, bbox_width, bbox_height, current_time)
        
        if object_id not in self.tracks:
            # Initialize a new track with Robust Kalman Filter
            self.tracks[object_id] = {
                'points': [],          # Track all (x,y) positions
                'filtered_points': [], # Track filtered (x,y) positions
                'times': [],           # Track time for each position (seconds)
                'velocities': [],      # Instantaneous velocities
                'speeds': [],          # Speed history
                'current_speed': None, # Current smoothed speed
                'confidence': 0.0,     # Confidence in speed estimate
                'total_distance': 0.0, # Total distance traveled
                'initial_time': current_time,
                'class_name': class_name,
                'kalman_filter': RobustKalmanFilter()
            }
            
        track = self.tracks[object_id]
        
        # Store raw position and time
        track['points'].append((float(x), float(y)))
        track['times'].append(current_time)
        
        # Update class name if provided
        if class_name is not None:
            track['class_name'] = class_name

        # Use Kalman filter to smooth position
        dt = track['times'][-1] - track['times'][-2] if len(track['times']) >= 2 else 1.0 / 30.0
        track['kalman_filter'].predict(dt)
        measurement = np.array([x, y])
        filtered_state = track['kalman_filter'].update(measurement)
        x_filtered, y_filtered = filtered_state[:2]
        track['filtered_points'].append((float(x_filtered), float(y_filtered)))
        
        # Calculate speed only if we have enough points
        if len(track['filtered_points']) >= 2:
            # Calculate distance between the last two filtered points
            last_point = track['filtered_points'][-2]
            current_point = track['filtered_points'][-1]
            
            dx_bg, dy_bg = background_motion
            adjusted_last_point = (last_point[0] + dx_bg, last_point[1] + dy_bg)
            adjusted_current_point = (current_point[0] + dx_bg, current_point[1] + dy_bg)
            
            # Calculate distance between adjusted points
            distance = self.calculate_distance(adjusted_last_point, adjusted_current_point)
            distance_meters = distance / pixels_per_meter
            track['total_distance'] += distance_meters
            
            # Calculate time difference
            dt = track['times'][-1] - track['times'][-2]
            
            # Calculate instantaneous velocity if time difference is significant
            if dt > 0.001:  # Avoid division by zero
                velocity = distance_meters / dt  # m/s
                velocity *= self.speed_correction_factor
                speed_kmh = velocity * 3.6  # Convert m/s to km/h
                
                # Basic validation - speed should be within reasonable limits
                if self.min_plausible_speed <= speed_kmh <= self.max_plausible_speed:
                    track['velocities'].append(speed_kmh)
                else:
                    track['velocities'].append(track['velocities'][-1] if track['velocities'] else 0)
                
                # Apply median filtering for robustness
                if len(track['velocities']) >= self.median_window:
                    recent_speeds = track['velocities'][-self.median_window:]
                    median_speed = np.median(recent_speeds)
                    
                    # Calculate speed variance for confidence
                    speed_variance = np.var(recent_speeds)
                    
                    # Update confidence (low variance = high confidence)
                    confidence_factor = 1.0 / (1.0 + speed_variance / 10.0)
                    track['confidence'] = min(0.95, confidence_factor)  # Cap at 0.95
                else:
                    median_speed = track['velocities'][-1]
                    # Lower confidence for few samples
                    track['confidence'] = min(0.4, len(track['velocities']) * 0.1)

                # Apply additional smoothing for the speed
                if track['current_speed'] is None:
                    track['current_speed'] = median_speed
                else:
                    # Exponential moving average for smoothing
                    track['current_speed'] = (1 - self.alpha) * track['current_speed'] + self.alpha * median_speed
                
                track['speeds'].append(track['current_speed'])
                
                # Log speed information
                if self.speed_log_file and len(track['speeds']) % 5 == 0:  # Log every 5 updates
                    self.speed_log_file.write(f"{object_id},{track['class_name']},{current_time:.3f},"
                                            f"{x_filtered:.1f},{y_filtered:.1f},{track['current_speed']:.1f},"
                                            f"{track['total_distance']:.2f},{track['confidence']:.2f}\n")
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
        
        # Only return speed if confidence is high enough
        reported_speed = track['current_speed'] if track['confidence'] >= self.confidence_threshold else None
        return reported_speed, track['filtered_points'], track['confidence']

    def draw_track(self, frame, track_points, color=(230, 230, 230), thickness=2):
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
    
    def draw(self, frame):
        # Display FPS in the top-left corner
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


class PerformanceOptimizer:
    """Optimize performance based on device capabilities (especially for Jetson Nano)"""
    def __init__(self, target_fps=15.0):
        self.target_fps = target_fps
        self.current_fps = 0
        self.processing_scale = 1.0  # Scale factor for input resolution
        self.skip_frames = 0
        self.last_adaptation_time = time.time()
        self.adaptation_interval = 5.0  # seconds between adaptations
        
        # Check if running on Jetson platform
        self.is_jetson = 'tegra' in platform.processor().lower() if hasattr(platform, 'processor') else False
        
        # If on Jetson, start with reduced resolution
        if self.is_jetson:
            self.processing_scale = 0.7
            self.skip_frames = 1
            
        LOGGER.info(f"Performance optimizer initialized on {'Jetson' if self.is_jetson else 'standard'} platform")
    
    def update(self, fps):
        """Update performance settings based on current FPS"""
        self.current_fps = fps
        current_time = time.time()
        
        # Only adapt periodically
        if current_time - self.last_adaptation_time < self.adaptation_interval:
            return self.processing_scale, self.skip_frames
            
        self.last_adaptation_time = current_time
        
        # Adjust settings based on performance
        if self.current_fps < self.target_fps * 0.7:  # Significantly below target
            # Reduce resolution and increase frame skipping
            self.processing_scale = max(0.5, self.processing_scale - 0.1)
            self.skip_frames = min(2, self.skip_frames + 1)
            LOGGER.info(f"Performance: Reducing resolution scale to {self.processing_scale:.2f}, skip frames: {self.skip_frames}")
        elif self.current_fps > self.target_fps * 1.3:  # Significantly above target
            # Can afford to improve quality
            self.processing_scale = min(1.0, self.processing_scale + 0.1)
            self.skip_frames = max(0, self.skip_frames - 1)
            LOGGER.info(f"Performance: Increasing resolution scale to {self.processing_scale:.2f}, skip frames: {self.skip_frames}")
            
        return self.processing_scale, self.skip_frames


@torch.no_grad()
def run(
        source='moving20_20km.mp4',
        yolo_weights='caraerial-320_fp16.engine',
        imgsz=(320, 320),
        conf_thres=0.35,
        iou_thres=0.45,
        device='0',
        show_vid=True,
        line_thickness=1,
        hide_labels=False,
        hide_conf=True,
        vid_stride=1,
        pixels_per_meter=26.0,  # Initial calibration: pixels per meter
        draw_tracks=True,       # Draw tracking paths
        save_output=True,       # Save output video
        target_fps=30.0,        # Target FPS for performance optimization
        stabilize=False,         # Enable frame stabilization
        auto_calibrate=True     # Enable automatic calibration
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

    # Initialize Performance Optimizer
    perf_optimizer = PerformanceOptimizer(target_fps=target_fps)
    
    # Initialize Frame Registration for stabilization
    frame_registration = None
    
    # Initialize FPS monitor
    fps_monitor = FPSMonitor(avg_frames=10)

    # Initialize SpeedEstimator with calibration
    speed_estimator = SpeedEstimator(
        save_dir=save_dir, 
        pixels_per_meter=pixels_per_meter,
        max_track_points=30,  # Keep 30 track points
        speed_window=5        # Use 5 frames for speed averaging
    )

    # Verify that ByteTrack config file exists
    tracking_config = ROOT / 'trackers' / 'bytetrack' / 'configs' / 'bytetrack.yaml'
    
    # Initialize the tracker
    tracker = create_tracker('bytetrack', tracking_config, None, device, False)

    # Run tracking
    seen, dt = 0, (Profile(), Profile(), Profile())
    frame_count = 0
    skip_frame_counter = 0
    
    # Create a debugging visualization window if needed
    debug_window_name = "Speed Estimation Debug"
    if show_vid:
        cv2.namedWindow(debug_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(debug_window_name, 640, 480)
    
    # Open a CSV file for saving speed data
    speed_csv = open(save_dir / 'vehicle_speeds.csv', 'w')
    speed_csv.write("frame,id,class,x,y,speed,confidence\n")
    
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        timestamp_ms = vid_cap.get(cv2.CAP_PROP_POS_MSEC) if vid_cap is not None else (frame_idx * 1000 / fps)
        frame_count += 1
        
        # Apply frame skipping based on performance optimization
        scale_factor, skip_frames = perf_optimizer.update(fps_monitor.fps)
        
        if skip_frame_counter < skip_frames:
            skip_frame_counter += 1
            continue
        else:
            skip_frame_counter = 0
        
        # Update FPS monitor
        current_fps = fps_monitor.update()

        # Resize frame based on performance optimization
        if scale_factor < 1.0:
            new_width = int(im0s.shape[1] * scale_factor)
            new_height = int(im0s.shape[0] * scale_factor)
            im0s_resized = cv2.resize(im0s, (new_width, new_height))
        else:
            im0s_resized = im0s

        # Apply frame registration/stabilization
        frame_registration=None
        if stabilize:
            if frame_registration is None:
                frame_registration = NCCFrameRegistration(im0s_resized)
                stabilized_frame = im0s_resized
            else:
                # Register frame and transform coordinates
                frame_registration.register_frame(im0s_resized)
                stabilized_frame = frame_registration.warp_image(im0s_resized)
        else:
            stabilized_frame = im0s_resized

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
            im0 = stabilized_frame.copy()
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                outputs = tracker.update(det.cpu(), im0)  # Use tracker for object tracking
                
                if len(outputs) > 0:
                    for output in outputs:
                        bbox = output[0:4]
                        id = int(output[4])
                        cls = int(output[5])

                        # Extract bbox information
                        x, y, x2, y2 = bbox
                        bbox_width = x2 - x
                        bbox_height = y2 - y
                        center_x = (x + x2) / 2
                        center_y = (y + y2) / 2
                        
                        # Get class name
                        class_name = names[cls]
                        
                        # If using stabilization, transform coordinates
                        dx_bg, dy_bg = 0.0, 0.0
                        if stabilize and frame_registration is not None:
                            # We use original coordinates for tracking
                            dx_bg = frame_registration.H_cumulative[0, 2]
                            dy_bg = frame_registration.H_cumulative[1, 2]

                        # Update speed estimator with auto-calibration
                        # Assuming a fixed drone altitude for now (replace with sensor data if available)
                        drone_altitude = 20.0  # meters
                        effective_ppm = speed_estimator.calibration.current_pixels_per_meter * scale_factor
                        speed, track_points, confidence = speed_estimator.update(
                            id, center_x, center_y, timestamp_ms, 
                            class_name=class_name,
                            bbox_width=bbox_width * scale_factor, 
                            bbox_height=bbox_height * scale_factor,
                            altitude=drone_altitude if auto_calibrate else None,
                            background_motion=(dx_bg, dy_bg)
                        )

                        # Create label with ID, class and speed if available
                        label = f"{id} {names[cls]}"
                        if speed is not None:
                            label += f" {speed:.1f}km/h"
                            
                            # Save to CSV
                            speed_csv.write(f"{frame_count},{id},{class_name},{center_x:.1f},{center_y:.1f},{speed:.1f},{confidence:.2f}\n")
                        
                        # Draw bounding box and label
                        color = colors(cls, True)
                        annotator.box_label(bbox, label, color=color)

                        # Draw tracking path if enabled
                        if draw_tracks and track_points and len(track_points) > 1:
                            speed_estimator.draw_track(im0, track_points, color=color, thickness=2)

            # Add FPS information and calibration info
            fps_monitor.draw(im0)
            
            # Add calibration information
            cv2.putText(im0, f"Calibration: {speed_estimator.calibration.current_pixels_per_meter:.1f} px/m", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frame
            if show_vid:
                cv2.imshow(debug_window_name, im0)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    if out is not None:
                        out.release()
                    speed_csv.close()
                    exit()
                
            # Save to output video if enabled
            if save_output and out is not None:
                # Resize to output size if needed
                output_frame = cv2.resize(im0, (640, 480))
                out.write(output_frame)

        # Log progress
        if frame_count % 30 == 0:  # Log every 30 frames
            LOGGER.info(f"Frame {frame_count}, FPS: {current_fps:.1f}, Calibration: {speed_estimator.calibration.current_pixels_per_meter:.1f} px/m")

    # Clean up
    speed_estimator.end()
    speed_csv.close()
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
