import cv2
import numpy as np
import argparse
import os
import time
import threading
import queue
from pathlib import Path
import math
import json

class RTMP_Stream_Reader:
    """Handles reading frames from RTMP stream with reconnection capability"""
    def __init__(self, source, img_size=640):
        self.source = source
        self.img_size = img_size
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.connection_healthy = False
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 2
        self.reconnect_backoff_factor = 1.5
        self.last_frame_time = time.time()
        self.target_height = 720  # Higher resolution for better calibration
        self.target_width = 1280
        
        # Set FFMPEG options for better RTMP handling
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;10000000|probesize;10000000"
        
        for attempt in range(self.max_reconnect_attempts):
            if self.initialize_capture():
                break
            reconnect_delay = self.reconnect_delay * (self.reconnect_backoff_factor ** attempt)
            print(f"Failed to connect to stream. Retry in {reconnect_delay:.1f}s ({attempt+1}/{self.max_reconnect_attempts})")
            time.sleep(reconnect_delay)
            
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()
        time.sleep(1.0)  # Allow time for initial frames
    
    def initialize_capture(self):
        """Initialize video capture from RTMP stream"""
        print(f"Opening RTMP stream: {self.source}")
        try:
            if self.cap is not None:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            
            if not self.cap.isOpened():
                print(f"Failed to open {self.source}")
                return False
                
            # Get stream properties
            self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if self.original_width <= 0 or self.original_height <= 0:
                print("Invalid stream dimensions received. Using defaults (1280x720).")
                self.original_width = 1280
                self.original_height = 720
                
            # Adjust aspect ratio if needed
            aspect_ratio = self.original_width / self.original_height
            if abs(aspect_ratio - (16/9)) > 0.1:
                self.target_width = int(self.target_height * aspect_ratio)
                print(f"Adjusted target resolution to {self.target_width}x{self.target_height}")
                
            success, test_frame = self.cap.read()
            if not success or test_frame is None:
                print("Failed to read initial frame from stream")
                return False
                
            self.connection_healthy = True
            return True
        except Exception as e:
            print(f"Error initializing stream: {str(e)}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False
    
    def _reader(self):
        """Background thread to continuously read frames"""
        consecutive_errors = 0
        max_consecutive_errors = 30
        
        while not self.stop_event.is_set():
            try:
                if self.cap is None or not self.cap.isOpened():
                    if self.initialize_capture():
                        consecutive_errors = 0
                    time.sleep(1)
                    continue
                    
                success, frame = self.cap.read()
                if not success or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"Too many consecutive errors ({consecutive_errors}). Reconnecting...")
                        if self.initialize_capture():
                            consecutive_errors = 0
                    time.sleep(0.1)
                    continue
                    
                consecutive_errors = 0
                self.last_frame_time = time.time()
                
                try:
                    # Resize for calibration purposes - keep higher resolution
                    frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
                except Exception as e:
                    print(f"Error resizing frame: {str(e)}")
                    continue
                    
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                        
                self.frame_queue.put(frame)
            except Exception as e:
                print(f"Error in reader thread: {str(e)}")
                consecutive_errors += 1
                time.sleep(0.5)
                
            time.sleep(0.001)  # Small sleep to prevent CPU hogging
    
    def read(self):
        """Read a frame from the queue"""
        try:
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            return False, None
    
    def release(self):
        """Release resources"""
        self.stop_event.set()
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()


class CalibrationTool:
    """Tool for calibrating pixels-per-meter from RTMP drone stream"""
    def __init__(self, rtmp_url, known_object_length=4.0, camera_height=30.0, camera_fov=84.0):
        self.rtmp_url = rtmp_url
        self.known_object_length = known_object_length  # Length in meters (e.g., typical car length)
        self.camera_height = camera_height  # Drone height in meters
        self.camera_fov = camera_fov  # Camera FOV in degrees
        
        # Points for line measurement
        self.points = []
        self.current_point = None
        self.dragging = False
        
        # Calibration results
        self.pixels_per_meter = None
        self.calibrations = []
        self.avg_pixels_per_meter = None
        
        # Frame information
        self.frame = None
        self.display_frame = None
        self.frame_width = 1280
        self.frame_height = 720
        
        # Camera parameters
        self.camera_fov_rad = math.radians(camera_fov)
        
        # Path for saving calibration data
        self.save_dir = Path('calibration_data')
        self.save_dir.mkdir(exist_ok=True)
        
        # UI state
        self.mode = 'draw'  # 'draw' or 'view'
        self.show_instructions = True
        self.show_grid = False
        
        # Initialize stream reader
        self.stream = RTMP_Stream_Reader(rtmp_url)
        
        # Load previous calibrations if available
        self.load_calibrations()
        
    def load_calibrations(self):
        """Load previously saved calibrations"""
        calibration_file = self.save_dir / 'calibration_history.json'
        if calibration_file.exists():
            try:
                with open(calibration_file, 'r') as f:
                    data = json.load(f)
                    if 'calibrations' in data:
                        self.calibrations = data['calibrations']
                        if len(self.calibrations) > 0:
                            self.avg_pixels_per_meter = sum([c['pixels_per_meter'] for c in self.calibrations]) / len(self.calibrations)
                            print(f"Loaded {len(self.calibrations)} previous calibrations.")
                            print(f"Average pixels per meter: {self.avg_pixels_per_meter:.2f}")
            except Exception as e:
                print(f"Error loading calibrations: {str(e)}")
    
    def save_calibration(self):
        """Save current calibration to history"""
        if self.pixels_per_meter is None:
            return
            
        # Add current calibration to history
        calibration_data = {
            'pixels_per_meter': self.pixels_per_meter,
            'camera_height': self.camera_height,
            'camera_fov': self.camera_fov,
            'known_object_length': self.known_object_length,
            'timestamp': time.time(),
            'date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.calibrations.append(calibration_data)
        
        # Calculate average
        self.avg_pixels_per_meter = sum([c['pixels_per_meter'] for c in self.calibrations]) / len(self.calibrations)
        
        # Save to file
        calibration_file = self.save_dir / 'calibration_history.json'
        try:
            with open(calibration_file, 'w') as f:
                json.dump({
                    'calibrations': self.calibrations,
                    'average_pixels_per_meter': self.avg_pixels_per_meter
                }, f, indent=2)
            print(f"Calibration saved. Average: {self.avg_pixels_per_meter:.2f} pixels/meter")
        except Exception as e:
            print(f"Error saving calibration: {str(e)}")
    
    def clear_calibrations(self):
        """Clear all saved calibrations"""
        self.calibrations = []
        self.avg_pixels_per_meter = None
        
        # Save empty calibration file
        calibration_file = self.save_dir / 'calibration_history.json'
        try:
            with open(calibration_file, 'w') as f:
                json.dump({'calibrations': []}, f)
            print("Calibration history cleared.")
        except Exception as e:
            print(f"Error clearing calibrations: {str(e)}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse interactions for drawing measurement line"""
        if self.mode == 'draw':
            if event == cv2.EVENT_LBUTTONDOWN:
                # Start drawing a line
                if len(self.points) < 2:
                    self.points.append((x, y))
                    self.current_point = (x, y)
                    self.dragging = True
                else:
                    # Reset points when we already have a line
                    self.points = [(x, y)]
                    self.current_point = (x, y)
                    self.dragging = True
                    
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
                # Update current point while dragging
                self.current_point = (x, y)
                
            elif event == cv2.EVENT_LBUTTONUP and self.dragging:
                # Finish drawing a line
                if len(self.points) < 2:
                    self.points.append((x, y))
                else:
                    self.points[1] = (x, y)
                self.dragging = False
                
                # Calculate distance if we have two points
                if len(self.points) == 2:
                    self.calculate_pixels_per_meter()
    
    def calculate_pixels_per_meter(self):
        """Calculate pixels per meter based on drawn line and known object length"""
        if len(self.points) != 2:
            return
            
        # Calculate pixel distance
        dx = self.points[1][0] - self.points[0][0]
        dy = self.points[1][1] - self.points[0][1]
        pixel_distance = math.sqrt(dx*dx + dy*dy)
        
        # Simple perspective correction based on distance from center
        center_x, center_y = self.frame_width // 2, self.frame_height // 2
        
        # Calculate average distance from image center for both points
        dist1_from_center = math.sqrt((self.points[0][0] - center_x)**2 + (self.points[0][1] - center_y)**2)
        dist2_from_center = math.sqrt((self.points[1][0] - center_x)**2 + (self.points[1][1] - center_y)**2)
        avg_dist_from_center = (dist1_from_center + dist2_from_center) / 2
        
        # Normalize by image diagonal
        image_diagonal = math.sqrt(self.frame_width**2 + self.frame_height**2)
        normalized_dist = avg_dist_from_center / (image_diagonal / 2)
        
        # Simple perspective correction factor (objects further from center cover more ground distance per pixel)
        # This is a simplified model - in reality would need proper camera calibration
        perspective_factor = 1.0 + normalized_dist * 0.5
        
        # Calculate pixels per meter
        self.pixels_per_meter = pixel_distance / (self.known_object_length * perspective_factor)
        
        print(f"Measured {pixel_distance:.1f} pixels for {self.known_object_length:.1f} meters")
        print(f"Perspective factor: {perspective_factor:.2f}")
        print(f"Calculated {self.pixels_per_meter:.2f} pixels per meter")
        
        # Calculate theoretical pixels per meter based on camera parameters
        # This serves as a sanity check
        fov_width_rad = self.camera_fov_rad
        view_width_at_ground = 2 * self.camera_height * math.tan(fov_width_rad / 2)
        theoretical_px_per_m = self.frame_width / view_width_at_ground
        
        print(f"Theoretical pixels per meter at {self.camera_height}m height: {theoretical_px_per_m:.2f}")
        print(f"Difference: {(self.pixels_per_meter - theoretical_px_per_m) / theoretical_px_per_m * 100:.1f}%")
        
    def draw_grid(self, frame, grid_size=5):
        """Draw a grid with known real-world size (in meters)"""
        if self.pixels_per_meter is None or self.pixels_per_meter <= 0:
            return frame
            
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Make a copy of the frame
        grid_frame = frame.copy()
        
        # Draw horizontal and vertical lines every 'grid_size' meters
        pixels_per_grid = int(grid_size * self.pixels_per_meter)
        
        # Calculate how many grid lines we need in each direction
        num_horizontal = max(1, h // pixels_per_grid)
        num_vertical = max(1, w // pixels_per_grid)
        
        # Draw horizontal grid lines
        for i in range(-num_horizontal, num_horizontal + 1):
            y = center_y + i * pixels_per_grid
            if 0 <= y < h:
                cv2.line(grid_frame, (0, int(y)), (w, int(y)), (0, 255, 0), 1)
                # Label the line with distance in meters
                cv2.putText(grid_frame, f"{i*grid_size}m", (5, int(y) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw vertical grid lines
        for i in range(-num_vertical, num_vertical + 1):
            x = center_x + i * pixels_per_grid
            if 0 <= x < w:
                cv2.line(grid_frame, (int(x), 0), (int(x), h), (0, 255, 0), 1)
                # Label the line with distance in meters
                cv2.putText(grid_frame, f"{i*grid_size}m", (int(x) + 5, 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw center crosshair
        cv2.line(grid_frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 2)
        cv2.line(grid_frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 2)
        
        return grid_frame
    
    def run(self):
        """Main calibration tool loop"""
        window_name = "Drone Calibration Tool"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        while True:
            # Read frame from stream
            ret, frame = self.stream.read()
            if not ret or frame is None:
                print("Waiting for frame...")
                time.sleep(0.5)
                continue
                
            self.frame = frame.copy()
            self.frame_height, self.frame_width = frame.shape[:2]
            self.display_frame = frame.copy()
            
            # Draw measurement line
            if len(self.points) >= 1:
                # Draw first point
                cv2.circle(self.display_frame, self.points[0], 5, (0, 0, 255), -1)
                
                # Draw line to current position or second point
                if len(self.points) == 1 and self.current_point:
                    cv2.line(self.display_frame, self.points[0], self.current_point, (0, 255, 0), 2)
                elif len(self.points) >= 2:
                    cv2.line(self.display_frame, self.points[0], self.points[1], (0, 255, 0), 2)
                    cv2.circle(self.display_frame, self.points[1], 5, (0, 0, 255), -1)
                    
                    # Calculate and display distance
                    dx = self.points[1][0] - self.points[0][0]
                    dy = self.points[1][1] - self.points[0][1]
                    pixel_distance = math.sqrt(dx*dx + dy*dy)
                    midpoint = ((self.points[0][0] + self.points[1][0]) // 2, 
                               (self.points[0][1] + self.points[1][1]) // 2)
                    
                    if self.pixels_per_meter is not None and self.pixels_per_meter > 0:
                        real_distance = pixel_distance / self.pixels_per_meter
                        cv2.putText(self.display_frame, f"{pixel_distance:.1f}px = {real_distance:.2f}m", 
                                  (midpoint[0] + 10, midpoint[1]), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(self.display_frame, f"{pixel_distance:.1f}px", 
                                  (midpoint[0] + 10, midpoint[1]), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw grid if enabled
            if self.show_grid and self.pixels_per_meter is not None:
                self.display_frame = self.draw_grid(self.display_frame)
            
            # Show calibration results
            if self.pixels_per_meter is not None:
                cv2.putText(self.display_frame, f"Calibration: {self.pixels_per_meter:.2f} pixels/meter", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if self.avg_pixels_per_meter is not None:
                    cv2.putText(self.display_frame, f"Average ({len(self.calibrations)} samples): {self.avg_pixels_per_meter:.2f} pixels/meter", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show camera parameters
            cv2.putText(self.display_frame, f"Camera Height: {self.camera_height:.1f}m  FOV: {self.camera_fov:.1f}°", 
                      (10, self.frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show instructions
            if self.show_instructions:
                instructions = [
                    "Instructions:",
                    "- Click and drag to measure object of known length",
                    f"- Known object length: {self.known_object_length:.1f}m (Press +/- to adjust)",
                    "- Press 'S' to save calibration",
                    "- Press 'G' to toggle grid",
                    "- Press 'C' to clear points",
                    "- Press 'R' to reset all calibrations",
                    "- Press 'H' to hide instructions",
                    "- Press 'Esc' to exit"
                ]
                
                y_offset = 120
                for line in instructions:
                    cv2.putText(self.display_frame, line, (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 25
            
            # Show the frame
            cv2.imshow(window_name, self.display_frame)
            
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('c'):
                # Clear points
                self.points = []
                self.current_point = None
            elif key == ord('s'):
                # Save calibration
                if self.pixels_per_meter is not None:
                    self.save_calibration()
            elif key == ord('g'):
                # Toggle grid
                self.show_grid = not self.show_grid
            elif key == ord('h'):
                # Toggle instructions
                self.show_instructions = not self.show_instructions
            elif key == ord('r'):
                # Reset all calibrations
                self.clear_calibrations()
            elif key == ord('+') or key == ord('='):
                # Increase known object length
                self.known_object_length += 0.1
                print(f"Known object length: {self.known_object_length:.1f}m")
                if len(self.points) == 2:
                    self.calculate_pixels_per_meter()
            elif key == ord('-') or key == ord('_'):
                # Decrease known object length
                self.known_object_length = max(0.1, self.known_object_length - 0.1)
                print(f"Known object length: {self.known_object_length:.1f}m")
                if len(self.points) == 2:
                    self.calculate_pixels_per_meter()
            elif key == ord('['):
                # Decrease camera height
                self.camera_height = max(1.0, self.camera_height - 1.0)
                print(f"Camera height: {self.camera_height:.1f}m")
            elif key == ord(']'):
                # Increase camera height
                self.camera_height += 1.0
                print(f"Camera height: {self.camera_height:.1f}m")
        
        # Clean up
        self.stream.release()
        cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Drone RTMP Stream Calibration Tool')
    parser.add_argument('--rtmp', type=str, default='rtmp://192.168.57.100:1935/streams',
                      help='RTMP stream URL')
    parser.add_argument('--length', type=float, default=4.0,
                      help='Known object length in meters (default: 4.0 = typical car length)')
    parser.add_argument('--height', type=float, default=30.0,
                      help='Camera/drone height in meters (default: 30.0)')
    parser.add_argument('--fov', type=float, default=84.0,
                      help='Camera field of view in degrees (default: 84.0 for DJI Phantom 4 Pro)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    print("Drone RTMP Stream Calibration Tool")
    print(f"RTMP URL: {args.rtmp}")
    print(f"Known object length: {args.length}m")
    print(f"Camera height: {args.height}m")
    print(f"Camera FOV: {args.fov}°")
    print("\nStarting calibration tool...")
    
    calibration_tool = CalibrationTool(
        rtmp_url=args.rtmp,
        known_object_length=args.length,
        camera_height=args.height,
        camera_fov=args.fov
    )
    
    calibration_tool.run()
