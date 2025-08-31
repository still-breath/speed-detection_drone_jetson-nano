import cv2
import os
import time
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import threading
import queue

# Limit CPU usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
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

def custom_colors(cls, bgr=False):
    color_map = {
        0: (0, 0, 255),    # Class 0: Red in BGR
        1: (128, 0, 0),    # Class 1: Dark Blue in BGR
        2: (0, 128, 0),    # Class 2: Dark Green in BGR
    }
    if cls in color_map:
        color = color_map[cls]
    else:
        color = [int((cls / 80) * 255), int(((cls % 80) / 80) * 255), int(((cls % 80) / 80) * 255)]
        color = tuple(color)
    return color if bgr else color[::-1]

class Load360pBase:
    """Base class with common functionality for video and stream processing"""
    def __init__(self, source, img_size=320, stride=32, auto=True):
        self.source = source
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.cap = None
        self.fps = 0
        self.frames = 0
        self.target_height = 720
        self.target_width = 1280
        self.frame_queue = queue.Queue(maxsize=120)
        self.stop_event = threading.Event()
        
    def preprocess_frame(self, im0):
        try:
            img = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]
            img = img.transpose((2, 0, 1))
            img = np.ascontiguousarray(img)
            return img
        except Exception as e:
            LOGGER.error(f"Error preprocessing frame: {str(e)}")
            blank = np.zeros((3, self.img_size, self.img_size), dtype=np.uint8)
            return blank
            
    def release(self):
        self.stop_event.set()
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()

class Load360pVideo(Load360pBase):
    """Handles local video files with conversion to 360p"""
    def __init__(self, source='video.mp4', img_size=320, stride=32, auto=True):
        super().__init__(source, img_size, stride, auto)
        self.connection_healthy = True  # Local file assumed to be healthy
        
        if not self.initialize_capture():
            LOGGER.error(f"Failed to open video file: {self.source}")
            return
            
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()
        time.sleep(1.0)  # Allow some time for the queue to populate

    def initialize_capture(self):
        LOGGER.info(f"Opening video file: {self.source}")
        try:
            if self.cap is not None:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                LOGGER.error(f"Failed to open {self.source}")
                return False

            self.fps = max(self.cap.get(cv2.CAP_PROP_FPS), 30)
            self.frames = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)

            self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if self.original_width <= 0 or self.original_height <= 0:
                LOGGER.warning("Invalid video dimensions. Using defaults (640x360).")
                self.original_width = 640
                self.original_height = 360

            # Calculate aspect ratio to maintain proportions
            aspect_ratio = self.original_width / self.original_height
            if abs(aspect_ratio - (16/9)) > 0.1:
                self.target_width = int(self.target_height * aspect_ratio)
                LOGGER.info(f"Adjusted target resolution to {self.target_width}x{self.target_height}")

            success, test_frame = self.cap.read()
            if not success or test_frame is None:
                LOGGER.error("Failed to read initial frame from video")
                return False

            # Reset to beginning of video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return True
        except Exception as e:
            LOGGER.error(f"Error initializing video: {str(e)}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False

    def _reader(self):
        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                LOGGER.error("Video file closed unexpectedly")
                time.sleep(0.5)
                continue

            success, frame = self.cap.read()
            if not success or frame is None:
                if self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.frames - 1:
                    LOGGER.info("End of video file reached")
                    # For video files, we can loop back to the beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    LOGGER.warning("Failed to read frame from video")
                    time.sleep(0.1)
                    continue

            try:
                frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
            except Exception as e:
                LOGGER.error(f"Error resizing frame: {str(e)}")
                continue

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            if timestamp_ms <= 0:
                timestamp_ms = time.time() * 1000
                
            self.frame_queue.put((frame, timestamp_ms))
            
            # For local videos, we don't need to process as fast as possible
            # This helps reduce CPU usage
            time.sleep(1.0 / self.fps)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        try:
            im0, timestamp_ms = self.frame_queue.get(timeout=5.0)
            success = True
        except queue.Empty:
            LOGGER.warning("Timeout waiting for frame, video may have ended")
            # Return a blank frame if queue is empty (end of video)
            blank_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
            return self.source, np.zeros((3, self.img_size, self.img_size)), blank_frame, self.cap, 0, "END"

        self.count += 1
        img = self.preprocess_frame(im0)
        return self.source, img, im0, self.cap, timestamp_ms, f"{self.count}/{self.frames}"

    def __len__(self):
        return self.frames

class Load360pStream(Load360pBase):
    """Handles RTMP/network streams with conversion to 360p"""
    def __init__(self, source='rtmp://example.com/live', img_size=320, stride=32, auto=True):
        super().__init__(source, img_size, stride, auto)
        self.connection_healthy = False
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 2
        self.reconnect_backoff_factor = 1.5
        self.last_frame_time = time.time()

        for attempt in range(self.max_reconnect_attempts):
            if self.initialize_capture():
                break
            reconnect_delay = self.reconnect_delay * (self.reconnect_backoff_factor ** attempt)
            LOGGER.warning(f"Failed to connect to stream. Retry in {reconnect_delay:.1f}s ({attempt+1}/{self.max_reconnect_attempts})")
            time.sleep(reconnect_delay)

        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()
        time.sleep(1.0)

    def initialize_capture(self):
        LOGGER.info(f"Opening RTMP stream: {self.source}")
        try:
            if self.cap is not None:
                self.cap.release()
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;10000000|probesize;10000000"
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 120)

            if not self.cap.isOpened():
                LOGGER.error(f"Failed to open {self.source}")
                return False

            self.fps = max(self.cap.get(cv2.CAP_PROP_FPS), 30)
            if self.fps > 1000:
                self.fps = 30
            self.frames = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)

            self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if self.original_width <= 0 or self.original_height <= 0:
                LOGGER.warning("Invalid stream dimensions received. Using defaults (640x360).")
                self.original_width = 640
                self.original_height = 360

            aspect_ratio = self.original_width / self.original_height
            if abs(aspect_ratio - (16/9)) > 0.1:
                self.target_width = int(self.target_height * aspect_ratio)
                LOGGER.info(f"Adjusted target resolution to {self.target_width}x{self.target_height}")

            success, test_frame = self.cap.read()
            if not success or test_frame is None:
                LOGGER.error("Failed to read initial frame from stream")
                return False

            self.connection_healthy = True
            return True
        except Exception as e:
            LOGGER.error(f"Error initializing stream: {str(e)}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False

    def _reader(self):
        consecutive_errors = 0
        max_consecutive_errors = 30
        last_reconnect_time = 0
        reconnect_interval = 5

        while not self.stop_event.is_set():
            try:
                if self.cap is None or not self.cap.isOpened():
                    current_time = time.time()
                    if current_time - last_reconnect_time > reconnect_interval:
                        LOGGER.info(f"Stream closed. Attempting to reconnect...")
                        if self.initialize_capture():
                            consecutive_errors = 0
                        last_reconnect_time = current_time
                    time.sleep(1)
                    continue

                success, frame = self.cap.read()
                if not success or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        current_time = time.time()
                        if current_time - last_reconnect_time > reconnect_interval:
                            LOGGER.warning(f"Too many consecutive errors ({consecutive_errors}). Reconnecting...")
                            if self.initialize_capture():
                                consecutive_errors = 0
                            last_reconnect_time = current_time
                    time.sleep(0.1)
                    continue

                consecutive_errors = 0
                self.last_frame_time = time.time()

                try:
                    frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
                except Exception as e:
                    LOGGER.error(f"Error resizing frame: {str(e)}")
                    continue

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                if timestamp_ms <= 0:
                    timestamp_ms = time.time() * 1000
                self.frame_queue.put((frame, timestamp_ms))
            except Exception as e:
                LOGGER.error(f"Error in reader thread: {str(e)}")
                consecutive_errors += 1
                time.sleep(0.5)
            time.sleep(0.001)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        while True:
            try:
                im0, timestamp_ms = self.frame_queue.get(block=True, timeout=5.0)
                success = True
                break
            except queue.Empty:
                # Keep trying until we get a frame
                LOGGER.warning("Waiting for frame...")
                if self.cap is None or not self.cap.isOpened():
                    self.initialize_capture()
                time.sleep(0.5)
                continue

        self.count += 1
        img = self.preprocess_frame(im0)
        return self.source, img, im0, self.cap, timestamp_ms, f"{self.count}/{self.frames}"

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

class SpeedEstimator:
    def __init__(self, save_dir=None, pixels_per_meter=15.0, max_track_points=30):
        self.tracks = {}
        self.pixels_per_meter = pixels_per_meter
        self.save_dir = save_dir
        self.max_track_points = max_track_points
        self.speed_log_file = open(save_dir / 'speed_log.txt', 'w') if save_dir else None

    def calculate_distance(self, point1, point2):
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def total_distance(self, polyline):
        distance = 0
        for i in range(1, len(polyline)):
            distance += self.calculate_distance(polyline[i-1], polyline[i])
        return distance / self.pixels_per_meter

    def update(self, object_id, x, y, timestamp_ms):
        current_time = timestamp_ms / 1000.0
        if object_id not in self.tracks:
            self.tracks[object_id] = {
                'points': [],
                'times': [],
                'distances': [],
                'speeds': [],
                'current_speed': None,
                'initial_time': current_time
            }
        track = self.tracks[object_id]
        track['points'].append((float(x), float(y)))
        track['times'].append(current_time)

        if len(track['points']) > 1:
            distance = self.total_distance(track['points'][-2:])
            total_distance = track['distances'][-1] + distance if track['distances'] else distance
            track['distances'].append(total_distance)

            elapsed_time = current_time - track['times'][-2]
            MIN_ELAPSED_TIME = 0.001
            MAX_REASONABLE_SPEED = 150.0

            if elapsed_time > MIN_ELAPSED_TIME:
                speed = (distance / elapsed_time) * 3.6
                track['speeds'].append(speed)
                smoothed_speed = sum(track['speeds'][-3:]) / min(len(track['speeds']), 3) if track['speeds'] else speed
                
                if speed > MAX_REASONABLE_SPEED:
                    speed = track['current_speed'] if track['current_speed'] is not None else 0
                else:
                    track['current_speed'] = round(smoothed_speed, 1)

                if self.speed_log_file and len(track['speeds']) % 5 == 0:
                    self.speed_log_file.write(f"ID: {object_id}, Speed: {track['current_speed']} km/h, Time: {current_time:.3f}s, Distance: {total_distance:.2f}m\n")
                    self.speed_log_file.flush()
            else:
                speed = track['current_speed'] if track['current_speed'] is not None else 0

        if len(track['points']) > self.max_track_points:
            track['points'].pop(0)
            track['times'].pop(0)
            if track['distances']:
                track['distances'].pop(0)
            if track['speeds']:
                track['speeds'].pop(0)

        return track['current_speed'], track['points']

    def draw_track(self, frame, track_points, color=(230, 230, 230), thickness=2):
        if len(track_points) > 1:
            points = np.array(track_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=thickness)

    def end(self):
        if self.speed_log_file:
            self.speed_log_file.close()

class FPSMonitor:
    def __init__(self, avg_frames=10):
        self.prev_time = time.time()
        self.curr_time = self.prev_time
        self.frame_times = []
        self.avg_frames = avg_frames
        self.fps = 0

    def update(self):
        self.curr_time = time.time()
        dt = self.curr_time - self.prev_time
        self.prev_time = self.curr_time
        self.frame_times.append(dt)
        if len(self.frame_times) > self.avg_frames:
            self.frame_times.pop(0)
        avg_time = sum(self.frame_times) / len(self.frame_times)
        self.fps = 1.0 / avg_time if avg_time > 0 else 0
        return self.fps

    def draw(self, frame):
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

@torch.no_grad()
def run(
        source="40m_day.mp4",
        yolo_weights='caraerial-320_fp16.engine',
        imgsz=(320, 320),
        conf_thres=0.25,
        iou_thres=0.35,
        device='0',
        show_vid=True,
        line_thickness=1,
        hide_labels=False,
        hide_conf=True,
        vid_stride=1,
        pixels_per_meter=10.72,
        draw_tracks=False,
        reconnect_attempts=float('inf'),  # Set to infinity for continuous reconnects
        reconnect_delay=3,
        convert_to_360p=False,
        rtmp_options=True,
):
    source = str(source)
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    if rtmp_options and is_url and source.lower().startswith('rtmp://'):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;10000000|probesize;10000000|fflags;nobuffer|max_delay;500000"
        LOGGER.info("Applied RTMP-specific FFMPEG options")

    project = ROOT / 'runs' / 'track'
    name = 'exp'
    exist_ok = False
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = AutoBackend(yolo_weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    names = ["car", "motor", "truck"]
    imgsz = check_imgsz(imgsz, stride=stride)

    tracking_config = ROOT / 'trackers' / 'ocsort' / 'configs' / 'ocsort.yaml'
    reid_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt'
    tracker = create_tracker('ocsort', tracking_config, reid_weights, device, False)

    speed_estimator = SpeedEstimator(
        save_dir=save_dir,
        pixels_per_meter=pixels_per_meter
    )

    fps_monitor = FPSMonitor()

    seen, dt = 0, (Profile(), Profile(), Profile())
    reconnect_count = 0
    dataset = None

    while True:  # Infinite loop to keep trying
        try:
            if convert_to_360p:
                if webcam and is_url and source.lower().startswith('rtmp://'):
                    LOGGER.info(f"Using threaded 360p converter for RTMP stream: {source}")
                    dataset = Load360pStream(source, img_size=imgsz[0], stride=stride, auto=pt)
                    frame_iter = enumerate(dataset)
                elif is_file:
                    LOGGER.info(f"Using threaded 360p converter for video file: {source}")
                    dataset = Load360pVideo(source, img_size=imgsz[0], stride=stride, auto=pt)
                    frame_iter = enumerate(dataset)
                else:
                    if webcam:
                        dataset = LoadStreams(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
                    else:
                        dataset = LoadImages(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
                    frame_iter = enumerate(dataset)
            else:
                if webcam:
                    dataset = LoadStreams(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
                else:
                    dataset = LoadImages(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
                frame_iter = enumerate(dataset)

            reconnect_count = 0

            for frame_idx, data in frame_iter:
                if isinstance(dataset, (Load360pStream, Load360pVideo)):
                    path, im, im0s, vid_cap, timestamp_ms, s = data
                elif webcam:
                    path, im, im0s, vid_cap, s = data[0], data[1], data[2], data[3], data[4]
                    im = im[0]
                    im0s = im0s[0]
                    timestamp_ms = time.time() * 1000
                else:
                    path, im, im0s, vid_cap, s = data
                    timestamp_ms = vid_cap.get(cv2.CAP_PROP_POS_MSEC)

                current_fps = fps_monitor.update()

                with dt[0]:
                    im = torch.from_numpy(im).to(device)
                    im = im.float()
                    im /= 255.0
                    if len(im.shape) == 3:
                        im = im[None]

                with dt[1]:
                    preds = model(im)

                with dt[2]:
                    preds = non_max_suppression(preds, conf_thres, iou_thres)

                for i, det in enumerate(preds):
                    seen += 1
                    im0 = im0s.copy()
                    h, w = im0.shape[:2]
                    resolution_text = f"{w}x{h}"
                    cv2.putText(im0, resolution_text, (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                    if det is not None and len(det):
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                        outputs = tracker.update(det.cpu(), im0)

                        if len(outputs) > 0:
                            for output in outputs:
                                bbox = output[0:4]
                                id = int(output[4])
                                cls = int(output[5])
                                x, y, x2, y2 = bbox
                                center_x = (x + x2) / 2
                                center_y = (y + y2) / 2
                                speed, track_points = speed_estimator.update(id, center_x, center_y, timestamp_ms)
                                label = f"{id} {names[cls]}"
                                if speed:
                                    label += f" {speed} km/h"
                                if not hide_labels:
                                    annotator.box_label(bbox, label, color=custom_colors(cls, True))
                                elif not hide_conf:
                                    annotator.box_label(bbox, label, color=custom_colors(cls, True))
                                if draw_tracks and track_points and len(track_points) > 1:
                                    speed_estimator.draw_track(im0, track_points, color=colors(id % 10, True), thickness=2)

                    im0 = annotator.result()
                    fps_monitor.draw(im0)

                    if frame_idx % 30 == 0:
                        LOGGER.info(f"Current FPS: {current_fps:.1f}")

                    if show_vid:
                        window_title = "Drone Tracking (360p)" if convert_to_360p else "Drone Tracking"
                        cv2.imshow(window_title, im0)
                        key = cv2.waitKey(1)
                        if key == ord('q'):
                            speed_estimator.end()
                            cv2.destroyAllWindows()
                            if isinstance(dataset, (Load360pStream, Load360pVideo)):
                                dataset.release()
                            return

                if frame_idx % 100 == 0:
                    LOGGER.info(f"{s}{'' if det is not None and len(det) else '(no detections), '}"
                               f"{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

        except (cv2.error, IOError, OSError) as e:
            reconnect_count += 1
            LOGGER.error(f"Stream/video connection error: {e}")
            LOGGER.info(f"Reconnection attempt {reconnect_count}...")
            time.sleep(reconnect_delay)
        finally:
            if 'dataset' in locals() and dataset is not None:
                if isinstance(dataset, (Load360pStream, Load360pVideo)):
                    dataset.release()

    t = tuple(x.t / seen * 1E3 for x in dt if hasattr(x, 't') and seen > 0)
    if t:
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    speed_estimator.end()
    cv2.destroyAllWindows()

def main():
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run()

if __name__ == "__main__":
    main()
