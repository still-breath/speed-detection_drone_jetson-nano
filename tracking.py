import cv2
import os
import time
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import sys
import logging

# Limit CPU usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, colorstr
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker


def custom_colors(cls, bgr=False):
    """Custom color mapping for specific classes"""
    color_map = {
        0: (0, 0, 255),    # Red for class 0
        1: (128, 0, 0),    # Dark Blue for class 1
        2: (0, 128, 0),    # Dark Green for class 2
    }
    return color_map.get(cls, colors(cls, bgr))


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints"""
    shape = im.shape[:2]
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
        source="uji40.mp4",
        yolo_weights='opset10-fp16-malam-fix.engine',
        imgsz=(320, 320),
        conf_thres=0.25,
        iou_thres=0.35,
        device='0',
        show_vid=True,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        vid_stride=1,
        draw_tracks=False,
):
    source = str(source)
    is_file = Path(source).suffix[1:] in VID_FORMATS
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

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

    fps_monitor = FPSMonitor()
    track_history = {}

    dataset = LoadStreams(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=vid_stride) if webcam \
        else LoadImages(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    seen, dt = 0, (Profile(), Profile(), Profile())

    try:
        for frame_idx, data in enumerate(dataset):
            path, im, im0s, vid_cap, s = data if not webcam else data[0]

            im = im[0] if webcam else im
            im0 = im0s.copy()

            with dt[0]:
                im_tensor = torch.from_numpy(im).to(device)
                im_tensor = im_tensor.float() / 255.0
                if len(im_tensor.shape) == 3:
                    im_tensor = im_tensor[None]

            with dt[1]:
                preds = model(im_tensor)

            with dt[2]:
                preds = non_max_suppression(preds, conf_thres, iou_thres)

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            for det in preds:
                seen += 1
                h, w = im0.shape[:2]
                resolution_text = f"{w}x{h}"
                cv2.putText(im0, resolution_text, (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

                if len(det):
                    det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
                    outputs = tracker.update(det.cpu(), im0)

                    for output in outputs:
                        x1, y1, x2, y2, id, cls, conf = output
                        id, cls, conf = int(id), int(cls), float(conf)
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)

                        if id not in track_history:
                            track_history[id] = []
                        track_history[id].append(center)
                        if len(track_history[id]) > 30:
                            track_history[id].pop(0)

                        label = ''
                        if not hide_labels:
                            label += f'{id} {names[cls]}'
                        if not hide_conf:
                            label += f' {conf:.2f}' if label else f'{conf:.2f}'

                        annotator.box_label([x1, y1, x2, y2], label, color=custom_colors(cls, True))

                        if draw_tracks and len(track_history[id]) > 1:
                            pts = np.array(track_history[id], np.int32).reshape((-1, 1, 2))
                            cv2.polylines(im0, [pts], isClosed=False, color=colors(id % 10, True), thickness=2)

            im0 = annotator.result()
            fps_monitor.update()
            fps_monitor.draw(im0)

            if frame_idx % 30 == 0:
                LOGGER.info(f"Current FPS: {fps_monitor.fps:.1f}")

            if show_vid:
                cv2.imshow("Detection and Tracking", im0)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

            if frame_idx % 100 == 0:
                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}"
                           f"{sum([d.dt for d in dt if hasattr(d, 'dt')]) * 1E3:.1f}ms")

    except Exception as e:
        LOGGER.error(f"Error during processing: {e}")
    finally:
        cv2.destroyAllWindows()

    t = tuple(x.t / seen * 1E3 for x in dt if hasattr(x, 't') and seen > 0)
    if t:
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)


def main():
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run()

if __name__ == "__main__":
    main()
