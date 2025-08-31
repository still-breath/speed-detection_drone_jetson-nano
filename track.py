import argparse
import cv2
import os
import time
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

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

# Kelas SpeedEstimator untuk mengestimasi kecepatan kendaraan
class SpeedEstimator:
    def __init__(self, distance_meters=10.0, save_dir=None, fixed_fps=16.0):
        self.tracks = {}
        self.distance_meters = distance_meters  # Jarak antara garis deteksi dalam meter
        self.save_dir = save_dir
        self.fixed_fps = fixed_fps  # FPS tetap untuk kalkulasi yang konsisten
        
        if save_dir:
            self.speed_log_file = open(save_dir / 'speed_log.txt', 'w')
        else:
            self.speed_log_file = None
    
    def update(self, object_id, y_position, frame_idx, fps):
        # Gunakan fixed_fps untuk kalkulasi kecepatan yang konsisten
        fps = self.fixed_fps
        
        if object_id not in self.tracks:
            self.tracks[object_id] = {
                'positions': [],  # Track semua posisi y untuk deteksi arah
                'entry_time': None,
                'exit_time': None,
                'entry_zone': None,
                'exit_zone': None,
                'speed': None,
                'passed_zone': False,
                'captured': False,
                'frames': []  # Track frame indexes
            }
        
        track = self.tracks[object_id]
        track['positions'].append(y_position)
        track['frames'].append(frame_idx)
        
        # Definisi zona deteksi kecepatan
        zone1_y1, zone1_y2 = 120, 130  # Zona 1
        zone2_y1, zone2_y2 = 220, 230  # Zona 2
        
        # Tentukan apakah sedang berada di dalam zona 1 atau 2
        in_zone1 = zone1_y1 <= y_position <= zone1_y2
        in_zone2 = zone2_y1 <= y_position <= zone2_y2
        
        # Catat waktu masuk zona pertama yang dilewati
        if track['entry_time'] is None and (in_zone1 or in_zone2):
            track['entry_time'] = frame_idx
            track['entry_zone'] = 1 if in_zone1 else 2
        
        # Catat waktu masuk zona kedua yang dilewati (setelah memasuki zona pertama)
        if track['entry_time'] is not None and track['exit_time'] is None:
            # Jika masuk dari zona 1, cek apakah sudah masuk zona 2
            if track['entry_zone'] == 1 and in_zone2:
                track['exit_time'] = frame_idx
                track['exit_zone'] = 2
            # Jika masuk dari zona 2, cek apakah sudah masuk zona 1
            elif track['entry_zone'] == 2 and in_zone1:
                track['exit_time'] = frame_idx
                track['exit_zone'] = 1
            
            # Hitung kecepatan jika sudah melewati kedua zona
            if track['exit_time'] is not None:
                # Waktu yang diperlukan untuk melewati kedua zona dalam detik
                time_seconds = (track['exit_time'] - track['entry_time']) / fps
                
                if time_seconds > 0:
                    # Kecepatan = jarak / waktu (km/jam)
                    speed = (self.distance_meters / time_seconds) * 3.6
                    track['speed'] = round(speed, 1)
                    track['passed_zone'] = True
                    
                    # Log kecepatan
                    direction = "down" if track['exit_zone'] > track['entry_zone'] else "up"
                    if self.speed_log_file:
                        self.speed_log_file.write(f"ID: {object_id}, Speed: {track['speed']} km/h, Direction: {direction}, Frame: {frame_idx}\n")
                        self.speed_log_file.flush()
        
        return track['speed'], track['passed_zone']
    
    def get_direction(self, object_id):
        """Tentukan arah pergerakan objek"""
        if object_id in self.tracks:
            track = self.tracks[object_id]
            if track['entry_zone'] is not None and track['exit_zone'] is not None:
                if track['exit_zone'] > track['entry_zone']:
                    return "down"
                else:
                    return "up"
        return None
    
    def capture(self, frame, x, y, w, h, speed, obj_id, save_dir):
        if obj_id in self.tracks and self.tracks[obj_id]['passed_zone'] and not self.tracks[obj_id]['captured']:
            # Simpan gambar kendaraan yang terdeteksi melaju
            save_path = save_dir / 'speeds'
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Crop gambar kendaraan
            vehicle_img = frame[y:y+h, x:x+w]
            
            # Tentukan arah untuk label
            direction = self.get_direction(obj_id)
            direction_label = f"_{direction}" if direction else ""
            
            # Simpan gambar dengan label kecepatan
            cv2.imwrite(str(save_path / f"id_{obj_id}_speed_{speed}{direction_label}.jpg"), vehicle_img)
            
            self.tracks[obj_id]['captured'] = True
    
    def get_speed(self, object_id):
        if object_id in self.tracks and self.tracks[object_id]['speed'] is not None:
            return self.tracks[object_id]['speed']
        return 0
    
    def draw_zones(self, frame, roi_offset=(0, 0)):
        # ROI offset: (x_offset, y_offset)
        x_offset, y_offset = roi_offset
        
        # Zona kecepatan - garis atas
        cv2.line(frame, (0 + x_offset, 120 + y_offset), (frame.shape[1] + x_offset, 120 + y_offset), (0, 0, 255), 2)
        cv2.line(frame, (0 + x_offset, 130 + y_offset), (frame.shape[1] + x_offset, 130 + y_offset), (0, 0, 255), 2)
        
        # Zona kecepatan - garis bawah
        cv2.line(frame, (0 + x_offset, 220 + y_offset), (frame.shape[1] + x_offset, 220 + y_offset), (0, 0, 255), 2)
        cv2.line(frame, (0 + x_offset, 230 + y_offset), (frame.shape[1] + x_offset, 230 + y_offset), (0, 0, 255), 2)
        
        # Label zona - menggunakan label netral (tidak hanya start/end)
        cv2.putText(frame, "SPEED ZONE 1", (10 + x_offset, 115 + y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "SPEED ZONE 2", (10 + x_offset, 215 + y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Tambahkan informasi kalibrasi FPS
        cv2.putText(frame, f"CALIBRATED FOR {self.fixed_fps} FPS", (10 + x_offset, 30 + y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def end(self):
        if self.speed_log_file:
            self.speed_log_file.close()


# Kelas untuk menghitung FPS
class FPSCalculator:
    def __init__(self, avg_frames=30):
        self.frames = []
        self.avg_frames = avg_frames
        self.last_time = time.time()
    
    def update(self):
        # Catat waktu saat ini
        current_time = time.time()
        
        # Hitung interval waktu antara frame
        interval = current_time - self.last_time
        self.last_time = current_time
        
        # Tambahkan interval ke daftar frame
        self.frames.append(1 / interval if interval > 0 else 0)
        
        # Batasi jumlah frame yang disimpan
        if len(self.frames) > self.avg_frames:
            self.frames.pop(0)
        
        # Hitung rata-rata FPS
        return sum(self.frames) / len(self.frames)

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
        speed_zone_distance=10.0,  # Jarak antara garis speed zone dalam meter
        draw_speed_zones=True,  # Tampilkan garis zona kecepatan
        fixed_fps=16.0,  # FPS tetap untuk perhitungan kecepatan
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    is_seg = '-seg' in str(yolo_weights)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size

    # Dataloader
    bs = 1
    if webcam:
        show_vid = check_imshow(warn=True)
        dataset = LoadStreams(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        bs = len(dataset)
    else:
        dataset = LoadImages(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * bs
    
    # Inisialisasi SpeedEstimator dengan fixed_fps
    speed_estimator = SpeedEstimator(
        distance_meters=speed_zone_distance,
        save_dir=save_dir,
        fixed_fps=fixed_fps
    )
    fps_calculator = FPSCalculator()

    # Run tracking
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs
    frame_idx = 0
    
    # Get video FPS info from dataset
    try:
        if webcam:
            fps = dataset.fps[0]
        else:
            if hasattr(dataset, 'cap'):
                fps = dataset.cap.get(cv2.CAP_PROP_FPS)
            else:
                fps = 30  # Default FPS jika tidak bisa didapatkan
    except:
        fps = 30  # Default FPS jika tidak bisa didapatkan

  # Tampilkan informasi fixed FPS yang digunakan
    LOGGER.info(f"Using fixed FPS of {fixed_fps} for speed estimation")

    
    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            preds = model(im, augment=augment, visualize=visualize)

        # Apply NMS
        with dt[2]:
            if is_seg:
                masks = []
                p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                proto = preds[1][-1]
            else:
                p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            
        # Process detections
        for i, det in enumerate(p):  # detections per image
            seen += 1
            if webcam:  # bs >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            # Hitung FPS
            current_fps = fps_calculator.update()
            
            # Gambar zona kecepatan jika diaktifkan
            if draw_speed_zones:
                speed_estimator.draw_zones(im0)
        
        # Tampilkan FPS di frame beserta info kalibrasi
            cv2.putText(im0, f"Real FPS: {current_fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(im0, f"Speed calc: {fixed_fps} FPS", (3, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                if is_seg:
                    shape = im0.shape
                    # scale bbox first the crop masks
                    if retina_masks:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                        masks.append(process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2]))  # HWC
                    else:
                        masks.append(process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                else:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)

                if len(outputs[i]) > 0:
                    if is_seg:
                        # Mask plotting
                        annotator.masks(
                            masks[i],
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                            255 if retina_masks else im[i]
                        )            
                    for j, (output) in enumerate(outputs[i]):
                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]
                
                        # Perhitungan kecepatan
                        x, y, x2, y2 = bbox
                        center_y = (y + y2) / 2
                
                # Update info kecepatan
                        speed, passed_zone = speed_estimator.update(int(id), center_y, frame_idx, fixed_fps)

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                    
                            # Dapatkan arah pergerakan
                            direction = speed_estimator.get_direction(id)
                            direction_text = f" ({direction})" if direction else ""
                    
                            # Tampilkan label dengan kecepatan
                            speed_text = f"{speed} km/h{direction_text}" if speed else ""
                            if hide_labels:
                                label = None
                            else:
                                if speed_text:
                                    label = f"{id} {names[c]} {speed_text}"
                                    color = (c, True)
                                else:
                                    label = f"{id} {names[c]}" if hide_conf else (f"{id} {conf:.2f}" if hide_class else f"{id} {names[c]} {conf:.2f}")
                                    color = colors(c, True)
                    
                            annotator.box_label(bbox, label, color=color)
                                           
                            # Simpan gambar kendaraan yang melaju
                            if speed and passed_zone:
                                speed_estimator.capture(im0, int(x), int(y), int(x2-x), int(y2-y), speed, id, save_dir)
                                
                            if save_trajectories and tracking_method == 'strongsort':
                                q = output[7]
                                tracker_list[i].trajectory(im0, q, color=color)
                            
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(np.array(bbox, dtype=np.int16), imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                            
            else:
                pass
                #tracker_list[i].tracker.pred_n_update_all_tracks()
                
            # Stream results
            im0 = annotator.result()
            if show_vid:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
            
        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)
        
    # Tutup estimator kecepatan
    speed_estimator.end()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov8s-seg.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='bytetrack', help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[320], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    
    # Parameter untuk estimasi kecepatan
    parser.add_argument('--speed-zone-distance', type=float, default=20.0, help='Distance between speed zone lines in meters')
    parser.add_argument('--draw-speed-zones', default=True, help='Display speed zone lines')
    parser.add_argument('--fixed-fps', type=float, default=16.0, help='Fixed FPS value used for speed calculation (default: 16.0)')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
