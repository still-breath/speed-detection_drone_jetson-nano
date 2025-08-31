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

# Import TensorRT dan libraries untuk confusion matrix
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT not available, using original model inference")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm
import json

class TensorRTInference:
    def __init__(self, engine_path: str):
        """Initialize TensorRT inference engine"""
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT is not available")
            
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
        
        self._load_engine()
        self._allocate_buffers()
    
    def _load_engine(self):
        """Load TensorRT engine"""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
    
    def _allocate_buffers(self):
        """Allocate GPU memory for inputs and outputs"""
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append to the appropriate list
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        
        # Create CUDA stream
        self.stream = cuda.Stream()
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data"""
        # Copy input data to host buffer
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # Transfer input data to GPU
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Execute inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer predictions back from GPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        # Synchronize stream
        self.stream.synchronize()
        
        return self.outputs[0]['host'].copy()

class ModelEvaluator:
    def __init__(self, class_names: list, iou_threshold: float = 0.5):
        """Initialize model evaluator for confusion matrix generation"""
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.iou_threshold = iou_threshold
        
        # Initialize confusion matrix (classes + background)
        self.confusion_matrix = np.zeros((self.num_classes + 1, self.num_classes + 1), dtype=int)
        
        # Store all predictions and ground truths for detailed analysis
        self.all_predictions = []
        self.all_ground_truths = []
        
    def update_confusion_matrix(self, pred_detections: list, gt_detections: list, image_name: str = ""):
        """Update confusion matrix with predictions and ground truth"""
        # Store for later analysis
        self.all_predictions.extend(pred_detections)
        self.all_ground_truths.extend(gt_detections)
        
        # Create matched pairs
        matched_preds = set()
        matched_gts = set()
        
        # Match predictions to ground truth based on IoU and class
        for gt_idx, gt in enumerate(gt_detections):
            best_pred_idx = -1
            best_iou = 0.0
            
            for pred_idx, pred in enumerate(pred_detections):
                if pred_idx in matched_preds:
                    continue
                
                iou = self._calculate_iou(pred['bbox'], gt['bbox'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            
            if best_pred_idx != -1:
                pred_class = pred_detections[best_pred_idx]['class_id']
                gt_class = gt['class_id']
                
                # Update confusion matrix
                self.confusion_matrix[gt_class, pred_class] += 1
                matched_preds.add(best_pred_idx)
                matched_gts.add(gt_idx)
            else:
                # False Negative (ground truth not detected)
                self.confusion_matrix[gt['class_id'], self.num_classes] += 1
        
        # Handle unmatched predictions (False Positives)
        for pred_idx, pred in enumerate(pred_detections):
            if pred_idx not in matched_preds:
                # False Positive (prediction without matching ground truth)
                self.confusion_matrix[self.num_classes, pred['class_id']] += 1
    
    def _calculate_iou(self, box1: list, box2: list) -> float:
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def plot_confusion_matrix(self, save_path: str = None, normalize: bool = True):
        """Plot and save confusion matrix"""
        class_names_with_bg = self.class_names + ['Background']
        
        if normalize:
            cm = self.confusion_matrix.astype('float')
            cm_sum = cm.sum(axis=1)[:, np.newaxis]
            cm_sum[cm_sum == 0] = 1  # Avoid division by zero
            cm = cm / cm_sum
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            cm = self.confusion_matrix
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                    xticklabels=class_names_with_bg, 
                    yticklabels=class_names_with_bg,
                    cbar_kws={'label': 'Count' if not normalize else 'Normalized Count'})
        
        plt.title(title)
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        plt.show()
    
    def calculate_metrics(self) -> dict:
        """Calculate precision, recall, F1-score, and mAP for each class"""
        metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[self.num_classes, i]  # Background predicted as class i
            fn = self.confusion_matrix[i, self.num_classes]  # Class i predicted as background
            
            # Add wrong class predictions to FP and FN
            for j in range(self.num_classes):
                if j != i:
                    fp += self.confusion_matrix[j, i]  # Other classes predicted as class i
                    fn += self.confusion_matrix[i, j]  # Class i predicted as other classes
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn)
            }
        
        # Calculate overall metrics
        total_tp = sum([metrics[cls]['tp'] for cls in metrics])
        total_fp = sum([metrics[cls]['fp'] for cls in metrics])
        total_fn = sum([metrics[cls]['fn'] for cls in metrics])
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        metrics['overall'] = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'mAP': np.mean([metrics[cls]['f1_score'] for cls in self.class_names])
        }
        
        return metrics
    
    def save_detailed_report(self, save_path: str):
        """Save detailed evaluation report"""
        metrics = self.calculate_metrics()
        
        report = {
            'confusion_matrix': self.confusion_matrix.tolist(),
            'class_names': self.class_names,
            'metrics': metrics,
            'total_predictions': len(self.all_predictions),
            'total_ground_truths': len(self.all_ground_truths)
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Detailed report saved to: {save_path}")

def load_ground_truth_annotations(gt_path: str, class_names: list) -> dict:
    """Load ground truth annotations from file"""
    annotations = {}
    
    if gt_path.endswith('.json'):
        # Load COCO format annotations
        with open(gt_path, 'r') as f:
            data = json.load(f)
        
        # Process COCO annotations
        for ann in data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            
            bbox = ann['bbox']  # [x, y, width, height]
            # Convert to [x1, y1, x2, y2] format
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            
            annotations[image_id].append({
                'bbox': bbox,
                'class_id': ann['category_id'],
                'confidence': 1.0
            })
    
    elif gt_path.endswith('.txt'):
        # Load YOLO format annotations
        with open(gt_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                image_name = parts[0]
                class_id = int(parts[1])
                bbox = [float(x) for x in parts[2:6]]
                
                if image_name not in annotations:
                    annotations[image_name] = []
                
                annotations[image_name].append({
                    'bbox': bbox,
                    'class_id': class_id,
                    'confidence': 1.0
                })
    
    return annotations

@torch.no_grad()
def run(
        source='valid',
        yolo_weights=WEIGHTS / 'yolov5m.pt',
        tensorrt_engine='opset10-fp16-malam-fix.engine',  # Path to TensorRT engine file
        ground_truth="valid/_annotations.coco.json",  # Path to ground truth annotations
        imgsz=(320, 320),
        conf_thres=0.35,
        iou_thres=0.45,
        max_det=4000,
        device='0',
        show_vid=False,
        save_txt=False,
        save_conf=False,
        save_crop=True,
        save_vid=True,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / 'runs' / 'detect',
        name='exp',
        exist_ok=False,
        line_thickness=2,
        hide_labels=False,
        hide_conf=False,
        hide_class=False,
        half=True,
        dnn=False,
        vid_stride=1,
        retina_masks=False,
        evaluate_model=True,  # Enable model evaluation
        plot_confusion_matrix=True,  # Plot confusion matrix
        save_metrics=True,  # Save detailed metrics
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)

    # Directories
    if not isinstance(yolo_weights, list):
        exp_name = yolo_weights.stem if not tensorrt_engine else Path(tensorrt_engine).stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:
        exp_name = Path(yolo_weights[0]).stem
    else:
        exp_name = 'ensemble'
    exp_name = name if name else exp_name
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    
    # Initialize TensorRT or regular model
    trt_model = None
    if tensorrt_engine and TRT_AVAILABLE:
        try:
            trt_model = TensorRTInference(tensorrt_engine)
            LOGGER.info(f"Loaded TensorRT engine: {tensorrt_engine}")
        except Exception as e:
            LOGGER.warning(f"Failed to load TensorRT engine: {e}")
            trt_model = None
    
    # Load regular model if TensorRT is not available
    if trt_model is None:
        is_seg = '-seg' in str(yolo_weights)
        model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_imgsz(imgsz, stride=stride)
        model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))
    else:
        # For TensorRT, we need to define class names manually or load from config
        names = {i: f'class_{i}' for i in range(80)}  # Default COCO classes
        is_seg = False

    # Dataloader
    if webcam:
        show_vid = check_imshow(warn=True)
        dataset = LoadStreams(
            source,
            imgsz=imgsz,
            stride=32,  # Default stride for TensorRT
            auto=True,
            vid_stride=vid_stride
        )
    else:
        dataset = LoadImages(
            source,
            imgsz=imgsz,
            stride=32,
            auto=True,
            vid_stride=vid_stride
        )

    # Initialize evaluator if ground truth is provided
    evaluator = None
    gt_annotations = {}
    if evaluate_model and ground_truth:
        class_names_list = list(names.values()) if isinstance(names, dict) else names
        evaluator = ModelEvaluator(class_names_list, iou_threshold=iou_thres)
        gt_annotations = load_ground_truth_annotations(ground_truth, class_names_list)
        LOGGER.info(f"Loaded ground truth annotations: {len(gt_annotations)} images")

    # Run inference
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    vid_path, vid_writer = None, None
    
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            if trt_model:
                # Preprocess for TensorRT
                im_tensor = im.astype(np.float32) / 255.0
                if len(im_tensor.shape) == 3:
                    im_tensor = np.expand_dims(im_tensor, axis=0)
                
                print(f"im_tensor shape: {im_tensor.shape}")
                assert im_tensor.ndim == 4, f"Expected 4D input tensor, got {im_tensor.ndim}D"
            else:
                im = torch.from_numpy(im).to(device)
                im = im.half() if half else im.float()
                im /= 255.0
                if len(im.shape) == 3:
                    im = im[None]

        # Inference
        with dt[1]:
             if trt_model:
                 pred = trt_model.infer(im_tensor)
                 pred = torch.from_numpy(pred).to(device)
                 num_classes = 4                 
                 pred = pred.view(1, num_classes + 5, 8400).transpose(1, 2)
                 print("Pred shape after fix:", pred.shape)
             else:
                 print("Input shape:", im.shape)
                 pred = model(im, augment=augment, visualize=visualize)

# Debug input shape
        print(f"Prediction shape before NMS: {pred.shape}")

# Pastikan prediksi tidak kosong
        if pred.numel() == 0 or pred.shape[-1] < 5:
            LOGGER.warning("Empty prediction tensor. Skipping NMS.")
            pred = torch.empty((0, 6))  # [x1, y1, x2, y2, conf, cls]
        else:
    # NMS
            with dt[2]:
                if is_seg and not trt_model:
                    pred = non_max_suppression(pred[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                else:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                p = Path(p)
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)
                else:
                    txt_file_name = p.parent.name
                    save_path = str(save_dir / p.parent.name)

            txt_path = str(save_dir / 'labels' / txt_file_name)
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                if not trt_model:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Extract predictions for evaluation
                pred_detections = []
                for *xyxy, conf, cls in reversed(det):
                    pred_detections.append({
                        'bbox': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                        'confidence': float(conf),
                        'class_id': int(cls)
                    })

                # Update confusion matrix if evaluating
                if evaluator and str(p.name) in gt_annotations:
                    evaluator.update_confusion_matrix(pred_detections, gt_annotations[str(p.name)], str(p.name))

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    id_text = names[c] if hide_conf else f'{names[c]} {conf:.2f}'
                    
                    if save_txt:
                        xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or show_vid:
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if show_vid:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            # Save results
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

    # Generate evaluation results
    if evaluator:
        LOGGER.info("Generating evaluation metrics...")
        
        if plot_confusion_matrix:
            # Plot normalized confusion matrix
            evaluator.plot_confusion_matrix(
                save_path=str(save_dir / 'confusion_matrix_normalized.png'),
                normalize=True
            )
            # Plot raw confusion matrix
            evaluator.plot_confusion_matrix(
                save_path=str(save_dir / 'confusion_matrix_raw.png'),
                normalize=False
            )
        
        if save_metrics:
            # Save detailed metrics
            evaluator.save_detailed_report(str(save_dir / 'evaluation_report.json'))
            
            # Print summary metrics
            metrics = evaluator.calculate_metrics()
            print("\n" + "="*50)
            print("EVALUATION SUMMARY")
            print("="*50)
            for class_name, metric in metrics.items():
                if class_name != 'overall':
                    print(f"{class_name:15} | Precision: {metric['precision']:.3f} | Recall: {metric['recall']:.3f} | F1: {metric['f1_score']:.3f}")
            print("-"*50)
            print(f"{'Overall':15} | Precision: {metrics['overall']['precision']:.3f} | Recall: {metrics['overall']['recall']:.3f} | F1: {metrics['overall']['f1_score']:.3f} | mAP: {metrics['overall']['mAP']:.3f}")
            print("="*50)

def main():
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run()


if __name__ == "__main__":
    main()


# Additional utility functions for enhanced functionality

def benchmark_model(model_path: str, tensorrt_engine: str = None, test_images: str = None, 
                   warmup_runs: int = 10, benchmark_runs: int = 100):
    """
    Benchmark model performance comparing regular inference vs TensorRT
    
    Args:
        model_path: Path to YOLO model
        tensorrt_engine: Path to TensorRT engine (optional)
        test_images: Path to test images directory
        warmup_runs: Number of warmup runs
        benchmark_runs: Number of benchmark runs
    """
    import time
    import statistics
    
    print("="*60)
    print("MODEL PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Load test image
    if test_images:
        test_img_path = Path(test_images)
        if test_img_path.is_dir():
            img_files = list(test_img_path.glob('*.jpg')) + list(test_img_path.glob('*.png'))
            if img_files:
                test_img = cv2.imread(str(img_files[0]))
            else:
                print("No test images found, using dummy data")
                test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        else:
            test_img = cv2.imread(str(test_img_path))
    else:
        test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Prepare input
    img_tensor = cv2.resize(test_img, (640, 640))
    img_tensor = img_tensor.transpose((2, 0, 1))  # HWC to CHW
    img_tensor = np.expand_dims(img_tensor, axis=0).astype(np.float32) / 255.0
    
    # Benchmark regular model
    device = select_device('')
    model = AutoBackend(model_path, device=device)
    model.warmup(imgsz=(1, 3, 640, 640))
    
    regular_times = []
    
    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(torch.from_numpy(img_tensor).to(device))
    
    # Benchmark
    for _ in range(benchmark_runs):
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(torch.from_numpy(img_tensor).to(device))
        end_time = time.perf_counter()
        regular_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    regular_avg = statistics.mean(regular_times)
    regular_std = statistics.stdev(regular_times)
    
    print(f"Regular Model Performance:")
    print(f"  Average: {regular_avg:.2f} ± {regular_std:.2f} ms")
    print(f"  Min: {min(regular_times):.2f} ms")
    print(f"  Max: {max(regular_times):.2f} ms")
    print(f"  FPS: {1000/regular_avg:.1f}")
    
    # Benchmark TensorRT if available
    if tensorrt_engine and TRT_AVAILABLE:
        try:
            trt_model = TensorRTInference(tensorrt_engine)
            trt_times = []
            
            # Warmup
            for _ in range(warmup_runs):
                _ = trt_model.infer(img_tensor)
            
            # Benchmark
            for _ in range(benchmark_runs):
                start_time = time.perf_counter()
                _ = trt_model.infer(img_tensor)
                end_time = time.perf_counter()
                trt_times.append((end_time - start_time) * 1000)
            
            trt_avg = statistics.mean(trt_times)
            trt_std = statistics.stdev(trt_times)
            
            print(f"\nTensorRT Model Performance:")
            print(f"  Average: {trt_avg:.2f} ± {trt_std:.2f} ms")
            print(f"  Min: {min(trt_times):.2f} ms")
            print(f"  Max: {max(trt_times):.2f} ms")
            print(f"  FPS: {1000/trt_avg:.1f}")
            
            speedup = regular_avg / trt_avg
            print(f"\nSpeedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"TensorRT benchmark failed: {e}")
    
    print("="*60)


def convert_to_tensorrt(model_path: str, output_path: str, input_shape: tuple = (1, 3, 640, 640),
                       precision: str = 'fp16', workspace_size: int = 1):
    """
    Convert YOLO model to TensorRT engine
    
    Args:
        model_path: Path to YOLO model
        output_path: Output path for TensorRT engine
        input_shape: Input tensor shape (batch, channels, height, width)
        precision: Precision mode ('fp32', 'fp16', 'int8')
        workspace_size: Workspace size in GB
    """
    if not TRT_AVAILABLE:
        print("TensorRT is not available for conversion")
        return False
    
    try:
        import onnx
        from torch2trt import torch2trt
        
        print(f"Converting {model_path} to TensorRT...")
        print(f"Input shape: {input_shape}")
        print(f"Precision: {precision}")
        
        # Load model
        device = torch.device('cuda:0')
        model = AutoBackend(model_path, device=device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(device)
        
        # Convert to TensorRT
        if precision == 'fp16':
            model_trt = torch2trt(model, [dummy_input], fp16_mode=True, 
                                max_workspace_size=workspace_size*1024*1024*1024)
        elif precision == 'int8':
            model_trt = torch2trt(model, [dummy_input], int8_mode=True,
                                max_workspace_size=workspace_size*1024*1024*1024)
        else:
            model_trt = torch2trt(model, [dummy_input],
                                max_workspace_size=workspace_size*1024*1024*1024)
        
        # Save engine
        torch.save(model_trt.state_dict(), output_path)
        print(f"TensorRT engine saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False


def create_evaluation_dataset(images_dir: str, annotations_dir: str, output_file: str,
                            format_type: str = 'yolo'):
    """
    Create evaluation dataset from images and annotations
    
    Args:
        images_dir: Directory containing images
        annotations_dir: Directory containing annotations
        output_file: Output annotation file
        format_type: Annotation format ('yolo', 'coco')
    """
    import glob
    
    annotations = {}
    
    if format_type == 'yolo':
        # Process YOLO format annotations
        annotation_files = glob.glob(os.path.join(annotations_dir, '*.txt'))
        
        for ann_file in annotation_files:
            image_name = os.path.basename(ann_file).replace('.txt', '')
            
            with open(ann_file, 'r') as f:
                lines = f.readlines()
            
            annotations[image_name] = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert to absolute coordinates (assuming image size is known)
                    # This is a simplified conversion - you may need to adjust based on actual image sizes
                    annotations[image_name].append({
                        'class_id': class_id,
                        'bbox': [x_center, y_center, width, height],  # Normalized coordinates
                        'confidence': 1.0
                    })
    
    elif format_type == 'coco':
        # Process COCO format - this would require more complex handling
        # Implementation depends on specific COCO annotation structure
        pass
    
    # Save consolidated annotations
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Evaluation dataset created: {output_file}")
    print(f"Total images: {len(annotations)}")


def analyze_detection_results(results_dir: str, ground_truth_file: str = None):
    """
    Analyze detection results and generate comprehensive report
    
    Args:
        results_dir: Directory containing detection results
        ground_truth_file: Optional ground truth file for comparison
    """
    results_path = Path(results_dir)
    
    print("="*60)
    print("DETECTION RESULTS ANALYSIS")
    print("="*60)
    
    # Analyze saved results
    label_files = list(results_path.glob('labels/*.txt'))
    image_files = list(results_path.glob('*.jpg')) + list(results_path.glob('*.png'))
    
    print(f"Total images processed: {len(image_files)}")
    print(f"Total label files: {len(label_files)}")
    
    # Statistics
    total_detections = 0
    class_counts = {}
    confidence_scores = []
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 6:  # class, x, y, w, h, conf
                class_id = int(parts[0])
                confidence = float(parts[5])
                
                total_detections += 1
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                confidence_scores.append(confidence)
    
    print(f"Total detections: {total_detections}")
    print(f"Average confidence: {np.mean(confidence_scores):.3f}")
    print(f"Confidence std: {np.std(confidence_scores):.3f}")
    
    print("\nClass distribution:")
    for class_id, count in sorted(class_counts.items()):
        print(f"  Class {class_id}: {count} detections ({count/total_detections*100:.1f}%)")
    
    # Generate visualization if matplotlib is available
    try:
        plt.figure(figsize=(15, 5))
        
        # Confidence distribution
        plt.subplot(1, 3, 1)
        plt.hist(confidence_scores, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Score Distribution')
        plt.grid(True, alpha=0.3)
        
        # Class distribution
        plt.subplot(1, 3, 2)
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        plt.bar(classes, counts, alpha=0.7, edgecolor='black')
        plt.xlabel('Class ID')
        plt.ylabel('Detection Count')
        plt.title('Class Distribution')
        plt.grid(True, alpha=0.3)
        
        # Detection per image
        detections_per_image = []
        for label_file in label_files:
            with open(label_file, 'r') as f:
                detections_per_image.append(len(f.readlines()))
        
        plt.subplot(1, 3, 3)
        plt.hist(detections_per_image, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Detections per Image')
        plt.ylabel('Frequency')
        plt.title('Detections per Image Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_path / 'analysis_summary.png', dpi=300, bbox_inches='tight')
        print(f"\nAnalysis visualization saved to: {results_path / 'analysis_summary.png'}")
        
    except ImportError:
        print("Matplotlib not available for visualization")
    
    print("="*60)
