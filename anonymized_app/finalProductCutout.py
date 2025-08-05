from ultralytics import YOLO
from ultralytics import RTDETR
import torch
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import supervision as sv
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES
import time
from datetime import datetime

torch.serialization.add_safe_globals([argparse.Namespace])

def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (box1_area + box2_area - inter_area)

def draw_box(draw, box, label, color, font):
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    bbox = draw.textbbox((0, 0), label, font=font)
    label_width = bbox[2] - bbox[0]
    label_height = bbox[3] - bbox[1]
    label_y = y1 - label_height - 5 if y1 - label_height - 5 > 0 else y1 + 5

    draw.rectangle(
        [x1, label_y, x1 + label_width + 10, label_y + label_height + 5],
        fill=color
    )
    draw.text((x1 + 5, label_y + 2), label, fill='white', font=font)

def blackout_human_regions(frame, human_detections):
    """
    Apply black boxes over human regions instead of blurring
    """
    blackout_frame = frame.copy()
   
    for detection in human_detections:
        x1, y1, x2, y2 = map(int, detection)
       
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
       
        cv2.rectangle(blackout_frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
   
    return blackout_frame

def process_damage_analysis(frame, entry_time, exit_time):
    weights_path_Tier1 = r"C:\Users\rishith\VSC\Tier1YOLOWeights.pt"
    weights_path_Tier2 = r"C:\Users\rishith\VSC\Tier2Weights.pt"
    confidence_threshold_Tier1 = 0.42
    confidence_threshold_Tier2 = 0.5
    save_annotated = True

    print(f"\n{'='*60}")
    print(f"DAMAGE ANALYSIS REPORT")
    print(f"{'='*60}")
    print(f"Human entered scene: {entry_time}")
    print(f"Human left scene: {exit_time}")
    print(f"Scene duration: {(exit_time.replace(microsecond=0) - entry_time.replace(microsecond=0))}")
    print(f"{'='*60}")

    modelRF = YOLO(weights_path_Tier1)
    YOLOmodel = YOLO(weights_path_Tier2)
   
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_image = Image.fromarray(frame_rgb)
    annotated_image = original_image.copy()
    draw = ImageDraw.Draw(annotated_image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    bounding_boxes = []

    results_stage1 = modelRF(frame, conf=confidence_threshold_Tier1)
    result1 = results_stage1[0]
    stage1_boxes = []

    if result1.boxes is not None:
        boxes = result1.boxes.xyxy.cpu().numpy()
        confidences = result1.boxes.conf.cpu().numpy()
        class_ids = result1.boxes.cls.cpu().numpy()

        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box
            class_name = YOLOmodel.names[int(cls_id)]
            color = '#00FF00'

            draw_box(draw, [x1, y1, x2, y2], f"{class_name} {conf:.2f}", color, font)

            bbox_data = {
                'box_id': f"S1-{i}",
                'model': 'YOLOv11-Stage1',
                'class_name': class_name,
                'class_id': int(cls_id),
                'confidence': float(conf),
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1),
                    'center_x': float((x1 + x2) / 2),
                    'center_y': float((y1 + y2) / 2)
                }
            }
            bounding_boxes.append(bbox_data)
            stage1_boxes.append([x1, y1, x2, y2])

    results_stage2 = YOLOmodel(frame, conf=confidence_threshold_Tier2)
    result2 = results_stage2[0]

    if result2.boxes is not None:
        boxes = result2.boxes.xyxy.cpu().numpy()
        confidences = result2.boxes.conf.cpu().numpy()
        class_ids = result2.boxes.cls.cpu().numpy()

        for j, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box
            class_name = YOLOmodel.names[int(cls_id)]
            color = '#0000FF'

            skip = False
            for existing_box in stage1_boxes:
                iou = calculate_iou(box, existing_box)
                if iou > 0.5:
                    skip = True
                    break
            if skip:
                continue

            draw_box(draw, [x1, y1, x2, y2], f"{class_name} {conf:.2f}", color, font)

            bbox_data = {
                'box_id': f"S2-{j}",
                'model': 'YOLOv11-Stage2',
                'class_name': class_name,
                'class_id': int(cls_id),
                'confidence': float(conf),
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1),
                    'center_x': float((x1 + x2) / 2),
                    'center_y': float((y1 + y2) / 2)
                }
            }
            bounding_boxes.append(bbox_data)

    if save_annotated:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"damage_report_{timestamp}.jpg"
        annotated_image.save(output_path)
        print(f"Damage analysis image saved to: {output_path}")

    print(f"Frame size: {original_image.size}")
    print(f"Total damage detections: {len(bounding_boxes)}")
    
    if len(bounding_boxes) > 0:
        print(f"\nDamage detected:")
        damage_summary = {}
        for bbox in bounding_boxes:
            damage_type = bbox['class_name']
            if damage_type not in damage_summary:
                damage_summary[damage_type] = 0
            damage_summary[damage_type] += 1
            print(f" • [{bbox['model']}] {bbox['class_name']} (confidence: {bbox['confidence']:.2f}) at center ({bbox['bbox']['center_x']:.0f}, {bbox['bbox']['center_y']:.0f})")
        
        print(f"\nDamage Summary:")
        for damage_type, count in damage_summary.items():
            print(f" • {damage_type}: {count} instance(s)")
    else:
        print("No damage detected in scene")
    
    print(f"{'='*60}\n")

class HumanTracker:
    def __init__(self):
        self.human_present = False
        self.entry_time = None
        self.last_frame = None
        self.no_human_frames = 0
        self.confirmation_frames = 5  

    def update(self, frame, human_detected):
        current_time = datetime.now()
        
        if human_detected and not self.human_present:
            
            self.human_present = True
            self.entry_time = current_time
            self.no_human_frames = 0
            print(f"Human entered scene at {current_time.strftime('%H:%M:%S')}")
            
        elif human_detected and self.human_present:
            self.no_human_frames = 0
            
        elif not human_detected and self.human_present:
            self.no_human_frames += 1
            
            if self.no_human_frames >= self.confirmation_frames:
                exit_time = current_time
                print(f"Human left scene at {exit_time.strftime('%H:%M:%S')}")
                
                if self.last_frame is not None:
                    process_damage_analysis(self.last_frame, self.entry_time, exit_time)
                
                self.human_present = False
                self.entry_time = None
                self.no_human_frames = 0
        
        if self.human_present or self.no_human_frames > 0:
            self.last_frame = frame.copy()

model = RFDETRNano()
cap = cv2.VideoCapture(0)
tracker = HumanTracker()

print("Starting damage detection system...")
print("System will analyze for damage after humans leave the scene")
print("Press 'q' to quit\n")

while True:
    success, frame = cap.read()
    if not success:
        break
   
    detections = model.predict(frame[:, :, ::1], threshold=0.5)
   
    human_indices = np.where(detections.class_id == 1)[0]
    human_detected = len(human_indices) > 0
    
    tracker.update(frame, human_detected)

    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    if human_detected:
        human_boxes = detections.xyxy[human_indices]
        annotated_frame = blackout_human_regions(frame, human_boxes)
    else:
        annotated_frame = frame.copy()
    
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

    status_text = "Human Present - Analysis pending..." if tracker.human_present else "Monitoring for humans..."
    cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if not tracker.human_present else (0, 165, 255), 2)

    cv2.imshow("Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()