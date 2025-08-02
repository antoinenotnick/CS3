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

def create_background_mask(frame, human_detections, padding=20):
    """Create a mask that excludes human areas from analysis"""
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    
    for detection in human_detections:
        x1, y1, x2, y2 = detection
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(frame.shape[1], int(x2) + padding)
        y2 = min(frame.shape[0], int(y2) + padding)
        
        mask[y1:y2, x1:x2] = 0
    
    return mask

def apply_background_mask(frame, mask):
    """Apply mask to frame, setting human areas to black"""
    masked_frame = frame.copy()
    masked_frame[mask == 0] = [0, 0, 0]
    return masked_frame

def save_temp_frame(frame, temp_path="temp_frame.jpg"):
    """Save frame temporarily for analysis"""
    cv2.imwrite(temp_path, frame)
    return temp_path

def process(image_source, human_detections=None):
    weights_path_RF = r"C:\Users\rishith\VSC\Tier1YOLOWeights.pt"
    weights_path_YOLO = r"C:\Users\rishith\VSC\Tier2Weights.pt"
    confidence_threshold_RF = 0.42
    confidence_threshold_YOLO = 0.5
    save_annotated = True
    output_path = None

    modelRF = YOLO(weights_path_RF)
    YOLOmodel = YOLO(weights_path_YOLO)
    
    if isinstance(image_source, str):
        original_image = Image.open(image_source)
        image_path = image_source
    else:
        temp_path = save_temp_frame(image_source)
        original_image = Image.open(temp_path)
        image_path = temp_path
    
    annotated_image = original_image.copy()
    draw = ImageDraw.Draw(annotated_image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    colors = [
        '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
        '#00FFFF', '#FFA500', '#800080', '#FFC0CB', '#A52A2A'
    ]

    bounding_boxes = []

    results_stage1 = modelRF(image_path, conf=confidence_threshold_RF)
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

    results_stage2 = YOLOmodel(image_path, conf=confidence_threshold_YOLO)
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
        if output_path is None:
            if isinstance(image_source, str):
                base_name = os.path.splitext(os.path.basename(image_source))[0]
                output_dir = os.path.dirname(image_source)
            else:
                base_name = "live_frame"
                output_dir = "."
            output_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
        annotated_image.save(output_path)
        print(f"Annotated image saved to: {output_path}")

    print(f"\nDetection Summary:")
    print(f"Image size: {original_image.size[0]} x {original_image.size[1]}")
    print(f"Total detections: {len(bounding_boxes)}")

    if len(bounding_boxes) > 0:
        print("\nDetected objects:")
        for bbox in bounding_boxes:
            print(f"  • {bbox['class_name']} (confidence: {bbox['confidence']:.2f})")
            print(f"    Location: ({bbox['bbox']['x1']:.0f}, {bbox['bbox']['y1']:.0f}) to ({bbox['bbox']['x2']:.0f}, {bbox['bbox']['y2']:.0f})")
            print(f"    Size: {bbox['bbox']['width']:.0f} x {bbox['bbox']['height']:.0f}")
            print(f"    Center: ({bbox['bbox']['center_x']:.0f}, {bbox['bbox']['center_y']:.0f})")
    else:
        print("No objects detected above confidence threshold")
    
    print(f"\nDetection Summary:")
    print(f"Image size: {original_image.size}")
    print(f"Total unique detections: {len(bounding_boxes)}")

    for bbox in bounding_boxes:
        print(f" • [{bbox['model']}] {bbox['class_name']} ({bbox['confidence']:.2f}) at center ({bbox['bbox']['center_x']:.0f}, {bbox['bbox']['center_y']:.0f})")

    if isinstance(image_source, np.ndarray) and os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")

model = RFDETRNano()
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    detections = model.predict(frame[:, :, ::1], threshold=0.5)
    
    if detections.class_id.any() == 1:
        human_boxes = []
        for i, class_id in enumerate(detections.class_id):
            if class_id == 1:  # Person class in COCO is 0
                box = detections.xyxy[i]
                human_boxes.append(box)
        
        if human_boxes:
            background_mask = create_background_mask(frame, human_boxes)
            masked_frame = apply_background_mask(frame, background_mask)
            
            print("Human detected! Analyzing background only...")
            process(masked_frame, human_boxes)

    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

    cv2.imshow("Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()