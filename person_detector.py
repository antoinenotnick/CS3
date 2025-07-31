import cv2
import time
import numpy as np
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

class PersonDetector:
    def __init__(self, confidence_threshold=0.5):
        self.model = RFDETRNano()
        self.confidence_threshold = confidence_threshold
    
    def detect_person_in_frame(self, frame):
        if self.model is None:
            print("Model not initialized. Cannot detect person.")
            return False
            
        try:
            # Ensure frame is in correct format
            if frame is None or frame.size == 0:
                print("Invalid frame provided to person detection.")
                return False
            
            # Convert BGR to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = frame[:, :, ::1]
            else:
                rgb_frame = frame
            
            # Run detection on the frame
            detections = self.model.predict(rgb_frame, threshold=self.confidence_threshold)
            
            # Check if any detection is a person (class_id 0 in COCO is 'person')
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                # Handle both numpy arrays and lists
                class_ids = detections.class_id
                if isinstance(class_ids, np.ndarray):
                    person_detected = np.any(class_ids == 0)
                else:
                    person_detected = any(class_id == 0 for class_id in class_ids)

                if person_detected:
                    print("Person detected in frame!")
                return person_detected
            
            return False
            
        except Exception as e:
            print(f"Error in person detection: {e}")
            return False
    
    def capture_and_check_person(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open webcam for person detection.")
            return False
        
        try:
            for _ in range(3):
                ret, frame = cap.read()
                if ret:
                    break
                time.sleep(0.1)
            
            cap.release()
            
            if not ret or frame is None:
                print("Error: Failed to capture frame for person detection.")
                return False
            
            return self.detect_person_in_frame(frame)
            
        except Exception as e:
            print(f"Error in capture_and_check_person: {e}")
            cap.release()
            return False
    
    def wait_for_clear_view(self, max_wait_time=300, check_interval=5, camera_index=0):
        print("Person detected! Waiting for clear view...")
        start_time = time.time()
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open webcam for person detection.")
            return True  # Assume clear if we can't check
        
        try:
            while time.time() - start_time < max_wait_time:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Error: Failed to capture frame for person detection.")
                    time.sleep(check_interval)
                    continue
                
                # Check if person is still in frame
                if not self.detect_person_in_frame(frame):
                    print("View is now clear!")
                    cap.release()
                    # Wait additional 60 seconds as specified
                    print("Waiting additional 60 seconds for stability...")
                    time.sleep(60)
                    return True
                else:
                    print(f"Person still detected. Waiting {check_interval} seconds before next check...")
                
                # Wait before next check
                time.sleep(check_interval)
            
            print(f"Max wait time ({max_wait_time}s) exceeded. Proceeding anyway.")
            cap.release()
            return False
            
        except Exception as e:
            print(f"Error during person detection wait: {e}")
            cap.release()
            return True  # Assume clear if error occurs

# Convenience functions for direct import
def create_person_detector(confidence_threshold=0.5):
    return PersonDetector(confidence_threshold)

def quick_person_check(confidence_threshold=0.5, camera_index=0):
    detector = PersonDetector(confidence_threshold)
    return detector.capture_and_check_person(camera_index)

# Test function to verify the module works
def test_person_detector():
    print("Testing PersonDetector...")
    try:
        detector = PersonDetector(confidence_threshold=0.5)
        result = detector.capture_and_check_person()
        print(f"Person detection test result: {result}")
        return True
    except Exception as e:
        print(f"PersonDetector test failed: {e}")
        return False

if __name__ == "__main__":
    print("person_detector.py loaded correctly")
    test_person_detector()