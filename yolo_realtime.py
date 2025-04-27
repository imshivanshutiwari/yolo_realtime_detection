import cv2
from ultralytics import YOLO
import time
import os
from collections import Counter

# Ask user for model choice
print("Choose YOLOv8 Model:")
print("1. YOLOv8n (Nano - Fastest)")
print("2. YOLOv8s (Small - Balanced)")
print("3. YOLOv8m (Medium - More accurate)")

choice = input("Enter your choice (1/2/3): ")

# Load model based on user choice
if choice == '1':
    model_path = 'yolov8n.pt'
elif choice == '2':
    model_path = 'yolov8s.pt'
elif choice == '3':
    model_path = 'yolov8m.pt'
else:
    print("Invalid choice. Defaulting to YOLOv8n.")
    model_path = 'yolov8n.pt'

print(f"Loading model: {model_path}")
model = YOLO(model_path)

# Create output folders if they don't exist
if not os.path.exists('output_videos'):
    os.makedirs('output_videos')
if not os.path.exists('results'):
    os.makedirs('results')

# Ask for input source
print("\nInput Source:")
print("1. Webcam")
print("2. Video File")

input_choice = input("Enter your choice (1/2): ")

if input_choice == '1':
    cap = cv2.VideoCapture(0)
    output_path = 'output_videos/webcam_output.avi'
elif input_choice == '2':
    video_path = input("Enter path to video file (example: input_videos/sample.mp4): ")
    cap = cv2.VideoCapture(video_path)
    output_path = 'output_videos/video_output.avi'
else:
    print("Invalid choice. Defaulting to Webcam.")
    cap = cv2.VideoCapture(0)
    output_path = 'output_videos/webcam_output.avi'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

# Create window and trackbar for confidence threshold
cv2.namedWindow('YOLOv8 Detection')
def nothing(x):
    pass
cv2.createTrackbar('Confidence', 'YOLOv8 Detection', 50, 100, nothing)  # default 50%

screenshot_count = 1  # To track number of screenshots

# Get class names from COCO dataset
class_names = model.names

while True:
    success, frame = cap.read()
    if not success:
        print("[INFO] End of video or no input.")
        break

    # Resize frame to 640x480
    frame = cv2.resize(frame, (640, 480))

    # Read confidence threshold from trackbar
    confidence = cv2.getTrackbarPos('Confidence', 'YOLOv8 Detection') / 100.0  # scale 0-1

    start = time.time()

    # Run detection with dynamic confidence
    results = model.predict(frame, imgsz=640, conf=confidence)

    # Visualize results
    annotated_frame = results[0].plot()

    end = time.time()
    fps = 1 / (end - start)

    # Object Counter
    detected_classes = []
    boxes = results[0].boxes

    if boxes is not None:
        for cls in boxes.cls:
            class_id = int(cls)
            class_label = class_names[class_id]
            detected_classes.append(class_label)

    counter = Counter(detected_classes)

    y_offset = 50  # starting y-coordinate to display counts
    for cls, count in counter.items():
        text = f"{cls}: {count}"
        cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)
        y_offset += 30

    # Put FPS text
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the frame to video
    out.write(annotated_frame)

    # Display frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save current frame
        screenshot_path = f'results/screenshot_{screenshot_count}.jpg'
        cv2.imwrite(screenshot_path, annotated_frame)
        print(f"[INFO] Screenshot saved: {screenshot_path}")
        screenshot_count += 1

cap.release()
out.release()
cv2.destroyAllWindows() 