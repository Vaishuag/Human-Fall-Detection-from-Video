from ultralytics import YOLO
import cv2
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# model = YOLO("./runs/detect/train/weights/best.pt")
model = YOLO("fall_det_1.pt")

video_path = "video_6.mp4";
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        
        results = model.track(frame, persist=True,conf=0.5)

        
        annotated_frame = results[0].plot()

        
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        
        break


cap.release()
cv2.destroyAllWindows()

