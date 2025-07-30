from ultralytics import YOLO

model = YOLO('yolov8x')

model.track('videos/input_video.mp4', save=True)