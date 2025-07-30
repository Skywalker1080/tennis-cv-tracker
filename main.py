from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2

def main():
    input_video_path = "videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect Player
    player_tracker = PlayerTracker(model_path="yolov8x")
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl")

    # Detect Ball
    ball_tracker = BallTracker(model_path='models/best.pt')
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball(ball_detections)

    # Court Line Detector
    court_line_detector = CourtLineDetector(model_path='models/model_tennis_court_det.pt')
    court_keypoints = court_line_detector.predict(video_frames[0])
    
    
    overlay_output = player_tracker.draw_bboxes(video_frames, player_detections)
    overlay_output = ball_tracker.draw_bboxes(overlay_output,  ball_detections)
    overlay_output = court_line_detector.draw_keypoints_video(overlay_output, court_keypoints)

    # Draw video fps
    for i, frame in enumerate(overlay_output):
        cv2.putText(frame, f"Frame {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


    save_video(overlay_output, "output_video/video.avi")

if __name__ == "__main__":
    main()