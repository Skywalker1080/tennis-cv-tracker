from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_player(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_player(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player }
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_player(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # Sort distance wise
        distances.sort(key = lambda x: x[1])

        # Choose the first 2 
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players



    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections


        for frame in frames:
            player_dict = self.detect_player(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_player(self, frame):
        results = self.model.track(frame, persist=True) # remember tracks across all frames
        id_name_dict = self.model.names

        player_dict = {}
        if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
            for box in results[0].boxes:
                track_id = int(box.id.tolist()[0])
                result = box.xyxy.tolist()[0]
                object_id = int(box.cls.tolist()[0])
                object_cls_name = id_name_dict[object_id]
                if object_cls_name == 'person':
                    player_dict[track_id] = result

        return player_dict
        
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Id: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            output_video_frames.append(frame)
        return output_video_frames