import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv
import pickle
import os

import sys
sys.path.append('../')
from utility import get_bbox_width, get_center_bbox
#load a object detection model + video to start processing each frame 
#load a tracking model too 

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    
    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        #Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1 : {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions


    def detect_frames(self, frames):
        print('detect_frames');
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1, device='cuda') #ensure cuda is used to accelerate process
            detections+=detections_batch
            #break #remove break if running full video

        return detections


    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub is True and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks


        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections): 
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            print(cls_names)

            #convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Convert goalkeeper to normal player
            for object_idx, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_idx] = cls_names_inv["player"]

            #Track objects with tracker id
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}   


            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}


        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks


    def draw_circle(self, frame, bbox, color, track_id = None):
        y2 = int(bbox[3])
        x_center, _ = get_center_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45.0,
            endAngle=235.0,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4,
            shift=0
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED
                          )
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
    

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ])

        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            color,
            cv2.FILLED
        )

        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            (0,0,0),
            2
        )

        return frame
    

    
    def draw_team_ball_possession(self, frame, frame_num, team_ball_possession):
        #display ball possession
        overlay = frame.copy()
        x_start, y_start = frame.shape[1] - 550, 20
        x_end, y_end = frame.shape[1] - 20, 140
        cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (255,255,255), -1)
        alpha = 1
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_possession_till_frame = team_ball_possession[:frame_num+1]
        #filter out the teams
        team_A_frames = team_ball_possession_till_frame[team_ball_possession_till_frame == 1].shape[0]
        team_B_frames = team_ball_possession_till_frame[team_ball_possession_till_frame == 2].shape[0]
        team_A = team_A_frames/(team_A_frames+team_B_frames)
        team_B= team_B_frames/(team_A_frames+team_B_frames)

        cv2.putText(frame, f"Team A Possession: {team_A*100:.2f}%", (x_start + 50, y_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team B Possession: {team_B*100:.2f}%", (x_start + 50, y_start + 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame
        
    def draw_annotations(self, video_frames, tracks, team_ball_possession):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            #draw players 
            for track_id, player in player_dict.items():
                colour = player.get("team_colour", (0,0,255))
                frame = self.draw_circle(frame, player["bbox"], colour, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0,0,255))

            #draw referee
            for _, referee in referee_dict.items():
                frame = self.draw_circle(frame, referee["bbox"], (0,255,255))

            #draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (255,255,255))

 
            #display statistics eg. ball possession

            frame = self.draw_team_ball_possession(frame, frame_num, team_ball_possession)

            output_video_frames.append(frame)

        return output_video_frames