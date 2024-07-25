import cv2
from utility import read_video, save_video, TeamAssigner, PlayerBallAssigner
from trackers import Tracker
import numpy

def main():
    print('Main python file.py')

    #read video
    video_frames = read_video('input_videos/train/A1606b0e6_0/A1606b0e6_0 (19).mp4')

    #initialise tracker 
    tracker = Tracker("models/model1/best.pt")
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    #interpolating ball position
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])
    
    #assign players to teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_colour(video_frames[0],
                                     tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_colour'] = team_assigner.team_colours[team]
    

    #detect ball possession
    player_assigner = PlayerBallAssigner()
    team_ball_possession = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_possession.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_possession.append(team_ball_possession[-1])
    team_ball_possession = numpy.array(team_ball_possession)



    #draw output video    
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_possession)

    #save video
    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()