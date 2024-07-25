from utility import read_video, save_video
from trackers import Tracker

def main():
    print('Main python file.py')

    #read video
    video_frames = read_video('input_videos/train/A1606b0e6_0/A1606b0e6_0 (20).mp4')

    #initialise tracker 
    tracker = Tracker("models/model1/best.pt")
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    

    #saved cropped images of a player
    

    #draw output video    
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    #save video
    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()