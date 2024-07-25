from .bbox_utils import get_center_bbox, get_measured_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_bbox(ball_bbox)

        min_dist = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']
            
            distance_left = get_measured_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = get_measured_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < min_dist:
                    min_dist = distance
                    assigned_player = player_id
        
        return assigned_player