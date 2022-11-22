import logging

import numpy as np
import torch
from scipy.spatial.distance import euclidean


def get_relative_pos(waypoint, ego_pos):
    return [waypoint.pos[0] - ego_pos[0], waypoint.pos[1] - ego_pos[1]]


def rotate2d_vector(vectors, angle):
    ae_cos = np.cos(angle)
    ae_sin = np.sin(angle)
    rot_matrix = np.array([[ae_cos, -ae_sin], [ae_sin, ae_cos]])

    vectors_rotated = np.inner(vectors, rot_matrix)
    return vectors_rotated


def get_closest_point_index(pts_arr, pts):
    # print("pts_arr:{}  pts:{}".format(pts_arr, pts))
    distance = [euclidean(each, pts[0:2]) for each in pts_arr]
    return np.argmin(distance)


def get_closest_waypoint(ego_position, ego_heading, num_lookahead, goal_path):
    closest_wp = min(goal_path, key=lambda wp: wp.dist_to(ego_position))
    min_dist = float("inf")
    min_dist_idx = -1
    for i, wp in enumerate(goal_path):

        if wp.dist_to(ego_position) < min_dist:
            min_dist = wp.dist_to(ego_position)
            min_dist_idx = i
            closest_wp = wp

    waypoints_lookahead = [
        get_relative_pos(wp, ego_position)
        for wp in goal_path[
            min_dist_idx: min(min_dist_idx + num_lookahead, len(goal_path))
        ]
    ]
    if len(waypoints_lookahead) > 0:
        while len(waypoints_lookahead) < num_lookahead:
            waypoints_lookahead.append(waypoints_lookahead[-1])
    else:
        waypoints_lookahead = [
            get_relative_pos(closest_wp.pos, ego_position) for i in range(num_lookahead)
        ]

    waypoints_lookahead = rotate2d_vector(waypoints_lookahead, -ego_heading)
    return closest_wp, waypoints_lookahead


def get_path_to_goal(goal, paths, start):
    goal_pos = goal.position
    start_pos = start.position
    path_start_pts = [each[0].pos for each in paths]

    best_path_ind = get_closest_point_index(path_start_pts, start_pos)
    path = paths[best_path_ind]

    middle_point = path[int(len(path) / 2)]
    goal_lane_id = middle_point.lane_id
    goal_lane_index = middle_point.lane_index

    path_pts = [each.pos for each in path]
    end_path_ind = get_closest_point_index(path_pts, goal_pos)

    return path
