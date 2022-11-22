import gym
import numpy as np

from scipy.spatial import distance
from smarts.core.sensors import Observation
import math
from util import get_closest_waypoint, get_path_to_goal


def new_reward(observation: Observation, reward: float) -> float:
    """Adapts a raw environment observation and an environment reward to a custom reward
    of type float.

    The raw observation from the environment must include the ego vehicle's state,
    events, and waypoint paths. See smarts.core.sensors for more information on the
    Observation type.

    Args:
        observation (Observation): The raw environment observation received from SMARTS.
        reward (float): The environment reward received from SMARTS.

    Returns:
        float: The adapted, custom reward which includes aspects of the ego vehicle's
            state and the ego vehicle's mission progress, in addition to the environment
            reward.
    """
    env_reward = reward
    ego_events = observation.events
    ego_observation = observation.ego_vehicle_state
    start = observation.ego_vehicle_state.mission.start
    goal = observation.ego_vehicle_state.mission.goal
    path = get_path_to_goal(
        goal=goal, paths=observation.waypoint_paths, start=start)

    linear_jerk = np.linalg.norm(ego_observation.linear_jerk)
    angular_jerk = np.linalg.norm(ego_observation.angular_jerk)

    # Distance to goal
    ego_2d_position = ego_observation.position[0:2]
    goal_dist = distance.euclidean(ego_2d_position, goal.position[0:2])

    closest_wp, _ = get_closest_waypoint(
        num_lookahead=20,
        goal_path=path,
        ego_position=ego_observation.position,
        ego_heading=ego_observation.heading,
    )
    angle_error = closest_wp.relative_heading(
        ego_observation.heading
    )  # relative heading radians [-pi, pi]

    # Distance from center
    signed_dist_from_center = closest_wp.signed_lateral_error(
        observation.ego_vehicle_state.position
    )
    lane_width = closest_wp.lane_width * 0.5
    ego_dist_center = signed_dist_from_center / lane_width

    # NOTE: This requires the NeighborhoodVehicles interface.
    # # number of violations
    # (ego_num_violations, social_num_violations,) = ego_social_safety(
    #     observation,
    #     d_min_ego=1.0,
    #     t_c_ego=1.0,
    #     d_min_social=1.0,
    #     t_c_social=1.0,
    #     ignore_vehicle_behind=True,
    # )

    speed_fraction = max(0, ego_observation.speed / closest_wp.speed_limit)
    ego_step_reward = 0.02 * min(speed_fraction, 1) * np.cos(angle_error)
    ego_speed_reward = min(
        0, (closest_wp.speed_limit - ego_observation.speed) * 0.01
    )  # m/s
    ego_collision = len(ego_events.collisions) > 0
    ego_collision_reward = -1.0 if ego_collision else 0.0
    ego_off_road_reward = -1.0 if ego_events.off_road else 0.0
    ego_off_route_reward = -1.0 if ego_events.off_route else 0.0
    ego_wrong_way = -0.02 if ego_events.wrong_way else 0.0
    ego_goal_reward = 0.0
    ego_time_out = 0.0
    ego_dist_center_reward = -0.002 * min(1, abs(ego_dist_center))
    ego_angle_error_reward = -0.005 * max(0, np.cos(angle_error))
    ego_reached_goal = 1.0 if ego_events.reached_goal else 0.0
    # NOTE: This requires the NeighborhoodVehicles interface.
    # ego_safety_reward = -0.02 if ego_num_violations > 0 else 0
    # NOTE: This requires the NeighborhoodVehicles interface.
    # social_safety_reward = -0.02 if social_num_violations > 0 else 0
    ego_lat_speed = 0.0  # -0.1 * abs(long_lat_speed[1])
    ego_linear_jerk = -0.0001 * linear_jerk
    ego_angular_jerk = -0.0001 * angular_jerk * math.cos(angle_error)
    env_reward /= 100
    # DG: Different speed reward
    ego_speed_reward = -0.1 if speed_fraction >= 1 else 0.0
    ego_speed_reward += -0.01 if speed_fraction < 0.01 else 0.0

    rewards = sum(
        [
            ego_goal_reward,
            ego_collision_reward,
            ego_off_road_reward,
            ego_off_route_reward,
            ego_wrong_way,
            ego_speed_reward,
            # ego_time_out,
            ego_dist_center_reward,
            ego_angle_error_reward,
            ego_reached_goal,
            ego_step_reward,
            env_reward,
            # ego_linear_jerk,
            # ego_angular_jerk,
            # ego_lat_speed,
            # ego_safety_reward,
            # social_safety_reward,
        ]
    )
    # print("using default reward design in {}".format(__file__))
    return rewards
