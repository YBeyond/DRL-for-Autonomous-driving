import logging
import math
import collections
import numpy as np


from smarts.core.sensors import Observation
from util import get_closest_waypoint, get_path_to_goal

_VEC_NORMALIZATION_VALUES = {
    "speed": 30,
    "target_speed": 30,
    "steering": 3.14  # In radians
}


class Normalization:
    def __init__(self, data) -> None:
        pass

    def update(self, data):
        return data / 255


class EnvPostProcessor():
    def __init__(self) -> None:
        self.img_width = 256
        self.img_height = 256
        # self.env_img_normalize = Normalization()
        # self.env_vec_normalize = Normalization()
        # self.vec_normalize = Normalization()
        self.deque_length = 5
        self.env_img_deque = collections.deque(maxlen=self.deque_length)
        self.env_vec_deque = collections.deque(maxlen=self.deque_length)
        self.ego_vec_deque = collections.deque(maxlen=self.deque_length)
        self.last_steering = 0
        self.last_accelerate = 0
        for _ in range(self.deque_length):
            self.env_img_deque.append(np.zeros(
                [self.img_width, self.img_height]).astype(np.float32))
    # TODO：归一化处理函数

    def AssembleEnvState(self, env_obs: Observation):

        # 获取连续四帧图像数据
        rgb_image = env_obs.top_down_rgb.data
        # print("rbg image shape :{}".format(rgb_image.shape))
        gray_image = np.dot(rgb_image, (0.1, 0.8, 0.1))
        gray_image = np.divide(gray_image, 255.0)
        self.env_img_deque.append(gray_image.astype(np.float32))

        # 获取自车向量信息，speed，speed_limit,speed_ratio,last_steering,last_accelerate
        ego_speed = env_obs.ego_vehicle_state.speed
        ego_steering = env_obs.ego_vehicle_state.steering

        ego_position = env_obs.ego_vehicle_state.position
        ego_heading = env_obs.ego_vehicle_state.heading
        ego_start = env_obs.ego_vehicle_state.mission.start
        ego_goal = env_obs.ego_vehicle_state.mission.goal
        ego_waypoits = env_obs.waypoint_paths

        ego_goal_path = get_path_to_goal(
            goal=ego_goal, paths=ego_waypoits, start=ego_start)
        ego_closest_waypoint, ego_lookahead_waypoits = get_closest_waypoint(
            num_lookahead=20, goal_path=ego_goal_path, ego_position=ego_position, ego_heading=ego_heading)
        speed_limit = ego_closest_waypoint.speed_limit
        # speed_limit = 30.0
        speed_ratio = max(0, ego_speed/speed_limit)

        # 归一化后的vec
        vec_state = {"target_speed": speed_limit / 30.0, "speed": ego_speed / 30.0, "steering": ego_steering,
                     "speed_ratio": speed_ratio, "last_steering": self.last_steering, "last_accelerate": self.last_accelerate}
        # 输入网络后聚合
        env_state = {"env_info": self.env_img_deque,
                     "vec_info": vec_state.values()}
        return env_state

    def RecordLastAction(self, action):
        self.last_steering = action[0]
        self.last_accelerate = action[1]

    def reset(self):
        self.env_img_deque = collections.deque(maxlen=self.deque_length)
        self.env_vec_deque = collections.deque(maxlen=self.deque_length)
        self.ego_vec_deque = collections.deque(maxlen=self.deque_length)
        for _ in range(self.deque_length):
            self.env_img_deque.append(
                np.zeros([self.img_width, self.img_height]).astype(np.float32))
