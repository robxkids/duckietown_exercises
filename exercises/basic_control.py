#!/usr/bin/env python3

"""
Simple exercise to construct a controller that controls the simulated Duckiebot using pose. 
"""

import time
import sys
import argparse
import math
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='straight_road')
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name = args.map_name,
        domain_rand = False,
        draw_bbox = False
    )
else:
    env = gym.make(args.env_name)

obs = env.reset()
env.render()

total_reward = 0

while True:
    duck = 0
    lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rads = lane_pose.angle_rad

    img = env.render('rgb_array')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bnw = cv2.inRange(img, (0, 150, 170), (0, 220, 255))
    contours, ret = cv2.findContours(bnw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = contours[0]
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        (x, y, w, h) = cv2.boundingRect(contours)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print(w, h)

        if h > 240:
            duck = 1
            print("stop")


    cv2.imshow('inRange', img)

    ###### Start changing the code here.
    # TODO: Decide how to calculate the speed and direction.

    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break

    k_p = 10
    k_d = 1
    
    # The speed is a value between [0, 1] (which corresponds to a real speed between 0m/s and 1.2m/s)
    if duck == 0:
        speed = 0.1 # TODO: You should overwrite this value
    else:
        speed = 0
    # angle of the steering wheel, which corresponds to the angular velocity in rad/s
    steering = k_p*distance_to_road_center + k_d*angle_from_straight_in_rads # TODO: You should overwrite this value

    ###### No need to edit code below.
    
    obs, reward, done, info = env.step([speed, steering])
    total_reward += reward
    
    print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))

    env.render()

    if done:
        if reward < 0:
            print('*** CRASHED ***')
        print ('Final Reward = %.3f' % total_reward)
        cv2.destroyAllWindows()
        break


