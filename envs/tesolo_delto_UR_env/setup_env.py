import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from delto_env import DeltoEnv, DeltoEnvCfg 
import time
import os
import numpy as np

##
# Pre-defined configs
##

import socket  
# IP-адрес и порт сервера  


if __name__ == "__main__":
    
    server_ip = '127.0.0.1'  
    server_port = 8081  

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  
    recv_buf_size = 1024
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buf_size)
    server_socket.bind((server_ip, server_port)) 
    server_socket.settimeout(0.001)
    server_socket.setblocking(False)

    # Создаем среду
    env = DeltoEnv(DeltoEnvCfg())

    # Сброс среды
    obs = env.reset()
    print("Initial observation:", obs)

    start = torch.zeros([env.cfg.num_env, env.cfg.action_space])

    obs, rewards, dones, info, _ = env.step(start)

    # pos = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0,
    #       0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
    #       -1.0, 0.0, -3.0, -3.0]])
    
    pos = torch.zeros([env.cfg.num_env, env.cfg.action_space])*0.3

    while(True):
        
        try:
            data, address = server_socket.recvfrom(1024)

            data = list(map(float, (str(data)[3:-2].split(", "))))

            data[0] += 30*np.pi/180
            data[5] -= 50*np.pi/180

            print(data)
            pos = torch.tensor([data])
            
        except BlockingIOError:
            # данных пока нет
            pass

        obs, rewards, dones, info, _ = env.step(pos)

        print("Obs: ", obs["state"]["object_pos"])

        # print("Joints_pose:", obs["state"]["joints_pos"])
        print("====================================")

        time.sleep(0.05)