from ast import arg
import gym
import os
os.add_dll_directory("C:\\Users\\gWX1180901\\.mujoco\\mujoco200\\bin")
import mujoco_py
from agent import Agent
from train import Train
from play import Play
import numpy as np
import argparse
import torch
import random
parser = argparse.ArgumentParser()
parser.add_argument("--h", default="10", help="number of heads")
parser.add_argument("--t", default=1)
parser.add_argument("--e", default=0)
parser.add_argument("--a", default="7")
parser.add_argument("--s", default=1, help="1==average -- 0==bootstrap")
args = parser.parse_args()
heads=int(args.h)
TRAIN_FLAG = bool(args.t)
entropy_reg =  bool(args.e)
system_merge =  bool(args.s)
bins=int(args.a)-1
ENV_NAME = "Hopper"
test_env = gym.make(ENV_NAME + "-v2")

n_states = test_env.observation_space.shape[0]
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
n_actions = test_env.action_space.shape[0]
upper_bound =action_bounds[1]
botton_bound = action_bounds[0]
binspace = [-round(upper_bound-(upper_bound-botton_bound)/bins*x,2) for x in range(bins)]+[-botton_bound]
# np.random.seed(1240)
# np.random.shuffle(binspace)

bins = len(binspace)
n_iterations = 200
lr = 3e-4
epochs = 10
clip_range = 0.2
mini_batch_size = 64
T = 2048
ENV_NAMEDIR = ENV_NAME+"_"+str(heads)
if entropy_reg==True:
    ENV_NAMEDIR="entropy"+"_"+ENV_NAMEDIR
if system_merge==True:
    ENV_NAMEDIR="avg"+"_"+ENV_NAMEDIR
else:
    ENV_NAMEDIR="bootstrap"+"_"+ENV_NAMEDIR

ls_seed =  [123, 5406, 6692, 1743, 5750, 7258, 1655, 4060, 7864, 6845,]
if __name__ == "__main__":
    for seed in range(5):
        torch.manual_seed(ls_seed[seed])
        os.environ['PYTHONHASHSEED']=str(ls_seed[seed])
        np.random.seed(ls_seed[seed])
        random.seed(ls_seed[seed])
        ENV_NAMEDIR_SEED = str(ls_seed[seed])+"_"+ENV_NAMEDIR
        if TRAIN_FLAG:
            lsdir = [x for x in os.listdir("C:\\Users\gWX1180901\Desktop\\mujoco\\") if ENV_NAMEDIR_SEED == x[:-2] and "pth" not in x]
            if len(lsdir)==0:
                ENV_NAMEDIR_SEED = ENV_NAMEDIR_SEED+"_1"
            else:
                numord = max([int(x.split("_")[-1]) for x in lsdir])
                ENV_NAMEDIR_SEED = ENV_NAMEDIR_SEED+"_"+str(numord+1)
        else:
            ENV_NAMEDIR_SEED = ENV_NAMEDIR_SEED+"_1"

        if not os.path.exists(ENV_NAMEDIR_SEED):
            os.mkdir(ENV_NAMEDIR_SEED)
            os.mkdir(ENV_NAMEDIR_SEED + "/logs")

        print("name dir: ", ENV_NAMEDIR_SEED)
        print("training: ", TRAIN_FLAG )
        print(f"number of states: {n_states}")
        print(f"number of actions: {n_actions}")
        print(f"action bounds: {action_bounds}")
        print("binspace: ",binspace)
        print("number of heads: ", heads)
        print("entropy reg: ",entropy_reg)
        print("system_merge: ","average" if system_merge else "bootstrap")
        print("seed: ",ls_seed[seed])


        env = gym.make(ENV_NAME + "-v2")

        agent = Agent(n_states=n_states, n_iter=n_iterations,env_name=ENV_NAMEDIR_SEED, 
                    action_bounds=action_bounds,n_actions=n_actions,bins= bins,binspace=binspace,lr=lr,heads=heads,entropy_reg=entropy_reg,system_merge=system_merge)
        if TRAIN_FLAG:
            trainer = Train(env=env,test_env=test_env,env_name=ENV_NAMEDIR_SEED,agent=agent, horizon=T, 
                    n_iterations=n_iterations,epochs=epochs,mini_batch_size=mini_batch_size, epsilon=clip_range)
            trainer.step()

        player = Play(env, agent, ENV_NAMEDIR_SEED)
        player.evaluate()




# if entropy_reg==True:
#     ENV_NAMEDIR = "entropy_" + ENV_NAMEDIR

# if diversity_reg==True:
#     ENV_NAMEDIR = "diversity_" + ENV_NAMEDIR