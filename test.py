import numpy as np
import torch
from collections import Counter

def generate_action_mean_head(dist,n_actions):
    cp_act=[]
    for i in range(n_actions):
        vec = 0
        for f, head in enumerate(dist.keys()):
            vec = vec +  dist[head][i]
        vecmean = vec/(f+1)
        cp_act.append(np.argmax(vecmean.detach().numpy(),axis=-1)[0])
    actions = cp_act
    return actions



def generate_deterministic_action(dist,n_actions):
    heads = {head: [np.argmax(dist[head][i].detach().numpy(),axis=-1) for i in range(n_actions)] for head in dist.keys()}
    action = []
    for act in range(n_actions):
        vec= []
        for head in heads.keys():
            vec.append(heads[head][act])
        ct = mostCommon(vec)
        action.append(ct[0])
    return action

def mostCommon(lst):
    return [Counter(col).most_common(1)[0][0] for col in zip(*lst)]


def evaluate_model(agent, env, state_rms,system="avg"):
    total_rewards = 0
    n_actions = agent.n_actions
    s = env.reset()
    done = False
    while not done:
        s = np.clip((s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5.0, 5.0)
        dist = agent.choose_dist(s)
        if system=="avg":
            action = generate_action_mean_head(dist,n_actions)
        elif system=="majority":
            action = generate_deterministic_action(dist,n_actions)
        CP_vectors = np.array([np.array(agent.binspace)[action[i]].tolist() for i in range(agent.n_actions)])
        # action = dist.sample().cpu().numpy()[0]
        # action = np.clip(action, action_bounds[0], action_bounds[1])
        next_state, reward, done, _ = env.step(CP_vectors)
        # env.render()
        s = next_state
        total_rewards += reward
    # env.close()
    return total_rewards
