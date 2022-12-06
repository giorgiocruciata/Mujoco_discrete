from torch import device
from mujoco_py.generated import const
from mujoco_py import GlfwContext
import numpy as np
import cv2
GlfwContext(offscreen=True)
import torch
def generate_action_mean_head(dist,n_actions):
    cp_act=[]
    cp_pol = []
    # actions_head = {head : {name : (out_policy_head[head][i].cumsum(-1) >= rand(out_policy_head[head][i].shape[:-1])[..., None]).byte().argmax(-1) for i, name in enumerate(self.CP_names)}  for head in out_policy_head.keys()}
    # temp = {head : {name : out_policy_head[head][i] for i, name in enumerate(self.CP_names)}  for head in out_policy_head.keys()}
    for i in range(n_actions):
        vec = 0
        for f, head in enumerate(dist.keys()):
            vec = vec +  dist[head][i]
        vecmean = vec/(f+1)
        cp_act.append(np.argmax(vecmean.detach().numpy(),axis=-1)[0])
        cp_pol.append(vecmean)
    actions = cp_act
    out_policy = cp_pol
    return actions,out_policy

class Play:
    def __init__(self, env, agent, env_name, max_episode=10):
        self.env = env
        self.max_episode = max_episode
        self.agent = agent
        _, self.state_rms_mean, self.state_rms_var = self.agent.load_weights()
        self.agent.set_to_eval_mode()
        self.device = device("cpu")
        # self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # self.VideoWriter = cv2.VideoWriter(env_name + ".avi", self.fourcc, 50.0, (250, 250))

    def evaluate(self):

        for _ in range(self.max_episode):
            s = self.env.reset()
            episode_reward = 0
            for _ in range(self.env._max_episode_steps):
                s = np.clip((s - self.state_rms_mean) / (self.state_rms_var ** 0.5 + 1e-8), -5.0, 5.0)
                dist = self.agent.choose_dist(s)
                with torch.no_grad():
                    action_avg,dist_avg = generate_action_mean_head(dist,self.agent.n_actions)
                CP_vectors = np.array([np.array(self.agent.binspace)[action_avg[i]].tolist() for i in range(self.agent.n_actions)])
                
                # action = dist.sample().cpu().numpy()[0]
                s_, r, done, _ = self.env.step(CP_vectors)
                self.env.render()
                episode_reward += r
                if done:
                    break
                s = s_
                # self.env.render(mode="human")
                # self.env.viewer.cam.type = const.CAMERA_FIXED
                # self.env.viewer.cam.fixedcamid = 0
                # time.sleep(0.03)
                # I = self.env.render(mode='rgb_array')
                # I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                # I = cv2.resize(I, (250, 250))
                # self.VideoWriter.write(I)
                # cv2.imshow("env", I)
                # cv2.waitKey(10)
            print(f"episode reward:{episode_reward:3.3f}")
        self.env.close()
        # self.VideoWriter.release()
        # cv2.destroyAllWindows()

