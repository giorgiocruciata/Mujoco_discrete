import torch
import numpy as np
import time
from running_mean_std import RunningMeanStd
from test import evaluate_model
from torch.utils.tensorboard import SummaryWriter
from torch import rand
import matplotlib.pyplot as plt
from torch import nn
from torch.distributions import normal

class Train:
    def __init__(self, env, test_env, env_name, n_iterations, agent, epochs, mini_batch_size, epsilon, horizon):
        self.env = env
        self.env_name = env_name
        self.test_env = test_env
        self.agent = agent
        self.epsilon = epsilon
        self.horizon = horizon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.n_iterations = n_iterations

        self.start_time = 0
        self.state_rms = RunningMeanStd(shape=(self.agent.n_states,))
        self.num_act = self.agent.n_actions
        self.running_reward = 0
        self.evalreward=[]
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.head_list = ["head"+str(x) for x in range(self.agent.heads)]
        self.shuffle_head= [x for x in range(self.agent.heads)]
        
        
    def calcdict(self,actions):
        dict_one_hot_tensor = []
        for i in range(self.num_act):
            dict_one_hot_tensor.append(torch.nn.functional.one_hot(actions[:,i], self.agent.bins))
        return dict_one_hot_tensor

    @staticmethod
    def choose_mini_batch(mini_batch_size, states, actions, returns, advs, values, log_probs,mask,h):
        index = np.where(np.array(mask)==h)[0]
        full_batch_size = len(index)
        for _ in range(full_batch_size // mini_batch_size):
            indices = np.random.choice(index, mini_batch_size)
            yield states[indices], actions[:,indices,:], returns[indices], advs[indices], values[indices],log_probs[:,indices]

    def generate_action_for_heads(self, dist):
        actions_head = [(dist[i].cumsum(-1) >= rand(dist[i].shape[:-1])[..., None]).byte().argmax(-1).item() for i in range(self.agent.n_actions)]
        dist = [dist[i].detach().squeeze().numpy() for i in range(self.agent.n_actions)]
        return actions_head,dist

    def generate_action_map(self, dist):
            actions_head = {head: [dist[head][i].argmax(-1).numpy() for i in range(self.agent.n_actions)] for head in dist.keys()}
            return actions_head

    def generate_dist_avg(self, dist):
        out_policy = []
        for i in range(self.num_act):
            vec = 0
            for f, head in enumerate(dist.keys()):
                vec = vec +  dist[head][i]
            vecmean = vec/(f+1)
            out_policy.append(vecmean.squeeze())
        return out_policy

    def generate_action_avg(self, dist):
        actions=[]
        out_policy = []
        for i in range(self.num_act):
            vec = 0
            for f, head in enumerate(dist.keys()):
                vec = vec +  dist[head][i]
            vecmean = vec/(f+1)
            actions.append((vecmean.cumsum(-1) >= rand(vecmean.shape[:-1])[..., None]).byte().argmax(-1).item())
            out_policy.append(vecmean.squeeze().numpy())
        return actions,out_policy


    def cos_sim(self,actions_head,CP_names):
        head_names = list(actions_head.keys())
        N_heads = len(head_names)
        cp_sim = 0
        for z in range(CP_names):
            similarity = 0
            for i in range(len(head_names)-1):
                for j in range(i+1,len(head_names),1):
                    head_name_i = head_names[i]
                    head_name_j = head_names[j]
                    similarity_i_j = self.cos(actions_head[head_name_i][z].float() , actions_head[head_name_j][z].float(),)
                    similarity += torch.mean(similarity_i_j)
            cp_sim += similarity
        return (2 / (N_heads * (N_heads - 1)))*cp_sim

    def generate_policy(self, dist):
        # dist = [dist[i] for i in range(self.agent.n_actions)]
        return dist

    def train(self, states, actions, advs, values,probs, mask):

        dict_one_hot_tensor = self.calcdict(actions)
        policy_probs_old = torch.stack([torch.sum(torch.mul(dict_one_hot_tensor[i],torch.tensor(probs[:,i,:])),dim=1)
        for i in range(self.num_act)])
        values = np.vstack(values[:-1])
        log_policy_prob_old_head = torch.log(policy_probs_old)
        returns = advs + values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        dict_one_hot_tensor = torch.stack(dict_one_hot_tensor)
        actor_loss_list = 0
        critic_loss_list = 0
        cos_sim_list = 0
        entropy_list = 0
        diversity_entropy_list=0
        cc = 0
        np.random.shuffle(self.shuffle_head)

        for h in self.shuffle_head if not self.agent.system_merge else [0]:
            index = np.where(np.array(mask)==h)[0]
            if len(index)>self.mini_batch_size:
                if not self.agent.system_merge:
                    self.agent.set_not_trainable()
                    self.agent.set_trainable_head(self.head_list[h])
                for epoch in range(self.epochs):
                    for state, dict_one_hot, return_, adv, old_value, old_log_prob in self.choose_mini_batch(self.mini_batch_size,
                                                                                                    states, dict_one_hot_tensor, returns,
                                                                                                    advs, values, log_policy_prob_old_head,mask,h):
                        state = torch.Tensor(state).to(self.agent.device)
                        dict_one_hot = torch.Tensor(dict_one_hot).to(self.agent.device)
                        return_ = torch.Tensor(return_).to(self.agent.device)
                        adv = torch.Tensor(adv).to(self.agent.device)
                        old_value = torch.Tensor(old_value).to(self.agent.device)
                        old_log_prob = torch.Tensor(old_log_prob).to(self.agent.device)
                        value = self.agent.critic(state)
                        critic_loss = self.agent.critic_loss(value, return_)
                        new_log_prob,policy_distribution = self.calculate_log_probs(self.agent.current_policy, state, dict_one_hot,self.head_list[h])
                        entropy = sum([self.entropy_loss(policy_distribution[head]) for head in policy_distribution.keys()])/len(policy_distribution.keys())
                        with torch.no_grad():    
                            if self.agent.heads>1:
                                actions_map = self.generate_action_map(policy_distribution)
                                dict_one_hot_entropy = {head : self.calcdict(torch.tensor(actions_map[head]).transpose(1,0)) for head in actions_map.keys()}
                                diversity_entropy = self.entropy_for_diversity(dict_one_hot_entropy)                                
                                cos_sim = self.cos_sim(policy_distribution,self.num_act)
                                cos_sim_list += cos_sim.item()
                                diversity_entropy_list+=diversity_entropy
                        
                        ratio = (new_log_prob - old_log_prob).exp().transpose(1,0)
                        actor_loss = self.compute_actor_loss(ratio, adv)
                        if self.agent.entropy_reg == True:
                            actor_loss += (- 0.01*entropy)
                        # if self.agent.diversity_reg == True:
                        #     actor_loss +=  (0.01*cos_sim)
                        self.agent.optimize(actor_loss, critic_loss)
                        actor_loss_list += actor_loss.item()
                        critic_loss_list += critic_loss.item()
                        entropy_list += entropy.item()
                        cc+=1

        return actor_loss_list/cc, critic_loss_list/cc,entropy_list/cc, cos_sim_list/cc, diversity_entropy_list/cc

    def step(self):
        state = self.env.reset()
        sampled_head = np.random.randint(0,self.agent.heads)
        for iteration in range(1, 1 + self.n_iterations):
            states = []
            actions = []
            rewards = []
            values = []
            probs = []
            dones = []
            mask = []
            self.start_time = time.time()
            for t in range(self.horizon):
                # self.env.render()
                # self.state_rms.update(state)
                state = np.clip((state - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
                dist = self.agent.choose_dist(state)
                if self.agent.system_merge:
                    action_avg,dist_avg = self.generate_action_avg(dist)
                else:
                    action_avg,dist_avg = self.generate_action_for_heads(dist[self.head_list[sampled_head]])
                value = self.agent.get_value(state)
                CP_vectors = np.array([np.array(self.agent.binspace)[action_avg[i]].tolist() for i in range(self.num_act)])#.transpose()
                next_state, reward, done, _ = self.env.step(CP_vectors)
                states.append(state)
                actions.append(action_avg)
                rewards.append(reward)
                values.append(value)
                probs.append(dist_avg)
                dones.append(done)
                mask.append(sampled_head if not self.agent.system_merge else 0)
                if done:
                    state = self.env.reset()
                    sampled_head = np.random.randint(0,self.agent.heads)
                else:
                    state = next_state
            # self.state_rms.update(next_state)
            next_state = np.clip((next_state - self.state_rms.mean) / (self.state_rms.var ** 0.5 + 1e-8), -5, 5)
            next_value = self.agent.get_value(next_state) * (1 - done)
            values.append(next_value)
            advs = self.get_gae(rewards, values, dones)
            states = np.vstack(states)
            actions = torch.tensor(actions)
            probs =  np.array(probs)
            actor_loss, critic_loss,entropy,cos,entropy_avg = self.train(states, actions, advs, values, probs, mask )
            self.agent.schedule_lr()
            eval_rewards = evaluate_model(self.agent, self.test_env, self.state_rms,system="avg")
            eval_rewards_majority = evaluate_model(self.agent, self.test_env, self.state_rms,system="majority")
            self.state_rms.update(states)
            self.print_logs(iteration, actor_loss, critic_loss, eval_rewards,cos,entropy,eval_rewards_majority,entropy_avg)

    @staticmethod
    def get_gae(rewards, values, dones, gamma=0.99, lam=0.95):
        advs = []
        gae = 0
        dones.append(0)
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * (values[step + 1]) * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advs.append(gae)

        advs.reverse()
        return np.vstack(advs)


    def calculate_log_probs(self,model, states, actions,head):
        policy_distribution = model(states)
        if self.agent.system_merge:
            policy=self.generate_dist_avg(policy_distribution)
        else:
            policy = policy_distribution[head]
        policy_probs_old = torch.stack([torch.sum(torch.mul(actions[i], policy[i]),dim=1)for i in range(self.num_act)])
        log_policy_prob_old_head = torch.log(policy_probs_old)
        return log_policy_prob_old_head,policy_distribution

    def compute_actor_loss(self, ratio, adv):
        pg_loss1 = adv * ratio
        pg_loss2 = adv * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(pg_loss1, pg_loss2).mean()
        return loss

    def my_entropy(self,y):
        loss=-torch.sum(y*torch.log(y),dim=-1)
        return loss
    def entropy_loss(self,out_policy,):
        entropy = 0#torch.zeros(out_policy[self.CP_names[0]].shape[0])#0
        for n in range(self.agent.n_actions):
            a = self.my_entropy(out_policy[n])
            b = self.my_entropy(torch.Tensor([1/self.agent.bins for _ in range(self.agent.bins)]))
            vec = torch.mean(a/b)  #torch.mean()
            entropy = entropy+vec
        return entropy/self.agent.n_actions

    def entropy_for_diversity(self,onehotencode,):
        out_policy = self.generate_dist_avg(onehotencode)
        entropy = 0#torch.zeros(out_policy[self.CP_names[0]].shape[0])#0
        for n in range(self.agent.n_actions):
            a = self.my_entropy(out_policy[n])
            b = self.my_entropy(torch.Tensor([1/self.agent.bins for _ in range(self.agent.bins)]))
            vec = torch.mean(a/b)  #torch.mean()
            entropy = entropy+vec
        return entropy/self.agent.n_actions


    def print_logs(self, iteration, actor_loss, critic_loss, eval_rewards,cos,entropy,eval_rewards_majority,entropy_avg):
        if iteration == 1:
            self.running_reward = eval_rewards
        else:
            self.running_reward = self.running_reward * 0.99 + eval_rewards * 0.01

        if iteration % 5 == 0:
            if self.agent.heads==1:
                print(f"Iter:{iteration}| "
                    f"Ep_Reward:{eval_rewards:.3f}| "
                    f"Running_reward:{self.running_reward:.3f}| "
                    f"Actor_Loss:{actor_loss:.3f}| "
                    f"Critic_Loss:{critic_loss:.3f}| "
                    f"Entropy:{entropy:.3f}| "
                    f"Iter_duration:{time.time() - self.start_time:.3f}| "
                    f"lr:{self.agent.actor_scheduler.get_last_lr()}")
            else:
                print(f"Iter:{iteration}| "
                    f"Ep_Reward:{eval_rewards:.3f}| "
                    f"Running_reward:{self.running_reward:.3f}| "
                    f"Actor_Loss:{actor_loss:.3f}| "
                    f"Critic_Loss:{critic_loss:.3f}| "
                    f"Entropy:{entropy:.3f}| "
                    f"Entropy avg:{entropy_avg:.3f}| "
                    f"Episode reward majority:{eval_rewards_majority:.3f}| "
                    f"Cos sim:{cos:.3f}| "
                    f"Iter_duration:{time.time() - self.start_time:.3f}| "
                    f"lr:{self.agent.actor_scheduler.get_last_lr()}")
            self.agent.save_weights(iteration, self.state_rms)

        with SummaryWriter(self.env_name + "/logs") as writer:
            writer.add_scalar("Episode running reward", self.running_reward, iteration)
            writer.add_scalar("Episode reward AVG", eval_rewards, iteration)
            writer.add_scalar("Episode reward majority", eval_rewards_majority, iteration)

            # writer.add_scalar("Correlation reward AVG diversity entropy", eval_rewards, entropy_avg)
            # writer.add_scalar("Correlation reward majority diversity entropy",  eval_rewards_majority, entropy_avg)

            writer.add_scalar("Actor loss", actor_loss, iteration)
            writer.add_scalar("Critic loss", critic_loss, iteration)
            writer.add_scalar("Entropy", entropy, iteration)
            writer.add_scalar("Entropy avg", entropy_avg, iteration)
            writer.add_scalar("Episode reward majority", eval_rewards_majority, iteration)
            if  self.agent.heads>1:
                writer.add_scalar("Cos sim", cos, iteration)
                # writer.add_scalar("Correlation reward AVG diversity cosine",eval_rewards, cos)
                # writer.add_scalar("Correlation reward majority diversity cosine", eval_rewards_majority, cos)


