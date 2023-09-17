import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
import torch.optim as optim

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size, mem_size = 2048, num_envs = 1, learning_rate=3e-4):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, act_size), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_size))

        self.init_learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)


        self.obs_size = obs_size
        self.act_size = act_size
        self.num_envs = num_envs
        self.mem_size = mem_size

        self.current_buffer_pos = -1

        self.register_buffer("obs", torch.zeros((mem_size, num_envs, obs_size)))
        self.register_buffer("actions", torch.zeros((mem_size, num_envs, act_size)))
        self.register_buffer("logprobs", torch.zeros((mem_size, num_envs)))
        self.register_buffer("rewards", torch.zeros((mem_size, num_envs)))
        self.register_buffer("dones", torch.zeros((mem_size, num_envs)))
        self.register_buffer("values", torch.zeros((mem_size, num_envs)))

    @property
    def lr(self):
        return self.optimizer.param_groups[0]["lr"]

    @lr.setter
    def lr(self,lr):
        self.optimizer.param_groups[0]["lr"] = lr

    def get_value(self, x):
        return self.critic(x)

    # def get_action_and_value(self, x, action=None):
    #     logits = self.actor(x)
    #     probs = Categorical(logits=logits)
    #     if action is None:
    #         action = probs.sample()
    #     return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    # def save_checkpoint(self, run_name, model_name, steps):
    #     torch.save(self.state_dict(), f"runs/{run_name}/checkpoints/{model_name}_{steps}.pt")
    #     print(f"checkpoint saved")

    # def load_checkpoint(self, run_name, model_name, steps):
    #     self.load_state_dict(torch.load(f"runs/{run_name}/checkpoints/{model_name}_{steps}.pt"))

    def get_next_buffer_pos(self):
        self.current_buffer_pos+=1
        if self.current_buffer_pos>=self.mem_size:
            self.current_buffer_pos=0
        return self.current_buffer_pos

    def step(self, next_obs, next_done):
        step = self.get_next_buffer_pos()
        self.obs[step] = next_obs
        self.dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = self.get_action_and_value(next_obs)
            self.values[step] = value.flatten()
        self.actions[step] = action
        self.logprobs[step] = logprob
        self.rewards[step] = torch.zeros(self.num_envs)
        return action

    def forward(self, *args, **kargs):
        return self.step(*args, **kargs)

    def reward(self, last_reward):
        self.rewards[self.current_buffer_pos] = last_reward

    def update(self, next_obs, next_done, gae_lambda = 0.95, gamma=0.99, minibatch_size = 32,
               update_epochs = 10, clip_coef = 0.2, norm_adv = True, clip_vloss = True,
               ent_coef = 0.0, vf_coef = 0.5, max_grad_norm = 0.5):
        # bootstrap value if not done, GAE
        with torch.no_grad():
            next_value = self.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards, device= self.rewards.device)
            lastgaelam = 0
            for t in np.roll(np.arange(self.mem_size)[::-1],self.current_buffer_pos+1):
                if t == self.current_buffer_pos:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values


        # flatten the batch
        b_obs = self.obs.reshape((-1,self.obs_size))
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,self.act_size))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimizing the policy and value network
        batch_size = self.mem_size * self.num_envs
        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                self.optimizer.step()