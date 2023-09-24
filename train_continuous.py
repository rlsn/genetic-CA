import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

def make_env(env_id, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

def train_single(run_name, envs, agent):
    total_timesteps = 1000000
    num_steps = agent.mem_size
    batch_size = num_steps * num_envs
    anneal_lr = True

    writer = SummaryWriter(f"runs/{run_name}")
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs, _ = envs.reset(seed=42)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)
    num_updates = total_timesteps // batch_size

    for update in range(1, num_updates + 1):
        # update loop
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * agent.init_learning_rate
            agent.lr = lrnow

        for step in range(0, num_steps//2):

            global_step += 1 * num_envs

            action = agent.step(next_obs, next_done)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            agent.reward(torch.tensor(reward).to(device).view(-1))

            done = np.logical_or(terminated, truncated)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        agent.update(5)
        print(f"{update} updates done")

if __name__=="__main__":
    from agent import Agent
    import time
    device="cpu"

    num_envs = 1
    envs = gym.vector.SyncVectorEnv(
            [make_env("BipedalWalker-v3", 0.99) for i in range(num_envs)]
        )

    run_name = f"BipedalWalker_{int(time.time())}"
    agent = Agent(envs.single_observation_space.shape[0],envs.single_action_space.shape[0],64).to(device)
    train_single(run_name, envs, agent)