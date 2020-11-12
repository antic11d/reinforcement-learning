
import numpy as np

from stable_baselines3 import PPO


def load_expert(model_file, device=None):
    model = PPO.load(model_file, device=device)
    return model


def load_trajectory_data(filename):
    trajectories = np.load(filename, allow_pickle=True)

    # data is stored in trajectories: (observations, actions, rewards)
    all_observations, all_actions, all_rewards = zip(*trajectories)

    all_observations=np.concatenate(all_observations, axis=0).squeeze()  # concat and remove extra middle dimension
    all_actions=np.concatenate(all_actions, axis=0).squeeze()

    rewards_of_expert = np.asarray([ np.sum(rews) for rews in all_rewards])
    print(f"{len(trajectories)} trajectories loaded: {len(all_observations)} transitions. Mean reward of expert:", np.mean(rewards_of_expert))
    return all_observations, all_actions, rewards_of_expert
