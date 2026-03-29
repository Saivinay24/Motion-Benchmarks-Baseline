import gym
import torch
import numpy as np

class VLMRewardWrapper(gym.Wrapper):
    """
    Project 2306 - DRL Integration.
    Wraps a character animation environment to use VLM scores as 
    reward shaping signals for 'human-like' motion.
    """
    def __init__(self, env, vlm_model, alpha=0.5):
        super().__init__(env)
        self.vlm = vlm_model
        self.alpha = alpha  # Balancing factor between physics and VLM perception
        self.prev_vlm_score = 0.0

    def step(self, action):
        obs, physics_reward, done, info = self.env.step(action)
        
        # Convert state/frame to VLM-compatible format
        vlm_input = self._preprocess_obs(obs)
        
        with torch.no_grad():
            # Query VLM: "How natural is this movement?"
            current_vlm_score = self.vlm.score_naturalness(vlm_input)
        
        # Reward Shaping: Positive signal if naturalness increases
        shaping_signal = current_vlm_score - self.prev_vlm_score
        total_reward = physics_reward + (self.alpha * shaping_signal)
        
        self.prev_vlm_score = current_vlm_score
        
        return obs, total_reward, done, info

    def _preprocess_obs(self, obs):
        # Implementation for converting DRL state to visual tensor
        return torch.tensor(obs).float().unsqueeze(0)