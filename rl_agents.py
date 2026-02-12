"""
RL Agents for Multi-Attribute Auction
Using Stable-Baselines3 for training
"""

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from typing import Callable, List
import gymnasium as gym


class CustomActorCriticPolicy(nn.Module):
    """
    Custom policy network for seller agents
    Architecture designed for auction bidding
    """
    
    def __init__(self, observation_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        
        # Build network
        layers = []
        input_dim = observation_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(input_dim, action_dim),
            nn.Softplus()  # Ensure positive bids
        )
        
        # Value head (critic)
        self.value_head = nn.Linear(input_dim, 1)
    
    def forward(self, obs):
        features = self.feature_extractor(obs)
        action_mean = self.policy_head(features)
        value = self.value_head(features)
        return action_mean, value


class AuctionCallback(BaseCallback):
    """
    Custom callback for logging auction-specific metrics
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.win_rates = []
        self.bid_amounts = []
    
    def _on_step(self) -> bool:
        # Log custom metrics
        if 'winner' in self.locals.get('infos', [{}])[0]:
            info = self.locals['infos'][0]
            self.episode_rewards.append(self.locals.get('rewards', [0])[0])
            
            # Log to tensorboard if available
            if self.logger is not None:
                self.logger.record('auction/transaction_price', info.get('transaction_price', 0))
                self.logger.record('auction/buyer_utility', info.get('buyer_utility', 0))
                self.logger.record('auction/total_utility', 
                                 info.get('buyer_utility', 0) + sum(info.get('seller_utilities', [0])))
        
        return True


def make_auction_env(env_id: str = 'auction', **kwargs) -> Callable:
    """
    Create auction environment factory
    """
    def _init():
        from mav_auction_env import MultiAttributeAuctionEnv
        env = MultiAttributeAuctionEnv(**kwargs)
        env = Monitor(env)
        return env
    return _init


class RLSellerAgent:
    """
    RL-based seller agent wrapper
    """
    
    def __init__(
        self,
        seller_id: int,
        observation_space,
        action_space,
        algorithm: str = 'PPO',
        **kwargs
    ):
        self.seller_id = seller_id
        self.observation_space = observation_space
        self.action_space = action_space
        self.algorithm = algorithm
        
        # Create environment for this seller
        env_kwargs = kwargs.get('env_kwargs', {})
        env = self._create_wrapped_env(env_kwargs)
        
        # Initialize RL algorithm
        if algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                env,
                learning_rate=kwargs.get('learning_rate', 3e-4),
                n_steps=kwargs.get('n_steps', 2048),
                batch_size=kwargs.get('batch_size', 64),
                n_epochs=kwargs.get('n_epochs', 10),
                gamma=kwargs.get('gamma', 0.99),
                verbose=1,
                tensorboard_log=f"./logs/seller_{seller_id}/"
            )
        elif algorithm == 'SAC':
            self.model = SAC(
                'MlpPolicy',
                env,
                learning_rate=kwargs.get('learning_rate', 3e-4),
                buffer_size=kwargs.get('buffer_size', 100000),
                batch_size=kwargs.get('batch_size', 256),
                gamma=kwargs.get('gamma', 0.99),
                verbose=1,
                tensorboard_log=f"./logs/seller_{seller_id}/"
            )
        elif algorithm == 'TD3':
            self.model = TD3(
                'MlpPolicy',
                env,
                learning_rate=kwargs.get('learning_rate', 3e-4),
                buffer_size=kwargs.get('buffer_size', 100000),
                batch_size=kwargs.get('batch_size', 256),
                gamma=kwargs.get('gamma', 0.99),
                verbose=1,
                tensorboard_log=f"./logs/seller_{seller_id}/"
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _create_wrapped_env(self, env_kwargs):
        """Create a wrapped environment that focuses on this seller"""
        from mav_auction_env import MultiAttributeAuctionEnv
        
        class SingleSellerWrapper(gym.Wrapper):
            def __init__(self, env, seller_id):
                super().__init__(env)
                self.seller_id = seller_id
                self.step_count = 0
            
            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                self.step_count = 0
                
                # Skip to this seller's turn
                while self.env.current_seller < self.seller_id:
                    # Other sellers bid (using dummy bids)
                    dummy_bid = np.array([0.0])
                    obs, _, _, _, _ = self.env.step(dummy_bid)
                    self.step_count += 1
                
                return obs, info
            
            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.step_count += 1
                
                # Continue stepping until auction completes
                while not (terminated or truncated) and self.env.current_seller != self.seller_id:
                    # Other sellers bid
                    dummy_bid = np.array([0.0])
                    obs, _, terminated, truncated, _ = self.env.step(dummy_bid)
                    self.step_count += 1
                
                return obs, reward, terminated, truncated, info
        
        base_env = MultiAttributeAuctionEnv(**env_kwargs)
        wrapped_env = SingleSellerWrapper(base_env, self.seller_id)
        return Monitor(wrapped_env)
    
    def train(self, total_timesteps: int = 100000, callback=None):
        """Train the RL agent"""
        if callback is None:
            callback = AuctionCallback()
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    
    def get_action(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action from trained model"""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def save(self, path: str):
        """Save the trained model"""
        self.model.save(path)
    
    def load(self, path: str):
        """Load a trained model"""
        if self.algorithm == 'PPO':
            self.model = PPO.load(path)
        elif self.algorithm == 'SAC':
            self.model = SAC.load(path)
        elif self.algorithm == 'TD3':
            self.model = TD3.load(path)


class MultiAgentTrainer:
    """
    Train multiple seller agents simultaneously
    """
    
    def __init__(
        self,
        n_sellers: int,
        env_kwargs: dict,
        algorithm: str = 'PPO',
        **training_kwargs
    ):
        self.n_sellers = n_sellers
        self.env_kwargs = env_kwargs
        self.algorithm = algorithm
        self.training_kwargs = training_kwargs
        
        # Create agents
        from mav_auction_env import MultiAttributeAuctionEnv
        dummy_env = MultiAttributeAuctionEnv(**env_kwargs)
        
        self.agents = []
        for i in range(n_sellers):
            agent = RLSellerAgent(
                seller_id=i,
                observation_space=dummy_env.observation_space,
                action_space=dummy_env.action_space,
                algorithm=algorithm,
                env_kwargs=env_kwargs,
                **training_kwargs
            )
            self.agents.append(agent)
    
    def train_all(self, total_timesteps: int = 100000, save_dir: str = './models'):
        """Train all agents"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            print(f"\n{'='*60}")
            print(f"Training Seller Agent {i}")
            print(f"{'='*60}\n")
            
            agent.train(total_timesteps=total_timesteps)
            
            # Save model
            model_path = os.path.join(save_dir, f'seller_{i}_{self.algorithm}')
            agent.save(model_path)
            print(f"Saved model to {model_path}")
    
    def load_all(self, load_dir: str = './models'):
        """Load all trained agents"""
        import os
        
        for i, agent in enumerate(self.agents):
            model_path = os.path.join(load_dir, f'seller_{i}_{self.algorithm}')
            if os.path.exists(model_path + '.zip'):
                agent.load(model_path)
                print(f"Loaded agent {i} from {model_path}")
            else:
                print(f"Warning: Model not found at {model_path}")


def train_single_seller(
    seller_id: int = 0,
    n_sellers: int = 3,
    algorithm: str = 'PPO',
    total_timesteps: int = 100000,
    save_path: str = None
):
    """
    Convenience function to train a single seller agent
    """
    env_kwargs = {
        'n_sellers': n_sellers,
        'n_attributes': 2,
        'max_price': 100.0
    }
    
    from mav_auction_env import MultiAttributeAuctionEnv
    dummy_env = MultiAttributeAuctionEnv(**env_kwargs)
    
    agent = RLSellerAgent(
        seller_id=seller_id,
        observation_space=dummy_env.observation_space,
        action_space=dummy_env.action_space,
        algorithm=algorithm,
        env_kwargs=env_kwargs
    )
    
    print(f"Training seller {seller_id} using {algorithm}...")
    agent.train(total_timesteps=total_timesteps)
    
    if save_path:
        agent.save(save_path)
        print(f"Model saved to {save_path}")
    
    return agent
