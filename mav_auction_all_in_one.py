"""
Complete Multi-Attribute Vickrey (MAV) Auction Implementation
All-in-One File

Based on: "An Ascending Bid Multi-Attribute Auction Method"
Authors: Jin Xing, Shi Chunyi (Tsinghua University, 2006)

This file contains:
- Environment (MAVAuctionEnv)
- Agents (TruthfulSellerAgent, StrategicBidder, RLAgent)
- Evaluation tools (AuctionEvaluator)
- Main experiment script

Usage:
    python mav_auction_all_in_one.py --mode all
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import torch


# ============================================================================
# SECTION 1: ENVIRONMENT
# ============================================================================

class MAVAuctionEnv(gym.Env):
    """
    Multi-Attribute Vickrey (MAV) Auction Environment
    
    Key concepts:
    - Single item for sale with multiple attributes (weight, color, size, etc.)
    - Multiple sellers competing
    - Single buyer with valuation function over attributes
    - Sellers have cost functions over attributes
    - Winner: seller with highest surplus (buyer_value - seller_cost)
    - Payment: Vickrey pricing (second-price mechanism)
    """
    
    def __init__(
        self,
        n_sellers: int = 3,
        n_attributes: int = 2,
        max_attribute_value: float = 100.0,
        valuation_type: str = 'linear'
    ):
        super().__init__()
        
        self.n_sellers = n_sellers
        self.n_attributes = n_attributes
        self.max_attribute_value = max_attribute_value
        self.valuation_type = valuation_type
        
        # Action space: propose attribute values
        self.action_space = spaces.Box(
            low=0.0,
            high=max_attribute_value,
            shape=(n_attributes,),
            dtype=np.float32
        )
        
        # Observation space: [buyer_weights, own_cost_params, current_bids]
        obs_dim = n_attributes + n_attributes + n_sellers * n_attributes
        self.observation_space = spaces.Box(
            low=0.0,
            high=max_attribute_value,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Generate buyer valuation and seller costs
        self._generate_auction_parameters()
        
        self.reset()
    
    def _generate_auction_parameters(self):
        """Generate buyer valuation and seller cost functions"""
        # Buyer valuation: linear combination of attributes
        self.buyer_weights = np.random.uniform(0.5, 2.0, self.n_attributes)
        self.buyer_weights /= np.sum(self.buyer_weights)  # Normalize
        
        # Seller cost parameters (each seller has different efficiency)
        self.seller_costs = []
        for _ in range(self.n_sellers):
            # Cost per unit of each attribute
            cost_params = np.random.uniform(0.2, 1.5, self.n_attributes)
            self.seller_costs.append(cost_params)
        
        self.seller_costs = np.array(self.seller_costs)
    
    def compute_buyer_value(self, attributes: np.ndarray) -> float:
        """Compute buyer's valuation for given attributes"""
        if self.valuation_type == 'linear':
            return np.dot(self.buyer_weights, attributes)
        elif self.valuation_type == 'cobb_douglas':
            # Cobb-Douglas: V = prod(a_i^w_i)
            return np.prod(np.power(attributes + 1e-8, self.buyer_weights))
        else:
            return np.dot(self.buyer_weights, attributes)
    
    def compute_seller_cost(self, seller_id: int, attributes: np.ndarray) -> float:
        """Compute seller's cost for providing given attributes"""
        return np.dot(self.seller_costs[seller_id], attributes)
    
    def compute_surplus(self, seller_id: int, attributes: np.ndarray) -> float:
        """Compute surplus: buyer_value - seller_cost"""
        value = self.compute_buyer_value(attributes)
        cost = self.compute_seller_cost(seller_id, attributes)
        return value - cost
    
    def determine_winner_and_payment(
        self,
        bids: Dict[int, np.ndarray]
    ) -> Tuple[int, float, np.ndarray]:
        """
        Determine winner and payment using Vickrey mechanism
        
        Returns:
            winner_id: ID of winning seller
            payment: Payment to winner (second-price)
            winning_attributes: Winning attribute proposal
        """
        # Compute surplus for each seller's bid
        surpluses = []
        for seller_id, attributes in bids.items():
            surplus = self.compute_surplus(seller_id, attributes)
            surpluses.append((seller_id, surplus, attributes))
        
        # Sort by surplus (descending)
        surpluses.sort(key=lambda x: x[1], reverse=True)
        
        if len(surpluses) == 0:
            return -1, 0.0, np.zeros(self.n_attributes)
        
        # Winner is highest surplus
        winner_id, winner_surplus, winner_attributes = surpluses[0]
        
        # Payment: buyer_value - second_highest_surplus (Vickrey pricing)
        buyer_value = self.compute_buyer_value(winner_attributes)
        
        if len(surpluses) > 1:
            second_surplus = surpluses[1][1]
        else:
            second_surplus = 0.0
        
        payment = buyer_value - second_surplus
        
        return winner_id, payment, winner_attributes
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Current seller submitting bid
        self.current_seller = 0
        
        # Bids from all sellers
        self.bids = {}
        
        obs = self._get_observation(self.current_seller)
        info = {}
        
        return obs, info
    
    def _get_observation(self, seller_id: int) -> np.ndarray:
        """Get observation for seller"""
        # Buyer weights (what buyer values)
        buyer_obs = self.buyer_weights.copy()
        
        # Own cost parameters
        own_cost = self.seller_costs[seller_id].copy()
        
        # Current bids from all sellers (including self)
        all_bids = np.zeros(self.n_sellers * self.n_attributes, dtype=np.float32)
        for sid, bid in self.bids.items():
            start_idx = sid * self.n_attributes
            end_idx = start_idx + self.n_attributes
            all_bids[start_idx:end_idx] = bid
        
        obs = np.concatenate([
            buyer_obs,
            own_cost,
            all_bids
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute step"""
        # Clip action to valid range
        attributes = np.clip(action, 0, self.max_attribute_value)
        
        # Store bid
        self.bids[self.current_seller] = attributes.copy()
        
        # Move to next seller
        self.current_seller += 1
        
        if self.current_seller >= self.n_sellers:
            # All sellers have bid - determine winner
            terminated = True
            
            winner_id, payment, winning_attributes = self.determine_winner_and_payment(self.bids)
            
            # Compute rewards for all sellers
            rewards = np.zeros(self.n_sellers)
            
            if winner_id >= 0:
                winner_cost = self.compute_seller_cost(winner_id, winning_attributes)
                winner_profit = payment - winner_cost
                rewards[winner_id] = winner_profit
            
            # Return reward for seller 0 (can extend to multi-agent)
            reward = float(rewards[0])
            
            info = {
                'winner_id': winner_id,
                'payment': payment,
                'winning_attributes': winning_attributes,
                'all_rewards': rewards,
                'buyer_value': self.compute_buyer_value(winning_attributes) if winner_id >= 0 else 0.0
            }
            
            obs = np.zeros_like(self._get_observation(0))
        else:
            obs = self._get_observation(self.current_seller)
            reward = 0.0
            terminated = False
            info = {}
        
        truncated = False
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render environment"""
        pass


# ============================================================================
# SECTION 2: AGENTS
# ============================================================================

class TruthfulSellerAgent:
    """
    Baseline seller that bids truthfully based on optimal surplus
    
    Computes: argmax_a [V(a) - C_i(a)]
    where V is buyer valuation, C_i is own cost
    """
    
    def __init__(self, seller_id: int, env):
        self.seller_id = seller_id
        self.env = env
    
    def get_bid(self, observation: np.ndarray) -> np.ndarray:
        """Get truthful bid that maximizes surplus"""
        # Extract buyer weights and own cost from observation
        n_attrs = self.env.n_attributes
        buyer_weights = observation[:n_attrs]
        own_cost = observation[n_attrs:2*n_attrs]
        
        # Compute optimal attributes: maximize (buyer_value - cost)
        # For linear valuation: optimal when buyer_weight_i / cost_i is maximized
        
        # Simple heuristic: allocate proportionally to (buyer_weight / cost)
        ratios = buyer_weights / (own_cost + 1e-8)
        
        # Normalize and scale
        attributes = ratios / np.sum(ratios) * self.env.max_attribute_value * 0.8
        
        return np.clip(attributes, 0, self.env.max_attribute_value)


class StrategicBidder:
    """
    Strategic seller that shades bid to increase profit
    
    Deviates from truthful bidding to test incentive compatibility
    """
    
    def __init__(self, seller_id: int, env, strategy: str = 'overbid'):
        self.seller_id = seller_id
        self.env = env
        self.strategy = strategy
    
    def get_bid(self, observation: np.ndarray) -> np.ndarray:
        """Get strategic bid"""
        # Start with truthful bid
        truthful_agent = TruthfulSellerAgent(self.seller_id, self.env)
        base_bid = truthful_agent.get_bid(observation)
        
        if self.strategy == 'overbid':
            # Increase attributes by 20%
            return np.clip(base_bid * 1.2, 0, self.env.max_attribute_value)
        elif self.strategy == 'underbid':
            # Decrease attributes by 20%
            return np.clip(base_bid * 0.8, 0, self.env.max_attribute_value)
        elif self.strategy == 'random':
            # Add random noise
            noise = np.random.uniform(-0.1, 0.1, self.env.n_attributes)
            return np.clip(base_bid * (1 + noise), 0, self.env.max_attribute_value)
        else:
            return base_bid


class RLAgent:
    """RL-based seller using PPO/SAC/TD3"""
    
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
        
        self.model = None
        self._init_model(**kwargs)
    
    def _init_model(self, **kwargs):
        """Initialize RL model"""
        env = MAVAuctionEnv(**kwargs.get('env_kwargs', {}))
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        if self.algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                env,
                learning_rate=kwargs.get('learning_rate', 3e-4),
                n_steps=kwargs.get('n_steps', 2048),
                batch_size=kwargs.get('batch_size', 64),
                verbose=0
            )
        elif self.algorithm == 'SAC':
            self.model = SAC(
                'MlpPolicy',
                env,
                learning_rate=kwargs.get('learning_rate', 3e-4),
                buffer_size=kwargs.get('buffer_size', 100000),
                verbose=0
            )
        elif self.algorithm == 'TD3':
            self.model = TD3(
                'MlpPolicy',
                env,
                learning_rate=kwargs.get('learning_rate', 3e-4),
                buffer_size=kwargs.get('buffer_size', 100000),
                verbose=0
            )
    
    def train(self, total_timesteps: int = 100000):
        """Train the RL agent"""
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    def get_bid(self, observation: np.ndarray) -> np.ndarray:
        """Get bid from RL policy"""
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def save(self, path: str):
        """Save model"""
        self.model.save(path)
    
    def load(self, path: str):
        """Load model"""
        if self.algorithm == 'PPO':
            self.model = PPO.load(path)
        elif self.algorithm == 'SAC':
            self.model = SAC.load(path)
        elif self.algorithm == 'TD3':
            self.model = TD3.load(path)


class MultiAgentAuctionSystem:
    """System for managing multiple sellers in MAV auction"""
    
    def __init__(
        self,
        n_sellers: int,
        n_attributes: int,
        agent_types: List[str] = None
    ):
        self.n_sellers = n_sellers
        self.n_attributes = n_attributes
        
        if agent_types is None:
            agent_types = ['truthful'] * n_sellers
        
        self.env = MAVAuctionEnv(n_sellers=n_sellers, n_attributes=n_attributes)
        self.agents = []
        self._create_agents(agent_types)
    
    def _create_agents(self, agent_types: List[str]):
        """Create agents based on types"""
        for seller_id, atype in enumerate(agent_types):
            if atype == 'truthful':
                agent = TruthfulSellerAgent(seller_id, self.env)
            elif atype in ['overbid', 'underbid', 'random']:
                agent = StrategicBidder(seller_id, self.env, strategy=atype)
            elif atype in ['PPO', 'SAC', 'TD3']:
                agent = RLAgent(
                    seller_id,
                    self.env.observation_space,
                    self.env.action_space,
                    algorithm=atype,
                    env_kwargs={'n_sellers': self.n_sellers, 'n_attributes': self.n_attributes}
                )
            else:
                raise ValueError(f"Unknown agent type: {atype}")
            
            self.agents.append(agent)
    
    def run_auction(self) -> Dict:
        """Run a complete auction"""
        obs, _ = self.env.reset()
        bids = {}
        
        for seller_id, agent in enumerate(self.agents):
            obs = self.env._get_observation(seller_id)
            
            if isinstance(agent, (TruthfulSellerAgent, StrategicBidder)):
                bid = agent.get_bid(obs)
            else:  # RLAgent
                bid = agent.get_bid(obs)
            
            bids[seller_id] = bid
        
        # Determine winner
        winner_id, payment, winning_attributes = self.env.determine_winner_and_payment(bids)
        
        # Compute utilities
        seller_utilities = {}
        for seller_id in range(self.n_sellers):
            if seller_id == winner_id:
                cost = self.env.compute_seller_cost(seller_id, winning_attributes)
                utility = payment - cost
            else:
                utility = 0.0
            seller_utilities[seller_id] = utility
        
        buyer_value = self.env.compute_buyer_value(winning_attributes) if winner_id >= 0 else 0.0
        buyer_utility = buyer_value - payment if winner_id >= 0 else 0.0
        
        # Compute optimal surplus
        optimal_surplus = max([
            self.env.compute_surplus(sid, bids[sid])
            for sid in range(self.n_sellers)
        ])
        
        achieved_surplus = buyer_value - self.env.compute_seller_cost(winner_id, winning_attributes) if winner_id >= 0 else 0.0
        
        efficiency = (achieved_surplus / optimal_surplus * 100) if optimal_surplus > 0 else 0
        
        return {
            'winner_id': winner_id,
            'payment': payment,
            'winning_attributes': winning_attributes,
            'seller_utilities': seller_utilities,
            'buyer_value': buyer_value,
            'buyer_utility': buyer_utility,
            'total_surplus': achieved_surplus,
            'optimal_surplus': optimal_surplus,
            'efficiency': efficiency,
            'bids': bids
        }
    
    def train_rl_agents(self, total_timesteps: int = 100000):
        """Train all RL agents"""
        for agent in self.agents:
            if isinstance(agent, RLAgent):
                print(f"Training RL agent {agent.seller_id}...")
                agent.train(total_timesteps)
    
    def save_models(self, save_dir: str = './models'):
        """Save all RL models"""
        os.makedirs(save_dir, exist_ok=True)
        
        for agent in self.agents:
            if isinstance(agent, RLAgent):
                path = os.path.join(save_dir, f'seller_{agent.seller_id}_{agent.algorithm}')
                agent.save(path)
                print(f"Saved model: {path}")


# ============================================================================
# SECTION 3: EVALUATION
# ============================================================================

class AuctionEvaluator:
    """Evaluate and compare different strategies"""
    
    def __init__(
        self,
        n_sellers: int = 3,
        n_attributes: int = 2
    ):
        self.n_sellers = n_sellers
        self.n_attributes = n_attributes
    
    def evaluate_strategy(
        self,
        agent_types: List[str],
        n_episodes: int = 1000,
        strategy_name: str = 'Unknown'
    ) -> Dict:
        """Evaluate a strategy over multiple episodes"""
        results = {
            'efficiencies': [],
            'total_surpluses': [],
            'seller_utilities': [[] for _ in range(self.n_sellers)],
            'buyer_utilities': [],
            'winner_distribution': np.zeros(self.n_sellers)
        }
        
        for episode in range(n_episodes):
            system = MultiAgentAuctionSystem(
                n_sellers=self.n_sellers,
                n_attributes=self.n_attributes,
                agent_types=agent_types
            )
            
            result = system.run_auction()
            
            results['efficiencies'].append(result['efficiency'])
            results['total_surpluses'].append(result['total_surplus'])
            results['buyer_utilities'].append(result['buyer_utility'])
            
            if result['winner_id'] >= 0:
                results['winner_distribution'][result['winner_id']] += 1
            
            for seller_id in range(self.n_sellers):
                results['seller_utilities'][seller_id].append(
                    result['seller_utilities'].get(seller_id, 0.0)
                )
        
        stats = {
            'strategy_name': strategy_name,
            'mean_efficiency': np.mean(results['efficiencies']),
            'std_efficiency': np.std(results['efficiencies']),
            'mean_surplus': np.mean(results['total_surpluses']),
            'mean_buyer_utility': np.mean(results['buyer_utilities']),
            'mean_seller_utilities': [
                np.mean(results['seller_utilities'][i])
                for i in range(self.n_sellers)
            ],
            'winner_distribution': results['winner_distribution'] / n_episodes,
            'raw_results': results
        }
        
        return stats
    
    def compare_strategies(
        self,
        strategies: Dict[str, List[str]],
        n_episodes: int = 1000
    ) -> pd.DataFrame:
        """Compare multiple strategies"""
        all_stats = []
        
        for strategy_name, agent_types in strategies.items():
            print(f"\nEvaluating strategy: {strategy_name}")
            stats = self.evaluate_strategy(
                agent_types,
                n_episodes=n_episodes,
                strategy_name=strategy_name
            )
            all_stats.append(stats)
        
        comparison_data = []
        for stats in all_stats:
            row = {
                'Strategy': stats['strategy_name'],
                'Mean Efficiency (%)': stats['mean_efficiency'],
                'Std Efficiency': stats['std_efficiency'],
                'Mean Surplus': stats['mean_surplus'],
                'Mean Buyer Utility': stats['mean_buyer_utility']
            }
            
            for seller_id in range(self.n_sellers):
                row[f'Seller {seller_id} Utility'] = stats['mean_seller_utilities'][seller_id]
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        self.detailed_stats = all_stats
        
        return df
    
    def plot_comparison(self, save_path: str = None):
        """Create visualization comparing strategies"""
        if not hasattr(self, 'detailed_stats'):
            print("No comparison data available. Run compare_strategies first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MAV Auction Strategy Comparison', fontsize=16)
        
        strategies = [s['strategy_name'] for s in self.detailed_stats]
        
        # 1. Efficiency
        ax = axes[0, 0]
        efficiencies = [s['mean_efficiency'] for s in self.detailed_stats]
        eff_stds = [s['std_efficiency'] for s in self.detailed_stats]
        ax.bar(strategies, efficiencies, yerr=eff_stds, capsize=5, color='steelblue')
        ax.set_ylabel('Efficiency (%)')
        ax.set_title('Mean Efficiency')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim([0, 105])
        
        # 2. Total Surplus
        ax = axes[0, 1]
        surpluses = [s['mean_surplus'] for s in self.detailed_stats]
        ax.bar(strategies, surpluses, color='green')
        ax.set_ylabel('Total Surplus')
        ax.set_title('Mean Total Surplus')
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Buyer Utility
        ax = axes[0, 2]
        buyer_utils = [s['mean_buyer_utility'] for s in self.detailed_stats]
        ax.bar(strategies, buyer_utils, color='coral')
        ax.set_ylabel('Buyer Utility')
        ax.set_title('Mean Buyer Utility')
        ax.tick_params(axis='x', rotation=45)
        
        # 4. Seller Utilities
        ax = axes[1, 0]
        x = np.arange(len(strategies))
        width = 0.25
        utilities = np.array([s['mean_seller_utilities'] for s in self.detailed_stats])
        
        for seller_id in range(self.n_sellers):
            ax.bar(x + seller_id*width, utilities[:, seller_id], width, 
                  label=f'Seller {seller_id}')
        
        ax.set_ylabel('Utility')
        ax.set_title('Mean Seller Utilities')
        ax.set_xticks(x + width)
        ax.set_xticklabels(strategies, rotation=45)
        ax.legend()
        
        # 5. Efficiency Distribution
        ax = axes[1, 1]
        for stats in self.detailed_stats:
            ax.hist(stats['raw_results']['efficiencies'], bins=20, alpha=0.5,
                   label=stats['strategy_name'])
        ax.set_xlabel('Efficiency (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Efficiency Distribution')
        ax.legend()
        
        # 6. Winner Distribution
        ax = axes[1, 2]
        x = np.arange(self.n_sellers)
        width = 0.8 / len(self.detailed_stats)
        
        for i, stats in enumerate(self.detailed_stats):
            offset = (i - len(self.detailed_stats)/2) * width
            ax.bar(x + offset, stats['winner_distribution'], width,
                  label=stats['strategy_name'])
        
        ax.set_xlabel('Seller ID')
        ax.set_ylabel('Win Rate')
        ax.set_title('Winner Distribution')
        ax.set_xticks(x)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def verify_incentive_compatibility(self, n_trials: int = 100) -> Dict:
        """
        Verify incentive compatibility:
        Truthful bidding should yield highest utility for sellers
        """
        print("\nVerifying Incentive Compatibility...")
        print("="*60)
        
        results = {}
        
        # Test 1: All truthful
        print("\nTest 1: All Truthful (Baseline)")
        truthful_stats = self.evaluate_strategy(
            ['truthful'] * self.n_sellers,
            n_episodes=n_trials,
            strategy_name='All Truthful'
        )
        
        results['truthful'] = truthful_stats
        
        # Test 2: One overbids
        print("\nTest 2: One Seller Overbids")
        overbid_types = ['overbid'] + ['truthful'] * (self.n_sellers - 1)
        overbid_stats = self.evaluate_strategy(
            overbid_types,
            n_episodes=n_trials,
            strategy_name='One Overbid'
        )
        
        results['overbid'] = overbid_stats
        
        # Test 3: One underbids
        print("\nTest 3: One Seller Underbids")
        underbid_types = ['underbid'] + ['truthful'] * (self.n_sellers - 1)
        underbid_stats = self.evaluate_strategy(
            underbid_types,
            n_episodes=n_trials,
            strategy_name='One Underbid'
        )
        
        results['underbid'] = underbid_stats
        
        # Comparison
        print("\n" + "="*60)
        print("Incentive Compatibility Verification:")
        
        truthful_seller0_util = truthful_stats['mean_seller_utilities'][0]
        overbid_seller0_util = overbid_stats['mean_seller_utilities'][0]
        underbid_seller0_util = underbid_stats['mean_seller_utilities'][0]
        
        print(f"\nSeller 0 utility:")
        print(f"  Truthful: {truthful_seller0_util:.2f}")
        print(f"  Overbid:  {overbid_seller0_util:.2f}")
        print(f"  Underbid: {underbid_seller0_util:.2f}")
        
        if (truthful_seller0_util >= overbid_seller0_util and 
            truthful_seller0_util >= underbid_seller0_util):
            print("\n✓ VERIFIED: Truthful bidding is optimal")
        else:
            print("\n✗ NOT VERIFIED: Strategic bidding can be better")
        
        print("="*60)
        
        return results


def create_comparison_report(
    evaluator: AuctionEvaluator,
    strategies: Dict[str, List[str]],
    n_episodes: int = 1000,
    output_dir: str = './results'
):
    """Create comprehensive comparison report"""
    os.makedirs(output_dir, exist_ok=True)
    
    df = evaluator.compare_strategies(strategies, n_episodes=n_episodes)
    
    csv_path = os.path.join(output_dir, 'mav_strategy_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    plot_path = os.path.join(output_dir, 'mav_strategy_comparison.png')
    evaluator.plot_comparison(save_path=plot_path)
    
    verification = evaluator.verify_incentive_compatibility(n_trials=n_episodes)
    
    report_path = os.path.join(output_dir, 'mav_verification_report.txt')
    with open(report_path, 'w') as f:
        f.write("MAV Auction Verification Report\n")
        f.write("="*60 + "\n\n")
        
        f.write("Strategy Comparison:\n")
        f.write(df.to_string())
        f.write("\n\n")
        
        f.write("Incentive Compatibility Verification:\n")
        for strategy, stats in verification.items():
            f.write(f"\n{strategy}:\n")
            f.write(f"  Mean Efficiency: {stats['mean_efficiency']:.2f}%\n")
            f.write(f"  Seller 0 Utility: {stats['mean_seller_utilities'][0]:.2f}\n")
        f.write("\n")
    
    print(f"Report saved to {report_path}")
    
    return df, verification


# ============================================================================
# SECTION 4: MAIN EXPERIMENT PIPELINE
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='MAV Auction RL Experiments'
    )
    
    parser.add_argument('--n_sellers', type=int, default=3)
    parser.add_argument('--n_attributes', type=int, default=2)
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'SAC', 'TD3'])
    parser.add_argument('--total_timesteps', type=int, default=100000)
    parser.add_argument('--n_eval_episodes', type=int, default=1000)
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'evaluate', 'verify', 'all'])
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def quick_demo():
    """Quick demonstration"""
    print("\n" + "="*80)
    print("QUICK DEMO - MAV Auction")
    print("="*80 + "\n")
    
    env = MAVAuctionEnv(n_sellers=2, n_attributes=2)
    
    print("Environment created:")
    print(f"  - {env.n_sellers} sellers")
    print(f"  - {env.n_attributes} attributes")
    print(f"  - Buyer weights: {env.buyer_weights}")
    
    print("\nSeller cost parameters:")
    for i in range(env.n_sellers):
        print(f"  Seller {i}: {env.seller_costs[i]}")
    
    print("\n" + "="*80)
    print("Testing Strategies:")
    print("="*80)
    
    strategies = [
        (['truthful', 'truthful'], 'Both Truthful'),
        (['overbid', 'truthful'], 'One Overbids'),
        (['underbid', 'truthful'], 'One Underbids')
    ]
    
    for agent_types, name in strategies:
        system = MultiAgentAuctionSystem(
            n_sellers=2,
            n_attributes=2,
            agent_types=agent_types
        )
        
        result = system.run_auction()
        
        print(f"\n{name}:")
        print(f"  Winner: Seller {result['winner_id']}")
        print(f"  Payment: {result['payment']:.2f}")
        print(f"  Attributes: {result['winning_attributes']}")
        print(f"  Efficiency: {result['efficiency']:.1f}%")
        print(f"  Seller utilities: {result['seller_utilities']}")
    
    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80)


def run_experiment(args):
    """Run complete experiment pipeline"""
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("MAV AUCTION RL EXPERIMENTS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Sellers: {args.n_sellers}")
    print(f"  Attributes: {args.n_attributes}")
    print(f"  RL Algorithm: {args.algorithm}")
    print(f"  Evaluation episodes: {args.n_eval_episodes}")
    
    evaluator = AuctionEvaluator(
        n_sellers=args.n_sellers,
        n_attributes=args.n_attributes
    )
    
    # Define strategies
    strategies = {
        'All Truthful': ['truthful'] * args.n_sellers,
        'One Overbid': ['overbid'] + ['truthful'] * (args.n_sellers - 1),
        'One Underbid': ['underbid'] + ['truthful'] * (args.n_sellers - 1),
    }
    
    if args.mode in ['evaluate', 'all']:
        print("\n" + "="*80)
        print("EVALUATION PHASE")
        print("="*80)
        
        df, verification = create_comparison_report(
            evaluator,
            strategies,
            n_episodes=args.n_eval_episodes,
            output_dir=args.output_dir
        )
        
        print("\nResults:")
        print(df.to_string(index=False))
    
    if args.mode in ['verify', 'all']:
        print("\n" + "="*80)
        print("VERIFICATION PHASE")
        print("="*80)
        
        evaluator.verify_incentive_compatibility(n_trials=args.n_eval_episodes)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 1 or '--demo' in sys.argv:
        quick_demo()
    else:
        args = parse_args()
        run_experiment(args)
