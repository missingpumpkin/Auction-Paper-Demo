"""
Complete Combinatorial Auction with Decreasing Marginal Utilities
All-in-One Implementation

Based on: "A Winner Determine Algorithm for Combinatorial Auctions 
with Decreasing Marginal Utilities" (Jin Xing, Shi Chunyi, 2006)

Usage:
    python combinatorial_auction_all_in_one.py --mode all
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Set
from itertools import combinations, chain
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import torch


# ============================================================================
# SECTION 1: ENVIRONMENT
# ============================================================================

class CombinatorialAuctionEnv(gym.Env):
    """
    Combinatorial Auction Environment with Decreasing Marginal Utilities
    """
    
    def __init__(
        self,
        n_items: int = 10,
        n_buyers: int = 3,
        max_bundle_size: int = 5,
        decreasing_rate: float = 0.8,
        reserve_price: float = 0.0
    ):
        super().__init__()
        
        self.n_items = n_items
        self.n_buyers = n_buyers
        self.max_bundle_size = min(max_bundle_size, n_items)
        self.decreasing_rate = decreasing_rate
        self.reserve_price = reserve_price
        
        # Action space: bid on bundles
        self.action_space = spaces.Box(
            low=0.0,
            high=1000.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation space
        obs_dim = n_items + 2
        self.observation_space = spaces.Box(
            low=0.0,
            high=1000.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self._generate_valuations()
        self.reset()
    
    def _generate_valuations(self):
        """Generate buyer valuations with decreasing marginal utilities"""
        self.buyer_valuations = []
        
        for buyer_id in range(self.n_buyers):
            base_values = np.random.uniform(10, 50, self.n_items)
            valuation_dict = {}
            
            for size in range(1, self.max_bundle_size + 1):
                for bundle in combinations(range(self.n_items), size):
                    bundle_set = frozenset(bundle)
                    sorted_items = sorted(bundle, key=lambda x: base_values[x], reverse=True)
                    
                    total_value = 0.0
                    for idx, item in enumerate(sorted_items):
                        marginal_value = base_values[item] * (self.decreasing_rate ** idx)
                        total_value += marginal_value
                    
                    valuation_dict[bundle_set] = total_value
            
            self.buyer_valuations.append(valuation_dict)
    
    def get_bundle_value(self, buyer_id: int, bundle: Set[int]) -> float:
        """Get buyer's valuation for a bundle"""
        bundle_set = frozenset(bundle)
        return self.buyer_valuations[buyer_id].get(bundle_set, 0.0)
    
    def check_decreasing_marginal_utility(self, buyer_id: int) -> bool:
        """Verify decreasing marginal utility property"""
        valuation = self.buyer_valuations[buyer_id]
        
        for item in range(self.n_items):
            for S_size in range(self.max_bundle_size):
                for S in combinations([x for x in range(self.n_items) if x != item], S_size):
                    S_set = frozenset(S)
                    S_with_i = frozenset(S) | {item}
                    
                    if S_with_i not in valuation:
                        continue
                    
                    marginal_S = valuation.get(S_with_i, 0) - valuation.get(S_set, 0)
                    
                    remaining_items = [x for x in range(self.n_items) if x not in S and x != item]
                    for add_size in range(1, len(remaining_items) + 1):
                        for add_items in combinations(remaining_items, add_size):
                            T_set = S_set | frozenset(add_items)
                            T_with_i = T_set | {item}
                            
                            if T_with_i not in valuation:
                                continue
                            
                            marginal_T = valuation.get(T_with_i, 0) - valuation.get(T_set, 0)
                            
                            if marginal_S < marginal_T - 1e-6:
                                return False
        
        return True
    
    def check_1_UNT_condition(self, allocation: Dict[int, Set[int]]) -> bool:
        """Check 1-UNT condition"""
        current_utility = self._compute_total_utility(allocation)
        
        for item in range(self.n_items):
            current_owner = None
            for buyer_id, bundle in allocation.items():
                if item in bundle:
                    current_owner = buyer_id
                    break
            
            if current_owner is None:
                continue
            
            for new_owner in range(self.n_buyers):
                if new_owner == current_owner:
                    continue
                
                new_allocation = {k: set(v) for k, v in allocation.items()}
                new_allocation[current_owner].remove(item)
                if new_owner not in new_allocation:
                    new_allocation[new_owner] = set()
                new_allocation[new_owner].add(item)
                
                new_utility = self._compute_total_utility(new_allocation)
                
                if new_utility > current_utility + 1e-6:
                    return False
        
        return True
    
    def _compute_total_utility(self, allocation: Dict[int, Set[int]]) -> float:
        """Compute total utility of allocation"""
        total = 0.0
        for buyer_id, bundle in allocation.items():
            if bundle:
                total += self.get_bundle_value(buyer_id, bundle)
        return total
    
    def winner_determination_greedy(
        self, 
        bids: Dict[int, Dict[Set[int], float]]
    ) -> Tuple[Dict[int, Set[int]], float]:
        """Greedy winner determination"""
        all_bids = []
        for buyer_id, buyer_bids in bids.items():
            for bundle, bid in buyer_bids.items():
                all_bids.append((buyer_id, bundle, bid))
        
        all_bids.sort(key=lambda x: x[2], reverse=True)
        
        allocation = {}
        allocated_items = set()
        total_value = 0.0
        
        for buyer_id, bundle, bid in all_bids:
            if not (bundle & allocated_items):
                allocation[buyer_id] = bundle
                allocated_items.update(bundle)
                total_value += bid
        
        return allocation, total_value
    
    def winner_determination_1UNT(
        self,
        bids: Dict[int, Dict[Set[int], float]]
    ) -> Tuple[Dict[int, Set[int]], float]:
        """1-UNT check-based iterative winner determination"""
        allocation, _ = self.winner_determination_greedy(bids)
        
        improved = True
        max_iterations = 100
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            current_value = sum([
                bids.get(buyer_id, {}).get(frozenset(bundle), 0.0)
                for buyer_id, bundle in allocation.items()
            ])
            
            for item in range(self.n_items):
                current_owner = None
                for buyer_id, bundle in allocation.items():
                    if item in bundle:
                        current_owner = buyer_id
                        break
                
                if current_owner is None:
                    continue
                
                for new_owner in range(self.n_buyers):
                    if new_owner == current_owner:
                        continue
                    
                    new_allocation = {k: set(v) for k, v in allocation.items()}
                    new_allocation[current_owner].remove(item)
                    if new_owner not in new_allocation:
                        new_allocation[new_owner] = set()
                    new_allocation[new_owner].add(item)
                    
                    new_value = sum([
                        bids.get(buyer_id, {}).get(frozenset(bundle), 0.0)
                        for buyer_id, bundle in new_allocation.items()
                        if bundle
                    ])
                    
                    if new_value > current_value + 1e-6:
                        allocation = new_allocation
                        improved = True
                        break
                
                if improved:
                    break
        
        final_value = sum([
            bids.get(buyer_id, {}).get(frozenset(bundle), 0.0)
            for buyer_id, bundle in allocation.items()
            if bundle
        ])
        
        return allocation, final_value
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        self.current_buyer = 0
        self.bids = {}
        self.current_bundle = set(range(min(3, self.n_items)))
        
        obs = self._get_observation(self.current_buyer, self.current_bundle)
        info = {}
        
        return obs, info
    
    def _get_observation(self, buyer_id: int, bundle: Set[int]) -> np.ndarray:
        """Get observation for buyer"""
        bundle_encoding = np.zeros(self.n_items, dtype=np.float32)
        for item in bundle:
            bundle_encoding[item] = 1.0
        
        valuation = self.get_bundle_value(buyer_id, bundle)
        
        current_price = 0.0
        if buyer_id in self.bids:
            current_price = self.bids[buyer_id].get(frozenset(bundle), 0.0)
        
        obs = np.concatenate([
            bundle_encoding,
            [valuation],
            [current_price]
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute step"""
        bid = float(action[0])
        
        if self.current_buyer not in self.bids:
            self.bids[self.current_buyer] = {}
        self.bids[self.current_buyer][frozenset(self.current_bundle)] = bid
        
        self.current_buyer += 1
        
        if self.current_buyer >= self.n_buyers:
            terminated = True
            
            allocation, total_value = self.winner_determination_1UNT(self.bids)
            
            reward = 0.0
            for buyer_id, bundle in allocation.items():
                if buyer_id == 0:
                    valuation = self.get_bundle_value(buyer_id, bundle)
                    payment = self.bids[buyer_id].get(frozenset(bundle), 0.0)
                    reward = valuation - payment
            
            info = {
                'allocation': allocation,
                'total_value': total_value,
                'is_1UNT': self.check_1_UNT_condition(allocation)
            }
            
            obs = np.zeros_like(self._get_observation(0, set()))
        else:
            obs = self._get_observation(self.current_buyer, self.current_bundle)
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

class TruthfulBidder:
    """Baseline bidder that bids truthfully"""
    
    def __init__(self, buyer_id: int):
        self.buyer_id = buyer_id
    
    def get_bid(self, valuation: float) -> float:
        """Bid exactly the valuation"""
        return valuation


class ShadedBidder:
    """Strategic bidder that shades bid below valuation"""
    
    def __init__(self, buyer_id: int, shade_factor: float = 0.9):
        self.buyer_id = buyer_id
        self.shade_factor = shade_factor
    
    def get_bid(self, valuation: float) -> float:
        """Bid a fraction of valuation"""
        return valuation * self.shade_factor


class RLBidder:
    """RL-based bidder using PPO/SAC/DQN"""
    
    def __init__(
        self,
        buyer_id: int,
        observation_space,
        action_space,
        algorithm: str = 'PPO',
        **kwargs
    ):
        self.buyer_id = buyer_id
        self.observation_space = observation_space
        self.action_space = action_space
        self.algorithm = algorithm
        
        self.model = None
        self._init_model(**kwargs)
    
    def _init_model(self, **kwargs):
        """Initialize RL model"""
        env = CombinatorialAuctionEnv(**kwargs.get('env_kwargs', {}))
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
        elif self.algorithm == 'DQN':
            self.model = DQN(
                'MlpPolicy',
                env,
                learning_rate=kwargs.get('learning_rate', 1e-3),
                buffer_size=kwargs.get('buffer_size', 50000),
                verbose=0
            )
    
    def train(self, total_timesteps: int = 50000):
        """Train the RL agent"""
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    def get_bid(self, observation: np.ndarray) -> float:
        """Get bid from RL policy"""
        action, _ = self.model.predict(observation, deterministic=True)
        return float(action[0])
    
    def save(self, path: str):
        """Save model"""
        self.model.save(path)
    
    def load(self, path: str):
        """Load model"""
        if self.algorithm == 'PPO':
            self.model = PPO.load(path)
        elif self.algorithm == 'SAC':
            self.model = SAC.load(path)
        elif self.algorithm == 'DQN':
            self.model = DQN.load(path)


class MultiAgentBiddingSystem:
    """System for managing multiple bidders"""
    
    def __init__(
        self,
        n_buyers: int,
        n_items: int,
        bidder_types: List[str] = None
    ):
        self.n_buyers = n_buyers
        self.n_items = n_items
        
        if bidder_types is None:
            bidder_types = ['truthful'] * n_buyers
        
        self.bidders = []
        self._create_bidders(bidder_types)
    
    def _create_bidders(self, bidder_types: List[str]):
        """Create bidders based on types"""
        env = CombinatorialAuctionEnv(n_items=self.n_items, n_buyers=self.n_buyers)
        
        for buyer_id, btype in enumerate(bidder_types):
            if btype == 'truthful':
                bidder = TruthfulBidder(buyer_id)
            elif btype == 'shaded':
                bidder = ShadedBidder(buyer_id, shade_factor=0.8)
            elif btype in ['PPO', 'SAC', 'DQN']:
                bidder = RLBidder(
                    buyer_id,
                    env.observation_space,
                    env.action_space,
                    algorithm=btype,
                    env_kwargs={'n_items': self.n_items, 'n_buyers': self.n_buyers}
                )
            else:
                raise ValueError(f"Unknown bidder type: {btype}")
            
            self.bidders.append(bidder)
    
    def collect_bids(
        self,
        env,
        bundles: List[Set[int]]
    ) -> Dict[int, Dict[Set[int], float]]:
        """Collect bids from all bidders for given bundles"""
        all_bids = {}
        
        for buyer_id, bidder in enumerate(self.bidders):
            buyer_bids = {}
            
            for bundle in bundles:
                obs = env._get_observation(buyer_id, bundle)
                valuation = env.get_bundle_value(buyer_id, bundle)
                
                if isinstance(bidder, (TruthfulBidder, ShadedBidder)):
                    bid = bidder.get_bid(valuation)
                else:
                    bid = bidder.get_bid(obs)
                
                buyer_bids[frozenset(bundle)] = max(0.0, bid)
            
            all_bids[buyer_id] = buyer_bids
        
        return all_bids
    
    def train_rl_bidders(self, total_timesteps: int = 50000):
        """Train all RL bidders"""
        for bidder in self.bidders:
            if isinstance(bidder, RLBidder):
                print(f"Training RL bidder {bidder.buyer_id}...")
                bidder.train(total_timesteps)
    
    def save_models(self, save_dir: str = './models'):
        """Save all RL models"""
        os.makedirs(save_dir, exist_ok=True)
        
        for bidder in self.bidders:
            if isinstance(bidder, RLBidder):
                path = os.path.join(save_dir, f'bidder_{bidder.buyer_id}_{bidder.algorithm}')
                bidder.save(path)
                print(f"Saved model: {path}")


def generate_all_bundles(n_items: int, max_size: int = None) -> List[Set[int]]:
    """Generate all possible bundles"""
    if max_size is None:
        max_size = n_items
    
    bundles = []
    for size in range(1, min(max_size, n_items) + 1):
        for combo in combinations(range(n_items), size):
            bundles.append(set(combo))
    
    return bundles


def generate_random_bundles(
    n_items: int,
    n_bundles: int = 50,
    min_size: int = 1,
    max_size: int = 5
) -> List[Set[int]]:
    """Generate random bundles"""
    bundles = []
    
    for _ in range(n_bundles):
        size = np.random.randint(min_size, min(max_size, n_items) + 1)
        items = np.random.choice(n_items, size, replace=False)
        bundles.append(set(items))
    
    return bundles


class AuctionSimulator:
    """Simulate combinatorial auctions"""
    
    def __init__(
        self,
        n_items: int = 10,
        n_buyers: int = 3,
        max_bundle_size: int = 5
    ):
        self.env = CombinatorialAuctionEnv(
            n_items=n_items,
            n_buyers=n_buyers,
            max_bundle_size=max_bundle_size
        )
        
        self.n_items = n_items
        self.n_buyers = n_buyers
        self.max_bundle_size = max_bundle_size
    
    def run_auction(
        self,
        bidding_system: MultiAgentBiddingSystem,
        algorithm: str = '1-UNT'
    ) -> Dict:
        """Run a complete auction"""
        if self.n_items <= 8:
            bundles = generate_all_bundles(self.n_items, self.max_bundle_size)
        else:
            bundles = generate_random_bundles(
                self.n_items,
                n_bundles=min(100, 2**self.n_items),
                max_size=self.max_bundle_size
            )
        
        all_bids = bidding_system.collect_bids(self.env, bundles)
        
        if algorithm == 'greedy':
            allocation, total_value = self.env.winner_determination_greedy(all_bids)
        elif algorithm == '1-UNT':
            allocation, total_value = self.env.winner_determination_1UNT(all_bids)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        buyer_utilities = {}
        buyer_payments = {}
        buyer_valuations = {}
        
        for buyer_id in range(self.n_buyers):
            if buyer_id in allocation:
                bundle = allocation[buyer_id]
                valuation = self.env.get_bundle_value(buyer_id, bundle)
                payment = all_bids[buyer_id].get(frozenset(bundle), 0.0)
                utility = valuation - payment
            else:
                valuation = 0.0
                payment = 0.0
                utility = 0.0
            
            buyer_valuations[buyer_id] = valuation
            buyer_payments[buyer_id] = payment
            buyer_utilities[buyer_id] = utility
        
        is_1UNT = self.env.check_1_UNT_condition(allocation)
        
        optimal_welfare = self._compute_optimal_welfare()
        achieved_welfare = sum(buyer_valuations.values())
        efficiency = (achieved_welfare / optimal_welfare * 100) if optimal_welfare > 0 else 0
        
        return {
            'allocation': allocation,
            'total_bid_value': total_value,
            'buyer_valuations': buyer_valuations,
            'buyer_payments': buyer_payments,
            'buyer_utilities': buyer_utilities,
            'total_welfare': achieved_welfare,
            'optimal_welfare': optimal_welfare,
            'efficiency': efficiency,
            'is_1UNT': is_1UNT
        }
    
    def _compute_optimal_welfare(self) -> float:
        """Compute optimal social welfare"""
        optimal = 0.0
        
        for buyer_id in range(self.n_buyers):
            for bundle_set, value in self.env.buyer_valuations[buyer_id].items():
                if len(bundle_set) <= self.max_bundle_size:
                    optimal = max(optimal, value)
        
        return optimal


# ============================================================================
# SECTION 3: EVALUATION
# ============================================================================

class AuctionEvaluator:
    """Evaluate and compare different strategies and algorithms"""
    
    def __init__(
        self,
        n_items: int = 10,
        n_buyers: int = 3,
        max_bundle_size: int = 5
    ):
        self.n_items = n_items
        self.n_buyers = n_buyers
        self.max_bundle_size = max_bundle_size
        
        self.simulator = AuctionSimulator(
            n_items=n_items,
            n_buyers=n_buyers,
            max_bundle_size=max_bundle_size
        )
    
    def evaluate_strategy(
        self,
        bidder_types: List[str],
        algorithm: str = '1-UNT',
        n_episodes: int = 100,
        strategy_name: str = 'Unknown'
    ) -> Dict:
        """Evaluate a bidding strategy over multiple episodes"""
        results = {
            'efficiencies': [],
            'total_welfares': [],
            'buyer_utilities': [[] for _ in range(self.n_buyers)],
            'is_1UNT': [],
            'allocations': []
        }
        
        for episode in range(n_episodes):
            system = MultiAgentBiddingSystem(
                n_buyers=self.n_buyers,
                n_items=self.n_items,
                bidder_types=bidder_types
            )
            
            result = self.simulator.run_auction(system, algorithm=algorithm)
            
            results['efficiencies'].append(result['efficiency'])
            results['total_welfares'].append(result['total_welfare'])
            results['is_1UNT'].append(result['is_1UNT'])
            results['allocations'].append(result['allocation'])
            
            for buyer_id in range(self.n_buyers):
                results['buyer_utilities'][buyer_id].append(
                    result['buyer_utilities'].get(buyer_id, 0.0)
                )
        
        stats = {
            'strategy_name': strategy_name,
            'algorithm': algorithm,
            'mean_efficiency': np.mean(results['efficiencies']),
            'std_efficiency': np.std(results['efficiencies']),
            'mean_welfare': np.mean(results['total_welfares']),
            'std_welfare': np.std(results['total_welfares']),
            'fraction_1UNT': np.mean(results['is_1UNT']),
            'mean_buyer_utilities': [
                np.mean(results['buyer_utilities'][i])
                for i in range(self.n_buyers)
            ],
            'raw_results': results
        }
        
        return stats
    
    def compare_strategies(
        self,
        strategies: Dict[str, List[str]],
        algorithm: str = '1-UNT',
        n_episodes: int = 100
    ) -> pd.DataFrame:
        """Compare multiple bidding strategies"""
        all_stats = []
        
        for strategy_name, bidder_types in strategies.items():
            print(f"\nEvaluating strategy: {strategy_name}")
            stats = self.evaluate_strategy(
                bidder_types,
                algorithm=algorithm,
                n_episodes=n_episodes,
                strategy_name=strategy_name
            )
            all_stats.append(stats)
        
        comparison_data = []
        for stats in all_stats:
            row = {
                'Strategy': stats['strategy_name'],
                'Algorithm': stats['algorithm'],
                'Mean Efficiency (%)': stats['mean_efficiency'],
                'Std Efficiency': stats['std_efficiency'],
                'Mean Welfare': stats['mean_welfare'],
                'Fraction 1-UNT': stats['fraction_1UNT']
            }
            
            for buyer_id in range(self.n_buyers):
                row[f'Buyer {buyer_id} Utility'] = stats['mean_buyer_utilities'][buyer_id]
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        self.detailed_stats = all_stats
        
        return df
    
    def compare_algorithms(
        self,
        bidder_types: List[str],
        algorithms: List[str] = ['greedy', '1-UNT'],
        n_episodes: int = 100,
        strategy_name: str = 'Test Strategy'
    ) -> pd.DataFrame:
        """Compare different winner determination algorithms"""
        all_stats = []
        
        for algorithm in algorithms:
            print(f"\nEvaluating algorithm: {algorithm}")
            stats = self.evaluate_strategy(
                bidder_types,
                algorithm=algorithm,
                n_episodes=n_episodes,
                strategy_name=f"{strategy_name} + {algorithm}"
            )
            all_stats.append(stats)
        
        comparison_data = []
        for stats in all_stats:
            row = {
                'Algorithm': stats['algorithm'],
                'Mean Efficiency (%)': stats['mean_efficiency'],
                'Mean Welfare': stats['mean_welfare'],
                'Fraction 1-UNT': stats['fraction_1UNT']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        return df
    
    def plot_comparison(self, save_path: str = None):
        """Create visualization comparing strategies"""
        if not hasattr(self, 'detailed_stats'):
            print("No comparison data available. Run compare_strategies first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Combinatorial Auction Strategy Comparison', fontsize=16)
        
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
        
        # 2. Welfare
        ax = axes[0, 1]
        welfares = [s['mean_welfare'] for s in self.detailed_stats]
        welfare_stds = [s['std_welfare'] for s in self.detailed_stats]
        ax.bar(strategies, welfares, yerr=welfare_stds, capsize=5, color='green')
        ax.set_ylabel('Total Welfare')
        ax.set_title('Mean Total Welfare')
        ax.tick_params(axis='x', rotation=45)
        
        # 3. 1-UNT Satisfaction
        ax = axes[0, 2]
        fractions = [s['fraction_1UNT'] for s in self.detailed_stats]
        ax.bar(strategies, fractions, color='coral')
        ax.set_ylabel('Fraction')
        ax.set_title('Fraction Satisfying 1-UNT')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim([0, 1.05])
        
        # 4. Buyer Utilities
        ax = axes[1, 0]
        x = np.arange(len(strategies))
        width = 0.25
        utilities = np.array([s['mean_buyer_utilities'] for s in self.detailed_stats])
        
        for buyer_id in range(self.n_buyers):
            ax.bar(x + buyer_id*width, utilities[:, buyer_id], width, 
                  label=f'Buyer {buyer_id}')
        
        ax.set_ylabel('Utility')
        ax.set_title('Mean Buyer Utilities')
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
        
        # 6. Welfare Distribution
        ax = axes[1, 2]
        for stats in self.detailed_stats:
            ax.hist(stats['raw_results']['total_welfares'], bins=20, alpha=0.5,
                   label=stats['strategy_name'])
        ax.set_xlabel('Total Welfare')
        ax.set_ylabel('Frequency')
        ax.set_title('Welfare Distribution')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def verify_paper_claims(self, n_trials: int = 100) -> Dict:
        """Verify key claims from the paper"""
        print("\nVerifying Paper Claims...")
        print("="*60)
        
        results = {}
        
        # Test 1-UNT algorithm
        print("\nTest 1: 1-UNT Algorithm Performance")
        truthful_types = ['truthful'] * self.n_buyers
        
        stats_1unt = self.evaluate_strategy(
            truthful_types,
            algorithm='1-UNT',
            n_episodes=n_trials,
            strategy_name='1-UNT'
        )
        
        results['1-UNT'] = stats_1unt
        
        print(f"Mean Efficiency: {stats_1unt['mean_efficiency']:.2f}%")
        print(f"Fraction satisfying 1-UNT: {stats_1unt['fraction_1UNT']:.3f}")
        
        # Test Greedy algorithm
        print("\nTest 2: Greedy Algorithm Performance")
        
        stats_greedy = self.evaluate_strategy(
            truthful_types,
            algorithm='greedy',
            n_episodes=n_trials,
            strategy_name='Greedy'
        )
        
        results['greedy'] = stats_greedy
        
        print(f"Mean Efficiency: {stats_greedy['mean_efficiency']:.2f}%")
        print(f"Fraction satisfying 1-UNT: {stats_greedy['fraction_1UNT']:.3f}")
        
        # Comparison
        print("\n" + "="*60)
        print("Verification Summary:")
        
        if stats_1unt['mean_efficiency'] >= 50:
            print("✓ VERIFIED: 1-UNT achieves at least 50% efficiency")
        else:
            print("✗ NOT VERIFIED: 1-UNT efficiency below 50%")
        
        if stats_1unt['mean_efficiency'] > stats_greedy['mean_efficiency']:
            print("✓ VERIFIED: 1-UNT outperforms Greedy on average")
        else:
            print("✗ NOT VERIFIED: Greedy performs as well or better")
        
        print("="*60)
        
        return results


def create_comparison_report(
    evaluator: AuctionEvaluator,
    strategies: Dict[str, List[str]],
    n_episodes: int = 100,
    output_dir: str = './results'
):
    """Create comprehensive comparison report"""
    os.makedirs(output_dir, exist_ok=True)
    
    df = evaluator.compare_strategies(strategies, n_episodes=n_episodes)
    
    csv_path = os.path.join(output_dir, 'strategy_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    plot_path = os.path.join(output_dir, 'strategy_comparison.png')
    evaluator.plot_comparison(save_path=plot_path)
    
    verification = evaluator.verify_paper_claims(n_trials=n_episodes)
    
    report_path = os.path.join(output_dir, 'verification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Combinatorial Auction Verification Report\n")
        f.write("="*60 + "\n\n")
        
        f.write("Strategy Comparison:\n")
        f.write(df.to_string())
        f.write("\n\n")
        
        f.write("Paper Claims Verification:\n")
        for alg, stats in verification.items():
            f.write(f"\n{alg} Algorithm:\n")
            f.write(f"  Mean Efficiency: {stats['mean_efficiency']:.2f}%\n")
            f.write(f"  Fraction 1-UNT: {stats['fraction_1UNT']:.3f}\n")
        f.write("\n")
    
    print(f"Report saved to {report_path}")
    
    return df, verification


# ============================================================================
# SECTION 4: MAIN EXPERIMENT PIPELINE
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Combinatorial Auction RL Experiments'
    )
    
    parser.add_argument('--n_items', type=int, default=10)
    parser.add_argument('--n_buyers', type=int, default=3)
    parser.add_argument('--max_bundle_size', type=int, default=5)
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'SAC', 'DQN'])
    parser.add_argument('--total_timesteps', type=int, default=50000)
    parser.add_argument('--n_eval_episodes', type=int, default=100)
    parser.add_argument('--winner_algorithm', type=str, default='1-UNT',
                       choices=['greedy', '1-UNT'])
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'evaluate', 'verify', 'all'])
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def quick_demo():
    """Quick demonstration"""
    print("\n" + "="*80)
    print("QUICK DEMO - Combinatorial Auction")
    print("="*80 + "\n")
    
    env = CombinatorialAuctionEnv(n_items=6, n_buyers=2, max_bundle_size=3)
    
    print("Environment created:")
    print(f"  - {env.n_items} items")
    print(f"  - {env.n_buyers} buyers")
    print(f"  - Max bundle size: {env.max_bundle_size}")
    
    print("\nChecking decreasing marginal utility...")
    for buyer_id in range(env.n_buyers):
        result = env.check_decreasing_marginal_utility(buyer_id)
        print(f"  Buyer {buyer_id}: {'✓ Satisfied' if result else '✗ Violated'}")
    
    simulator = AuctionSimulator(n_items=6, n_buyers=2, max_bundle_size=3)
    
    print("\n" + "="*80)
    print("Testing Strategies:")
    print("="*80)
    
    strategies = [
        (['truthful', 'truthful'], 'Both Truthful'),
        (['shaded', 'shaded'], 'Both Shaded'),
        (['truthful', 'shaded'], 'Mixed')
    ]
    
    for bidder_types, name in strategies:
        system = MultiAgentBiddingSystem(
            n_buyers=2,
            n_items=6,
            bidder_types=bidder_types
        )
        
        result = simulator.run_auction(system, algorithm='1-UNT')
        
        print(f"\n{name}:")
        print(f"  Allocation: {result['allocation']}")
        print(f"  Total welfare: {result['total_welfare']:.2f}")
        print(f"  Efficiency: {result['efficiency']:.1f}%")
        print(f"  Satisfies 1-UNT: {result['is_1UNT']}")
    
    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80)


def run_experiment(args):
    """Run complete experiment pipeline"""
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("COMBINATORIAL AUCTION RL EXPERIMENTS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Items: {args.n_items}")
    print(f"  Buyers: {args.n_buyers}")
    print(f"  Max bundle size: {args.max_bundle_size}")
    print(f"  RL Algorithm: {args.algorithm}")
    print(f"  Winner algorithm: {args.winner_algorithm}")
    print(f"  Evaluation episodes: {args.n_eval_episodes}")
    
    evaluator = AuctionEvaluator(
        n_items=args.n_items,
        n_buyers=args.n_buyers,
        max_bundle_size=args.max_bundle_size
    )
    
    # Define strategies
    strategies = {
        'All Truthful': ['truthful'] * args.n_buyers,
        'All Shaded': ['shaded'] * args.n_buyers,
    }
    
    if args.n_buyers >= 2:
        mixed = ['truthful'] * (args.n_buyers // 2) + ['shaded'] * (args.n_buyers - args.n_buyers // 2)
        strategies['Mixed Strategy'] = mixed
    
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
        
        env = evaluator.simulator.env
        print("\nProperty: Decreasing Marginal Utility")
        for buyer_id in range(args.n_buyers):
            result = env.check_decreasing_marginal_utility(buyer_id)
            print(f"  Buyer {buyer_id}: {'✓ Satisfied' if result else '✗ Violated'}")
        
        evaluator.verify_paper_claims(n_trials=args.n_eval_episodes)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 1 or '--demo' in sys.argv:
        # Run demo if no arguments
        quick_demo()
    else:
        # Run full experiment
        args = parse_args()
        run_experiment(args)
