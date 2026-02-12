"""
Evaluation and Analysis Module
Compare RL agents with truthful strategy and verify paper's theoretical results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import pandas as pd
from mav_auction_env import MultiAttributeAuctionEnv, TruthfulSellerAgent
from rl_agents import RLSellerAgent


class AuctionEvaluator:
    """
    Evaluate and compare different auction strategies
    """
    
    def __init__(
        self,
        n_sellers: int = 3,
        n_attributes: int = 2,
        max_price: float = 100.0
    ):
        self.n_sellers = n_sellers
        self.n_attributes = n_attributes
        self.max_price = max_price
        
        # Create environment
        self.env = MultiAttributeAuctionEnv(
            n_sellers=n_sellers,
            n_attributes=n_attributes,
            max_price=max_price
        )
    
    def run_auction_episode(
        self,
        agents: List,
        reset_env: bool = True
    ) -> Dict:
        """
        Run a single auction episode with given agents
        
        Args:
            agents: List of agents (one per seller)
            reset_env: Whether to reset environment
        
        Returns:
            Dictionary with auction results
        """
        if reset_env:
            obs, info = self.env.reset()
        
        bids = []
        
        # Collect bids from all sellers
        for seller_id in range(self.n_sellers):
            obs = self.env._get_observation(seller_id)
            action = agents[seller_id].get_action(obs)
            bids.append(float(action[0]))
        
        # Execute auction
        self.env.reset()  # Reset to execute all bids
        
        for bid in bids:
            obs, reward, terminated, truncated, info = self.env.step(np.array([bid]))
        
        return info
    
    def evaluate_strategy(
        self,
        agents: List,
        n_episodes: int = 1000,
        strategy_name: str = 'unknown'
    ) -> Dict:
        """
        Evaluate a strategy over multiple episodes
        
        Returns:
            Dictionary with aggregated statistics
        """
        results = {
            'buyer_utilities': [],
            'seller_utilities': [[] for _ in range(self.n_sellers)],
            'total_utilities': [],
            'transaction_prices': [],
            'winners': [],
            'bids': [[] for _ in range(self.n_sellers)]
        }
        
        for episode in range(n_episodes):
            info = self.run_auction_episode(agents, reset_env=True)
            
            results['buyer_utilities'].append(info.get('buyer_utility', 0))
            results['total_utilities'].append(
                info.get('buyer_utility', 0) + sum(info.get('seller_utilities', [0]))
            )
            results['transaction_prices'].append(info.get('transaction_price', 0))
            results['winners'].append(info.get('winner', -1))
            
            # Store individual seller utilities
            seller_utils = info.get('seller_utilities', np.zeros(self.n_sellers))
            for i in range(self.n_sellers):
                results['seller_utilities'][i].append(seller_utils[i])
            
            # Store bids
            all_bids = info.get('all_bids', np.zeros(self.n_sellers))
            for i in range(self.n_sellers):
                results['bids'][i].append(all_bids[i])
        
        # Compute statistics
        stats = {
            'strategy_name': strategy_name,
            'mean_buyer_utility': np.mean(results['buyer_utilities']),
            'std_buyer_utility': np.std(results['buyer_utilities']),
            'mean_total_utility': np.mean(results['total_utilities']),
            'std_total_utility': np.std(results['total_utilities']),
            'mean_transaction_price': np.mean(results['transaction_prices']),
            'win_rates': [
                np.sum(np.array(results['winners']) == i) / n_episodes 
                for i in range(self.n_sellers)
            ],
            'mean_seller_utilities': [
                np.mean(results['seller_utilities'][i]) 
                for i in range(self.n_sellers)
            ],
            'mean_bids': [
                np.mean(results['bids'][i]) 
                for i in range(self.n_sellers)
            ]
        }
        
        # Store raw results
        stats['raw_results'] = results
        
        return stats
    
    def compare_strategies(
        self,
        strategies: Dict[str, List],
        n_episodes: int = 1000
    ) -> pd.DataFrame:
        """
        Compare multiple strategies
        
        Args:
            strategies: Dict mapping strategy name to list of agents
            n_episodes: Number of episodes to evaluate each strategy
        
        Returns:
            DataFrame with comparison results
        """
        all_stats = []
        
        for strategy_name, agents in strategies.items():
            print(f"\nEvaluating strategy: {strategy_name}")
            stats = self.evaluate_strategy(agents, n_episodes, strategy_name)
            all_stats.append(stats)
        
        # Create comparison DataFrame
        comparison_data = []
        for stats in all_stats:
            row = {
                'Strategy': stats['strategy_name'],
                'Mean Buyer Utility': stats['mean_buyer_utility'],
                'Mean Total Utility': stats['mean_total_utility'],
                'Mean Transaction Price': stats['mean_transaction_price']
            }
            
            for i in range(self.n_sellers):
                row[f'Seller {i} Win Rate'] = stats['win_rates'][i]
                row[f'Seller {i} Mean Utility'] = stats['mean_seller_utilities'][i]
                row[f'Seller {i} Mean Bid'] = stats['mean_bids'][i]
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Store detailed stats
        self.detailed_stats = all_stats
        
        return df
    
    def plot_comparison(self, save_path: str = None):
        """
        Create visualization comparing strategies
        """
        if not hasattr(self, 'detailed_stats'):
            print("No comparison data available. Run compare_strategies first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Strategy Comparison in Multi-Attribute Auction', fontsize=16)
        
        # Extract data
        strategies = [s['strategy_name'] for s in self.detailed_stats]
        
        # 1. Buyer Utility
        ax = axes[0, 0]
        buyer_utils = [s['mean_buyer_utility'] for s in self.detailed_stats]
        buyer_stds = [s['std_buyer_utility'] for s in self.detailed_stats]
        ax.bar(strategies, buyer_utils, yerr=buyer_stds, capsize=5)
        ax.set_ylabel('Utility')
        ax.set_title('Mean Buyer Utility')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. Total Utility
        ax = axes[0, 1]
        total_utils = [s['mean_total_utility'] for s in self.detailed_stats]
        total_stds = [s['std_total_utility'] for s in self.detailed_stats]
        ax.bar(strategies, total_utils, yerr=total_stds, capsize=5, color='green')
        ax.set_ylabel('Utility')
        ax.set_title('Mean Total Utility (Buyer + All Sellers)')
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Transaction Price
        ax = axes[0, 2]
        prices = [s['mean_transaction_price'] for s in self.detailed_stats]
        ax.bar(strategies, prices, color='orange')
        ax.set_ylabel('Price')
        ax.set_title('Mean Transaction Price')
        ax.tick_params(axis='x', rotation=45)
        
        # 4. Win Rates
        ax = axes[1, 0]
        win_rates = np.array([s['win_rates'] for s in self.detailed_stats])
        x = np.arange(len(strategies))
        width = 0.25
        for i in range(self.n_sellers):
            ax.bar(x + i*width, win_rates[:, i], width, label=f'Seller {i}')
        ax.set_ylabel('Win Rate')
        ax.set_title('Seller Win Rates')
        ax.set_xticks(x + width)
        ax.set_xticklabels(strategies, rotation=45)
        ax.legend()
        
        # 5. Seller Utilities
        ax = axes[1, 1]
        seller_utils = np.array([s['mean_seller_utilities'] for s in self.detailed_stats])
        for i in range(self.n_sellers):
            ax.bar(x + i*width, seller_utils[:, i], width, label=f'Seller {i}')
        ax.set_ylabel('Utility')
        ax.set_title('Mean Seller Utilities')
        ax.set_xticks(x + width)
        ax.set_xticklabels(strategies, rotation=45)
        ax.legend()
        
        # 6. Bid Distribution
        ax = axes[1, 2]
        for stats in self.detailed_stats:
            all_bids = []
            for i in range(self.n_sellers):
                all_bids.extend(stats['raw_results']['bids'][i])
            ax.hist(all_bids, bins=30, alpha=0.5, label=stats['strategy_name'])
        ax.set_xlabel('Bid Amount')
        ax.set_ylabel('Frequency')
        ax.set_title('Bid Distribution')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def verify_incentive_compatibility(self, n_trials: int = 100) -> Dict:
        """
        Verify that truthful bidding is incentive compatible
        
        Test: Compare truthful strategy vs deviating strategies
        """
        print("\nVerifying Incentive Compatibility...")
        print("="*60)
        
        results = {}
        
        # Baseline: All sellers use truthful strategy
        truthful_agents = [
            TruthfulSellerAgent(i, self.env.observation_space, self.env.action_space)
            for i in range(self.n_sellers)
        ]
        
        baseline_stats = self.evaluate_strategy(
            truthful_agents, 
            n_episodes=n_trials,
            strategy_name='All Truthful'
        )
        
        results['all_truthful'] = baseline_stats
        
        # Test: One seller deviates (overbids)
        print("\nTesting deviation: Seller 0 overbids by 20%...")
        
        class OverbiddingAgent:
            def __init__(self, seller_id, multiplier=1.2):
                self.seller_id = seller_id
                self.multiplier = multiplier
                self.truthful = TruthfulSellerAgent(seller_id, None, None)
            
            def get_action(self, obs):
                truthful_bid = self.truthful.get_action(obs)
                return truthful_bid * self.multiplier
        
        deviating_agents = truthful_agents.copy()
        deviating_agents[0] = OverbiddingAgent(0, multiplier=1.2)
        
        deviation_stats = self.evaluate_strategy(
            deviating_agents,
            n_episodes=n_trials,
            strategy_name='Seller 0 Overbids'
        )
        
        results['seller_0_overbids'] = deviation_stats
        
        # Compare utilities for seller 0
        print("\nResults:")
        print(f"Seller 0 utility (truthful): {baseline_stats['mean_seller_utilities'][0]:.2f}")
        print(f"Seller 0 utility (overbid):  {deviation_stats['mean_seller_utilities'][0]:.2f}")
        print(f"Seller 0 win rate (truthful): {baseline_stats['win_rates'][0]:.3f}")
        print(f"Seller 0 win rate (overbid):  {deviation_stats['win_rates'][0]:.3f}")
        
        # Test: One seller deviates (underbids)
        print("\nTesting deviation: Seller 0 underbids by 20%...")
        
        class UnderbiddingAgent:
            def __init__(self, seller_id, multiplier=0.8):
                self.seller_id = seller_id
                self.multiplier = multiplier
                self.truthful = TruthfulSellerAgent(seller_id, None, None)
            
            def get_action(self, obs):
                truthful_bid = self.truthful.get_action(obs)
                return truthful_bid * self.multiplier
        
        deviating_agents = truthful_agents.copy()
        deviating_agents[0] = UnderbiddingAgent(0, multiplier=0.8)
        
        underbid_stats = self.evaluate_strategy(
            deviating_agents,
            n_episodes=n_trials,
            strategy_name='Seller 0 Underbids'
        )
        
        results['seller_0_underbids'] = underbid_stats
        
        print(f"Seller 0 utility (underbid): {underbid_stats['mean_seller_utilities'][0]:.2f}")
        print(f"Seller 0 win rate (underbid): {underbid_stats['win_rates'][0]:.3f}")
        
        print("\n" + "="*60)
        print("Incentive Compatibility Verification:")
        
        if (baseline_stats['mean_seller_utilities'][0] >= deviation_stats['mean_seller_utilities'][0] and
            baseline_stats['mean_seller_utilities'][0] >= underbid_stats['mean_seller_utilities'][0]):
            print("✓ VERIFIED: Truthful bidding maximizes utility")
            print("  Deviations lead to lower or equal utility")
        else:
            print("✗ NOT VERIFIED: Deviations can increase utility")
        
        print("="*60)
        
        return results
    
    def verify_efficiency(self, n_trials: int = 100) -> Dict:
        """
        Verify that truthful bidding achieves maximum total utility
        as stated in the paper
        """
        print("\nVerifying Efficiency (Total Utility Maximization)...")
        print("="*60)
        
        # Compute theoretical maximum
        max_theoretical_utility = 0
        for i in range(self.n_sellers):
            D_i, _ = self.env.compute_D_i(i)
            max_theoretical_utility = max(max_theoretical_utility, D_i)
        
        print(f"Theoretical maximum total utility: {max_theoretical_utility:.2f}")
        
        # Test truthful strategy
        truthful_agents = [
            TruthfulSellerAgent(i, self.env.observation_space, self.env.action_space)
            for i in range(self.n_sellers)
        ]
        
        stats = self.evaluate_strategy(
            truthful_agents,
            n_episodes=n_trials,
            strategy_name='Truthful'
        )
        
        achieved_utility = stats['mean_total_utility']
        efficiency = (achieved_utility / max_theoretical_utility) * 100 if max_theoretical_utility > 0 else 0
        
        print(f"Achieved total utility: {achieved_utility:.2f}")
        print(f"Efficiency: {efficiency:.1f}%")
        
        if efficiency >= 95:
            print("✓ VERIFIED: Truthful bidding achieves near-optimal efficiency")
        else:
            print("⚠ WARNING: Efficiency lower than expected")
        
        print("="*60)
        
        return {
            'theoretical_max': max_theoretical_utility,
            'achieved': achieved_utility,
            'efficiency': efficiency,
            'stats': stats
        }


def create_comparison_report(
    evaluator: AuctionEvaluator,
    strategies: Dict[str, List],
    n_episodes: int = 1000,
    output_dir: str = './results'
):
    """
    Create comprehensive comparison report
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Run comparison
    df = evaluator.compare_strategies(strategies, n_episodes)
    
    # Save results
    csv_path = os.path.join(output_dir, 'strategy_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Create visualization
    plot_path = os.path.join(output_dir, 'strategy_comparison.png')
    evaluator.plot_comparison(save_path=plot_path)
    
    # Verify properties
    ic_results = evaluator.verify_incentive_compatibility(n_trials=n_episodes)
    efficiency_results = evaluator.verify_efficiency(n_trials=n_episodes)
    
    # Save verification results
    report_path = os.path.join(output_dir, 'verification_report.txt')
    with open(report_path, 'w') as f:
        f.write("MAV Auction Verification Report\n")
        f.write("="*60 + "\n\n")
        
        f.write("Strategy Comparison:\n")
        f.write(df.to_string())
        f.write("\n\n")
        
        f.write("Efficiency Verification:\n")
        f.write(f"Theoretical Maximum: {efficiency_results['theoretical_max']:.2f}\n")
        f.write(f"Achieved: {efficiency_results['achieved']:.2f}\n")
        f.write(f"Efficiency: {efficiency_results['efficiency']:.1f}%\n")
        f.write("\n")
    
    print(f"Report saved to {report_path}")
    
    return df, ic_results, efficiency_results
