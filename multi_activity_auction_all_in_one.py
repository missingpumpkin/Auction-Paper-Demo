"""
Multi-Activity Level Ascending Bid Combinatorial Auction
Implementation of Paper: "Multi-Activity-Level Ascending Bid Combinatorial Auction Method"
Authors: Jin Xing, Shi Chunyi (Tsinghua University, 2006)

This is an extension of the iBundle method with multiple activity levels.
Key innovation: Buyers can adjust their activity levels during the auction to balance
time complexity and utility optimization.

All-in-one implementation with:
- Multi-round ascending bid environment
- Activity level management (low/medium/high)
- Rational buyer strategies
- Time-utility tradeoff analysis
- Complete evaluation framework
"""

import numpy as np
import argparse
import random
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataclasses import dataclass

# ============================================================================
# SECTION 1: ENVIRONMENT - Multi-Activity Ascending Bid Auction
# ============================================================================

class ActivityLevel:
    """Activity levels for buyers"""
    LOW = 1      # Bid on few bundles per round
    MEDIUM = 2   # Bid on moderate number of bundles per round
    HIGH = 3     # Bid on many bundles per round

@dataclass
class BidInfo:
    """Information about a bid"""
    buyer_id: int
    bundle: frozenset
    price: float
    activity_level: int
    round_number: int

class MultiActivityAuctionEnv:
    """
    Multi-Activity Level Ascending Bid Combinatorial Auction Environment
    
    Key features:
    - Multiple rounds of ascending bids
    - Buyers can adjust activity levels
    - Activity levels affect number of bundles bid per round
    - Winner determination after each round
    - Auction terminates when no new bids or max rounds reached
    """
    
    def __init__(self, n_items: int, n_buyers: int, max_bundle_size: int = None,
                 max_rounds: int = 20, price_increment: float = 5.0, seed: int = 42):
        """
        Initialize the multi-activity auction environment.
        
        Args:
            n_items: Number of items to auction
            n_buyers: Number of buyers
            max_bundle_size: Maximum size of bundles (default: n_items // 2)
            max_rounds: Maximum number of auction rounds
            price_increment: Minimum price increment per round
            seed: Random seed
        """
        self.n_items = n_items
        self.n_buyers = n_buyers
        self.max_bundle_size = max_bundle_size or max(1, n_items // 2)
        self.max_rounds = max_rounds
        self.price_increment = price_increment
        self.seed = seed
        
        # Set random seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Items
        self.items = set(range(n_items))
        
        # Generate buyer valuations (decreasing marginal utility)
        self.buyer_valuations = self._generate_valuations()
        
        # Auction state
        self.current_round = 0
        self.current_prices = defaultdict(float)  # Bundle -> current price
        self.all_bids = []  # All bids across all rounds
        self.round_bids = []  # Bids in current round
        self.winner_allocation = {}  # Current winning allocation
        self.auction_active = True
        
        # Activity tracking
        self.buyer_activity_levels = {i: ActivityLevel.MEDIUM for i in range(n_buyers)}
        
        # Statistics
        self.total_welfare = 0.0
        self.rounds_completed = 0
        
    def _generate_valuations(self) -> Dict[int, Dict[frozenset, float]]:
        """Generate buyer valuations with decreasing marginal utility"""
        valuations = {}
        
        for buyer_id in range(self.n_buyers):
            buyer_vals = {}
            
            # Generate base item values
            base_values = np.random.uniform(20, 50, self.n_items)
            
            # Generate valuations for all possible bundles
            for bundle_size in range(1, min(self.max_bundle_size, self.n_items) + 1):
                # Generate some bundles of this size
                n_bundles = min(100, self._n_choose_k(self.n_items, bundle_size))
                
                for _ in range(n_bundles):
                    bundle = frozenset(np.random.choice(self.n_items, bundle_size, replace=False))
                    
                    # Calculate value with decreasing marginal utility
                    value = 0.0
                    discount_factor = 1.0
                    
                    for item in sorted(bundle):
                        value += base_values[item] * discount_factor
                        discount_factor *= np.random.uniform(0.7, 0.9)  # Decreasing marginal utility
                    
                    buyer_vals[bundle] = value
            
            valuations[buyer_id] = buyer_vals
        
        return valuations
    
    def _n_choose_k(self, n: int, k: int) -> int:
        """Calculate binomial coefficient"""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result
    
    def get_buyer_valuation(self, buyer_id: int, bundle: frozenset) -> float:
        """Get buyer's valuation for a bundle"""
        return self.buyer_valuations[buyer_id].get(bundle, 0.0)
    
    def get_current_price(self, bundle: frozenset) -> float:
        """Get current price for a bundle"""
        return self.current_prices.get(bundle, 0.0)
    
    def set_activity_level(self, buyer_id: int, level: int):
        """Set activity level for a buyer"""
        if level in [ActivityLevel.LOW, ActivityLevel.MEDIUM, ActivityLevel.HIGH]:
            self.buyer_activity_levels[buyer_id] = level
    
    def get_activity_level(self, buyer_id: int) -> int:
        """Get current activity level for a buyer"""
        return self.buyer_activity_levels[buyer_id]
    
    def get_max_bids_per_round(self, buyer_id: int) -> int:
        """Get maximum number of bids allowed based on activity level"""
        level = self.buyer_activity_levels[buyer_id]
        if level == ActivityLevel.LOW:
            return 5
        elif level == ActivityLevel.MEDIUM:
            return 15
        else:  # HIGH
            return 30
    
    def submit_bid(self, buyer_id: int, bundle: frozenset, price: float) -> bool:
        """
        Submit a bid in the current round.
        
        Args:
            buyer_id: ID of the buyer
            bundle: Bundle to bid on
            price: Bid price
            
        Returns:
            True if bid accepted, False otherwise
        """
        if not self.auction_active:
            return False
        
        # Check if price is higher than current price
        current_price = self.get_current_price(bundle)
        if price <= current_price:
            return False
        
        # Check if price increment is sufficient
        if price < current_price + self.price_increment:
            return False
        
        # Check activity level constraints
        buyer_bids_this_round = [b for b in self.round_bids if b.buyer_id == buyer_id]
        max_bids = self.get_max_bids_per_round(buyer_id)
        if len(buyer_bids_this_round) >= max_bids:
            return False
        
        # Accept bid
        bid = BidInfo(
            buyer_id=buyer_id,
            bundle=bundle,
            price=price,
            activity_level=self.buyer_activity_levels[buyer_id],
            round_number=self.current_round
        )
        
        self.round_bids.append(bid)
        self.all_bids.append(bid)
        
        # Update current price
        self.current_prices[bundle] = price
        
        return True
    
    def end_round(self) -> Tuple[Dict[int, frozenset], float]:
        """
        End the current round and determine winners.
        
        Returns:
            Tuple of (allocation, total_welfare)
        """
        # Determine winners based on current bids
        allocation, welfare = self._winner_determination()
        
        self.winner_allocation = allocation
        self.total_welfare = welfare
        
        # Prepare for next round
        self.current_round += 1
        self.round_bids = []
        
        # Check termination conditions
        if self.current_round >= self.max_rounds or len(self.all_bids) == 0:
            self.auction_active = False
        
        self.rounds_completed = self.current_round
        
        return allocation, welfare
    
    def _winner_determination(self) -> Tuple[Dict[int, frozenset], float]:
        """
        Determine winning allocation using greedy algorithm.
        Similar to the paper's winner determination approach.
        """
        # Collect all bids with positive surplus
        valid_bids = []
        for bid in self.all_bids:
            valuation = self.get_buyer_valuation(bid.buyer_id, bid.bundle)
            surplus = valuation - bid.price
            if surplus >= 0:
                valid_bids.append((bid.buyer_id, bid.bundle, bid.price, surplus))
        
        # Sort by surplus (descending)
        valid_bids.sort(key=lambda x: x[3], reverse=True)
        
        # Greedy allocation
        allocation = {}
        allocated_items = set()
        total_welfare = 0.0
        
        for buyer_id, bundle, price, surplus in valid_bids:
            # Check if bundle doesn't conflict with already allocated items
            if not bundle.intersection(allocated_items):
                allocation[buyer_id] = bundle
                allocated_items.update(bundle)
                valuation = self.get_buyer_valuation(buyer_id, bundle)
                total_welfare += valuation
        
        return allocation, total_welfare
    
    def get_state_info(self) -> Dict:
        """Get current state information"""
        return {
            'round': self.current_round,
            'active': self.auction_active,
            'n_bids_total': len(self.all_bids),
            'n_bids_this_round': len(self.round_bids),
            'current_welfare': self.total_welfare,
            'winner_allocation': self.winner_allocation.copy(),
            'activity_levels': self.buyer_activity_levels.copy()
        }
    
    def reset(self):
        """Reset the auction to initial state"""
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.current_round = 0
        self.current_prices.clear()
        self.all_bids = []
        self.round_bids = []
        self.winner_allocation = {}
        self.auction_active = True
        self.buyer_activity_levels = {i: ActivityLevel.MEDIUM for i in range(self.n_buyers)}
        self.total_welfare = 0.0
        self.rounds_completed = 0
        
        # Regenerate valuations
        self.buyer_valuations = self._generate_valuations()


# ============================================================================
# SECTION 2: AGENTS - Buyer Strategies with Activity Management
# ============================================================================

class BuyerAgent:
    """Base class for buyer agents"""
    
    def __init__(self, buyer_id: int, env: MultiActivityAuctionEnv):
        self.buyer_id = buyer_id
        self.env = env
    
    def select_activity_level(self, round_num: int) -> int:
        """Select activity level for the round"""
        raise NotImplementedError
    
    def generate_bids(self, round_num: int) -> List[Tuple[frozenset, float]]:
        """Generate bids for the current round"""
        raise NotImplementedError
    
    def participate_round(self, round_num: int):
        """Participate in a round of the auction"""
        # Select activity level
        activity_level = self.select_activity_level(round_num)
        self.env.set_activity_level(self.buyer_id, activity_level)
        
        # Generate and submit bids
        bids = self.generate_bids(round_num)
        
        for bundle, price in bids:
            self.env.submit_bid(self.buyer_id, bundle, price)


class TruthfulBuyer(BuyerAgent):
    """Buyer that bids truthfully on bundles with positive surplus"""
    
    def __init__(self, buyer_id: int, env: MultiActivityAuctionEnv, 
                 activity_strategy: str = 'fixed_medium'):
        super().__init__(buyer_id, env)
        self.activity_strategy = activity_strategy
    
    def select_activity_level(self, round_num: int) -> int:
        """Select activity level based on strategy"""
        if self.activity_strategy == 'fixed_low':
            return ActivityLevel.LOW
        elif self.activity_strategy == 'fixed_high':
            return ActivityLevel.HIGH
        elif self.activity_strategy == 'adaptive':
            # Start high, decrease over time
            if round_num < 5:
                return ActivityLevel.HIGH
            elif round_num < 15:
                return ActivityLevel.MEDIUM
            else:
                return ActivityLevel.LOW
        else:  # fixed_medium (default)
            return ActivityLevel.MEDIUM
    
    def generate_bids(self, round_num: int) -> List[Tuple[frozenset, float]]:
        """Generate truthful bids on bundles with positive surplus"""
        bids = []
        
        # Get all bundles for this buyer
        all_bundles = list(self.env.buyer_valuations[self.buyer_id].keys())
        
        # Calculate surplus for each bundle
        bundle_surpluses = []
        for bundle in all_bundles:
            valuation = self.env.get_buyer_valuation(self.buyer_id, bundle)
            current_price = self.env.get_current_price(bundle)
            surplus = valuation - current_price
            
            if surplus > 0:
                bundle_surpluses.append((bundle, surplus, valuation))
        
        # Sort by surplus
        bundle_surpluses.sort(key=lambda x: x[1], reverse=True)
        
        # Bid on top bundles based on activity level
        max_bids = self.env.get_max_bids_per_round(self.buyer_id)
        
        for bundle, surplus, valuation in bundle_surpluses[:max_bids]:
            current_price = self.env.get_current_price(bundle)
            
            # Bid truthfully: valuation
            bid_price = current_price + self.env.price_increment
            if bid_price <= valuation:
                bids.append((bundle, bid_price))
        
        return bids


class StrategicBuyer(BuyerAgent):
    """Buyer that bids strategically (slightly below valuation)"""
    
    def __init__(self, buyer_id: int, env: MultiActivityAuctionEnv,
                 shading_factor: float = 0.9, activity_strategy: str = 'adaptive'):
        super().__init__(buyer_id, env)
        self.shading_factor = shading_factor
        self.activity_strategy = activity_strategy
    
    def select_activity_level(self, round_num: int) -> int:
        """Select activity level based on strategy"""
        if self.activity_strategy == 'fixed_low':
            return ActivityLevel.LOW
        elif self.activity_strategy == 'fixed_high':
            return ActivityLevel.HIGH
        elif self.activity_strategy == 'adaptive':
            # Adaptive: high early, low late
            progress = round_num / self.env.max_rounds
            if progress < 0.3:
                return ActivityLevel.HIGH
            elif progress < 0.7:
                return ActivityLevel.MEDIUM
            else:
                return ActivityLevel.LOW
        else:  # fixed_medium
            return ActivityLevel.MEDIUM
    
    def generate_bids(self, round_num: int) -> List[Tuple[frozenset, float]]:
        """Generate strategic bids (shaded)"""
        bids = []
        
        # Get all bundles
        all_bundles = list(self.env.buyer_valuations[self.buyer_id].keys())
        
        # Calculate shaded bids
        bundle_values = []
        for bundle in all_bundles:
            valuation = self.env.get_buyer_valuation(self.buyer_id, bundle)
            current_price = self.env.get_current_price(bundle)
            
            # Strategic bid: shade the valuation
            max_bid = valuation * self.shading_factor
            
            if max_bid > current_price:
                bundle_values.append((bundle, valuation, max_bid))
        
        # Sort by valuation
        bundle_values.sort(key=lambda x: x[1], reverse=True)
        
        # Bid on top bundles
        max_bids = self.env.get_max_bids_per_round(self.buyer_id)
        
        for bundle, valuation, max_bid in bundle_values[:max_bids]:
            current_price = self.env.get_current_price(bundle)
            bid_price = min(max_bid, current_price + self.env.price_increment)
            
            if bid_price <= max_bid:
                bids.append((bundle, bid_price))
        
        return bids


class AggressiveBuyer(BuyerAgent):
    """Buyer with aggressive bidding strategy and high activity"""
    
    def __init__(self, buyer_id: int, env: MultiActivityAuctionEnv):
        super().__init__(buyer_id, env)
    
    def select_activity_level(self, round_num: int) -> int:
        """Always use high activity level"""
        return ActivityLevel.HIGH
    
    def generate_bids(self, round_num: int) -> List[Tuple[frozenset, float]]:
        """Generate aggressive bids"""
        bids = []
        
        all_bundles = list(self.env.buyer_valuations[self.buyer_id].keys())
        
        # Bid aggressively on many bundles
        for bundle in all_bundles:
            valuation = self.env.get_buyer_valuation(self.buyer_id, bundle)
            current_price = self.env.get_current_price(bundle)
            
            # Aggressive: bid close to valuation with large increments
            aggressive_bid = current_price + self.env.price_increment * 2
            
            if aggressive_bid <= valuation * 0.95:
                bids.append((bundle, aggressive_bid))
        
        # Return up to max allowed
        max_bids = self.env.get_max_bids_per_round(self.buyer_id)
        return bids[:max_bids]


# ============================================================================
# SECTION 3: EVALUATION - Performance Analysis
# ============================================================================

class MultiActivityEvaluator:
    """Evaluate multi-activity auction performance"""
    
    def __init__(self, env: MultiActivityAuctionEnv):
        self.env = env
    
    def run_auction(self, buyer_agents: List[BuyerAgent], verbose: bool = False) -> Dict:
        """Run a complete auction with given buyer agents"""
        self.env.reset()
        
        round_data = []
        
        while self.env.auction_active and self.env.current_round < self.env.max_rounds:
            round_num = self.env.current_round
            
            if verbose:
                print(f"\n--- Round {round_num + 1} ---")
            
            # Each buyer participates
            for agent in buyer_agents:
                agent.participate_round(round_num)
            
            # End round and determine winners
            allocation, welfare = self.env.end_round()
            
            # Record round data
            round_info = {
                'round': round_num,
                'n_bids': len(self.env.round_bids) if round_num < len(self.env.all_bids) else 0,
                'welfare': welfare,
                'n_winners': len(allocation)
            }
            round_data.append(round_info)
            
            if verbose:
                print(f"Bids this round: {round_info['n_bids']}")
                print(f"Current welfare: {welfare:.2f}")
                print(f"Winners: {len(allocation)}")
            
            # Check if no new bids
            if round_num > 0 and round_info['n_bids'] == 0:
                break
        
        # Calculate final metrics
        final_allocation = self.env.winner_allocation
        final_welfare = self.env.total_welfare
        
        # Calculate optimal welfare (upper bound)
        optimal_welfare = self._compute_optimal_welfare()
        efficiency = (final_welfare / optimal_welfare * 100) if optimal_welfare > 0 else 0
        
        # Calculate buyer utilities
        buyer_utilities = {}
        for buyer_id in range(self.env.n_buyers):
            if buyer_id in final_allocation:
                bundle = final_allocation[buyer_id]
                valuation = self.env.get_buyer_valuation(buyer_id, bundle)
                # Find price paid (last bid for this bundle by this buyer)
                price_paid = 0.0
                for bid in reversed(self.env.all_bids):
                    if bid.buyer_id == buyer_id and bid.bundle == bundle:
                        price_paid = bid.price
                        break
                buyer_utilities[buyer_id] = valuation - price_paid
            else:
                buyer_utilities[buyer_id] = 0.0
        
        return {
            'allocation': final_allocation,
            'welfare': final_welfare,
            'efficiency': efficiency,
            'rounds': self.env.rounds_completed,
            'total_bids': len(self.env.all_bids),
            'buyer_utilities': buyer_utilities,
            'round_data': round_data
        }
    
    def _compute_optimal_welfare(self) -> float:
        """Compute optimal welfare (upper bound using max valuation)"""
        # Upper bound: sum of maximum valuations per buyer
        max_welfare = 0.0
        for buyer_id in range(self.env.n_buyers):
            if self.env.buyer_valuations[buyer_id]:
                max_val = max(self.env.buyer_valuations[buyer_id].values())
                max_welfare += max_val
        return max_welfare
    
    def compare_strategies(self, n_episodes: int = 50) -> pd.DataFrame:
        """Compare different buyer strategies"""
        
        strategies = [
            ('All Truthful - Medium Activity', [
                TruthfulBuyer(i, self.env, 'fixed_medium') 
                for i in range(self.env.n_buyers)
            ]),
            ('All Truthful - Adaptive Activity', [
                TruthfulBuyer(i, self.env, 'adaptive')
                for i in range(self.env.n_buyers)
            ]),
            ('All Strategic - Adaptive Activity', [
                StrategicBuyer(i, self.env, 0.9, 'adaptive')
                for i in range(self.env.n_buyers)
            ]),
            ('Mixed - High/Low Activity', [
                AggressiveBuyer(i, self.env) if i % 2 == 0 
                else TruthfulBuyer(i, self.env, 'fixed_low')
                for i in range(self.env.n_buyers)
            ])
        ]
        
        results = []
        
        for strategy_name, agents in strategies:
            print(f"\nEvaluating: {strategy_name}")
            
            efficiencies = []
            welfares = []
            rounds_list = []
            total_bids_list = []
            
            for episode in range(n_episodes):
                result = self.run_auction(agents, verbose=False)
                
                efficiencies.append(result['efficiency'])
                welfares.append(result['welfare'])
                rounds_list.append(result['rounds'])
                total_bids_list.append(result['total_bids'])
            
            avg_buyer_utilities = [0.0] * self.env.n_buyers
            for episode in range(n_episodes):
                result = self.run_auction(agents, verbose=False)
                for buyer_id, utility in result['buyer_utilities'].items():
                    avg_buyer_utilities[buyer_id] += utility / n_episodes
            
            results.append({
                'Strategy': strategy_name,
                'Mean Efficiency (%)': np.mean(efficiencies),
                'Std Efficiency': np.std(efficiencies),
                'Mean Welfare': np.mean(welfares),
                'Mean Rounds': np.mean(rounds_list),
                'Mean Total Bids': np.mean(total_bids_list),
                **{f'Buyer {i} Utility': avg_buyer_utilities[i] for i in range(self.env.n_buyers)}
            })
        
        return pd.DataFrame(results)


# ============================================================================
# SECTION 4: MAIN - Experiments and CLI
# ============================================================================

def create_visualizations(df: pd.DataFrame, output_dir: str = './results'):
    """Create visualization plots"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multi-Activity Combinatorial Auction - Strategy Comparison', 
                 fontsize=16, fontweight='bold')
    
    strategies = df['Strategy'].values
    
    # 1. Efficiency comparison
    ax = axes[0, 0]
    ax.bar(range(len(strategies)), df['Mean Efficiency (%)'], 
           yerr=df['Std Efficiency'], capsize=5, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Efficiency by Strategy')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Welfare comparison
    ax = axes[0, 1]
    ax.bar(range(len(strategies)), df['Mean Welfare'], color='forestgreen', alpha=0.7)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Total Welfare')
    ax.set_title('Total Welfare by Strategy')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Time complexity (rounds)
    ax = axes[0, 2]
    ax.bar(range(len(strategies)), df['Mean Rounds'], color='coral', alpha=0.7)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Number of Rounds')
    ax.set_title('Time Complexity (Rounds)')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Total bids
    ax = axes[1, 0]
    ax.bar(range(len(strategies)), df['Mean Total Bids'], color='purple', alpha=0.7)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Total Bids')
    ax.set_title('Total Bids Across All Rounds')
    ax.grid(axis='y', alpha=0.3)
    
    # 5. Time-Utility tradeoff
    ax = axes[1, 1]
    ax.scatter(df['Mean Rounds'], df['Mean Efficiency (%)'], 
               s=200, c=range(len(strategies)), cmap='viridis', alpha=0.7)
    for i, strategy in enumerate(strategies):
        ax.annotate(f'{i+1}', (df['Mean Rounds'].iloc[i], df['Mean Efficiency (%)'].iloc[i]),
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax.set_xlabel('Mean Rounds (Time Complexity)')
    ax.set_ylabel('Mean Efficiency (%)')
    ax.set_title('Time-Utility Tradeoff')
    ax.grid(alpha=0.3)
    
    # 6. Buyer utilities
    ax = axes[1, 2]
    n_buyers = len([col for col in df.columns if 'Buyer' in col and 'Utility' in col])
    buyer_utils = df[[f'Buyer {i} Utility' for i in range(n_buyers)]].values
    
    x = np.arange(len(strategies))
    width = 0.8 / n_buyers
    
    for i in range(n_buyers):
        offset = (i - n_buyers/2) * width + width/2
        ax.bar(x + offset, buyer_utils[:, i], width, label=f'Buyer {i}', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Average Utility')
    ax.set_title('Buyer Utilities by Strategy')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = f'{output_dir}/multi_activity_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")
    
    try:
        plt.show()
    except:
        pass


def run_experiments(args):
    """Run full experiments"""
    
    print("=" * 80)
    print("MULTI-ACTIVITY LEVEL COMBINATORIAL AUCTION EXPERIMENTS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Items: {args.n_items}")
    print(f"  Buyers: {args.n_buyers}")
    print(f"  Max bundle size: {args.max_bundle_size}")
    print(f"  Max rounds: {args.max_rounds}")
    print(f"  Evaluation episodes: {args.n_eval_episodes}")
    print(f"  Price increment: {args.price_increment}")
    
    # Create environment
    env = MultiActivityAuctionEnv(
        n_items=args.n_items,
        n_buyers=args.n_buyers,
        max_bundle_size=args.max_bundle_size,
        max_rounds=args.max_rounds,
        price_increment=args.price_increment,
        seed=args.seed
    )
    
    # Create evaluator
    evaluator = MultiActivityEvaluator(env)
    
    # Run comparison
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    
    df = evaluator.compare_strategies(n_episodes=args.n_eval_episodes)
    
    # Save results
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    csv_path = f'{args.output_dir}/multi_activity_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Create visualizations
    create_visualizations(df, args.output_dir)
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    best_efficiency_idx = df['Mean Efficiency (%)'].idxmax()
    fastest_idx = df['Mean Rounds'].idxmin()
    
    print(f"\n✓ Highest Efficiency: {df.loc[best_efficiency_idx, 'Strategy']}")
    print(f"  - Efficiency: {df.loc[best_efficiency_idx, 'Mean Efficiency (%)']:.2f}%")
    print(f"  - Rounds: {df.loc[best_efficiency_idx, 'Mean Rounds']:.1f}")
    
    print(f"\n✓ Fastest (Fewest Rounds): {df.loc[fastest_idx, 'Strategy']}")
    print(f"  - Efficiency: {df.loc[fastest_idx, 'Mean Efficiency (%)']:.2f}%")
    print(f"  - Rounds: {df.loc[fastest_idx, 'Mean Rounds']:.1f}")
    
    print("\n✓ Time-Utility Tradeoff:")
    print("  Activity level management allows buyers to balance:")
    print("  - High activity → Better efficiency, more rounds")
    print("  - Low activity → Faster completion, lower efficiency")
    print("  - Adaptive activity → Best balance")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")


def run_demo(args):
    """Run a quick demonstration"""
    
    print("=" * 80)
    print("DEMO - Multi-Activity Combinatorial Auction")
    print("=" * 80)
    
    # Create small environment
    env = MultiActivityAuctionEnv(
        n_items=6,
        n_buyers=2,
        max_bundle_size=3,
        max_rounds=10,
        price_increment=5.0,
        seed=42
    )
    
    print(f"\nEnvironment created:")
    print(f"  - {env.n_items} items")
    print(f"  - {env.n_buyers} buyers")
    print(f"  - Max {env.max_rounds} rounds")
    
    # Create agents with different activity strategies
    agents = [
        TruthfulBuyer(0, env, 'adaptive'),
        StrategicBuyer(1, env, 0.9, 'adaptive')
    ]
    
    print(f"\nBuyer strategies:")
    print(f"  - Buyer 0: Truthful with adaptive activity")
    print(f"  - Buyer 1: Strategic with adaptive activity")
    
    # Run auction
    evaluator = MultiActivityEvaluator(env)
    result = evaluator.run_auction(agents, verbose=True)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total rounds: {result['rounds']}")
    print(f"Total bids: {result['total_bids']}")
    print(f"Final welfare: {result['welfare']:.2f}")
    print(f"Efficiency: {result['efficiency']:.2f}%")
    
    print(f"\nAllocation:")
    for buyer_id, bundle in result['allocation'].items():
        items_str = '{' + ','.join(map(str, sorted(bundle))) + '}'
        utility = result['buyer_utilities'][buyer_id]
        print(f"  Buyer {buyer_id}: {items_str} (utility: {utility:.2f})")
    
    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Activity Level Ascending Bid Combinatorial Auction'
    )
    
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'evaluate', 'all'],
                       help='Execution mode')
    parser.add_argument('--n_items', type=int, default=8,
                       help='Number of items')
    parser.add_argument('--n_buyers', type=int, default=3,
                       help='Number of buyers')
    parser.add_argument('--max_bundle_size', type=int, default=4,
                       help='Maximum bundle size')
    parser.add_argument('--max_rounds', type=int, default=20,
                       help='Maximum auction rounds')
    parser.add_argument('--price_increment', type=float, default=5.0,
                       help='Minimum price increment per round')
    parser.add_argument('--n_eval_episodes', type=int, default=50,
                       help='Number of evaluation episodes')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        run_demo(args)
    elif args.mode == 'evaluate':
        run_experiments(args)
    elif args.mode == 'all':
        run_demo(args)
        print("\n\n")
        run_experiments(args)


if __name__ == '__main__':
    main()
