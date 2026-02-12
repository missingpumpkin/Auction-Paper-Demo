"""
Main Training and Evaluation Script
Run complete experiments for Multi-Attribute Auction with RL
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For headless environments

from mav_auction_env import MultiAttributeAuctionEnv, TruthfulSellerAgent
from rl_agents import RLSellerAgent, MultiAgentTrainer
from evaluation import AuctionEvaluator, create_comparison_report


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Attribute Auction RL Experiments')
    
    # Environment parameters
    parser.add_argument('--n_sellers', type=int, default=3, help='Number of sellers')
    parser.add_argument('--n_attributes', type=int, default=2, help='Number of attributes')
    parser.add_argument('--max_price', type=float, default=100.0, help='Maximum price')
    
    # Training parameters
    parser.add_argument('--algorithm', type=str, default='PPO', 
                       choices=['PPO', 'SAC', 'TD3'], help='RL algorithm')
    parser.add_argument('--total_timesteps', type=int, default=100000, 
                       help='Total training timesteps per agent')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    
    # Evaluation parameters
    parser.add_argument('--n_eval_episodes', type=int, default=1000, 
                       help='Number of episodes for evaluation')
    
    # Experiment control
    parser.add_argument('--mode', type=str, default='all',
                       choices=['train', 'evaluate', 'verify', 'all'],
                       help='Experiment mode')
    parser.add_argument('--load_models', action='store_true', 
                       help='Load pre-trained models instead of training')
    parser.add_argument('--model_dir', type=str, default='./models', 
                       help='Directory for saving/loading models')
    parser.add_argument('--output_dir', type=str, default='./results', 
                       help='Directory for saving results')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def train_agents(args):
    """Train RL agents"""
    print("\n" + "="*80)
    print("TRAINING PHASE")
    print("="*80 + "\n")
    
    env_kwargs = {
        'n_sellers': args.n_sellers,
        'n_attributes': args.n_attributes,
        'max_price': args.max_price
    }
    
    training_kwargs = {
        'learning_rate': args.learning_rate,
    }
    
    trainer = MultiAgentTrainer(
        n_sellers=args.n_sellers,
        env_kwargs=env_kwargs,
        algorithm=args.algorithm,
        **training_kwargs
    )
    
    trainer.train_all(
        total_timesteps=args.total_timesteps,
        save_dir=args.model_dir
    )
    
    print("\n✓ Training completed!")
    
    return trainer


def evaluate_agents(args, trainer=None):
    """Evaluate different strategies"""
    print("\n" + "="*80)
    print("EVALUATION PHASE")
    print("="*80 + "\n")
    
    # Create evaluator
    evaluator = AuctionEvaluator(
        n_sellers=args.n_sellers,
        n_attributes=args.n_attributes,
        max_price=args.max_price
    )
    
    # Create strategy sets
    strategies = {}
    
    # 1. Truthful strategy (theoretical optimal)
    truthful_agents = [
        TruthfulSellerAgent(i, evaluator.env.observation_space, evaluator.env.action_space)
        for i in range(args.n_sellers)
    ]
    strategies['Truthful (Theory)'] = truthful_agents
    
    # 2. RL-trained strategy
    if trainer is not None or args.load_models:
        rl_agents = []
        env_kwargs = {
            'n_sellers': args.n_sellers,
            'n_attributes': args.n_attributes,
            'max_price': args.max_price
        }
        
        for i in range(args.n_sellers):
            agent = RLSellerAgent(
                seller_id=i,
                observation_space=evaluator.env.observation_space,
                action_space=evaluator.env.action_space,
                algorithm=args.algorithm,
                env_kwargs=env_kwargs
            )
            
            if args.load_models:
                model_path = os.path.join(args.model_dir, f'seller_{i}_{args.algorithm}')
                agent.load(model_path)
            
            rl_agents.append(agent)
        
        strategies[f'RL ({args.algorithm})'] = rl_agents
    
    # 3. Random strategy (baseline)
    class RandomAgent:
        def __init__(self, seller_id, max_price):
            self.seller_id = seller_id
            self.max_price = max_price
        
        def get_action(self, obs):
            return np.array([np.random.uniform(0, self.max_price)])
    
    random_agents = [
        RandomAgent(i, args.max_price) for i in range(args.n_sellers)
    ]
    strategies['Random'] = random_agents
    
    # 4. Greedy overbidding (deviating strategy)
    class GreedyAgent:
        def __init__(self, seller_id):
            self.seller_id = seller_id
            self.truthful = TruthfulSellerAgent(seller_id, None, None)
        
        def get_action(self, obs):
            truthful_bid = self.truthful.get_action(obs)
            return truthful_bid * 1.5  # Overbid by 50%
    
    greedy_agents = [
        GreedyAgent(i) for i in range(args.n_sellers)
    ]
    strategies['Greedy (Overbid 50%)'] = greedy_agents
    
    # Run comparison
    print(f"\nComparing {len(strategies)} strategies over {args.n_eval_episodes} episodes each...")
    
    df, ic_results, efficiency_results = create_comparison_report(
        evaluator=evaluator,
        strategies=strategies,
        n_episodes=args.n_eval_episodes,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("\n✓ Evaluation completed!")
    
    return df, ic_results, efficiency_results


def verify_properties(args):
    """Verify theoretical properties from the paper"""
    print("\n" + "="*80)
    print("VERIFICATION PHASE - Testing Theoretical Properties")
    print("="*80 + "\n")
    
    evaluator = AuctionEvaluator(
        n_sellers=args.n_sellers,
        n_attributes=args.n_attributes,
        max_price=args.max_price
    )
    
    print("Property 1: Seller Incentive Compatibility")
    print("-" * 60)
    ic_results = evaluator.verify_incentive_compatibility(n_trials=args.n_eval_episodes)
    
    print("\n\nProperty 2: Efficiency (Total Utility Maximization)")
    print("-" * 60)
    efficiency_results = evaluator.verify_efficiency(n_trials=args.n_eval_episodes)
    
    print("\n✓ Verification completed!")
    
    return ic_results, efficiency_results


def run_experiment(args):
    """Run complete experiment pipeline"""
    np.random.seed(args.seed)
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("MULTI-ATTRIBUTE AUCTION RL EXPERIMENTS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Number of sellers: {args.n_sellers}")
    print(f"  Number of attributes: {args.n_attributes}")
    print(f"  RL Algorithm: {args.algorithm}")
    print(f"  Training timesteps: {args.total_timesteps}")
    print(f"  Evaluation episodes: {args.n_eval_episodes}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {args.output_dir}")
    
    trainer = None
    
    # Training
    if args.mode in ['train', 'all']:
        if not args.load_models:
            trainer = train_agents(args)
    
    # Evaluation
    if args.mode in ['evaluate', 'all']:
        evaluate_agents(args, trainer)
    
    # Verification
    if args.mode in ['verify', 'all']:
        verify_properties(args)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Models saved to: {args.model_dir}")
    print("\nCheck the following files:")
    print(f"  - {os.path.join(args.output_dir, 'strategy_comparison.csv')}")
    print(f"  - {os.path.join(args.output_dir, 'strategy_comparison.png')}")
    print(f"  - {os.path.join(args.output_dir, 'verification_report.txt')}")


def quick_demo():
    """
    Quick demonstration of the MAV auction system
    """
    print("\n" + "="*80)
    print("QUICK DEMO - Multi-Attribute Auction")
    print("="*80 + "\n")
    
    # Create simple environment
    env = MultiAttributeAuctionEnv(
        n_sellers=3,
        n_attributes=2,
        max_price=100.0
    )
    
    print("Environment created:")
    print(f"  - {env.n_sellers} sellers")
    print(f"  - {env.n_attributes} attributes")
    print(f"  - Buyer valuation type: {env.buyer_type}")
    print(f"  - Seller cost types: {env.seller_types}")
    
    # Create truthful agents
    agents = [
        TruthfulSellerAgent(i, env.observation_space, env.action_space)
        for i in range(env.n_sellers)
    ]
    
    print("\nRunning 5 auction episodes with truthful bidding...\n")
    
    for episode in range(5):
        obs, info = env.reset()
        
        print(f"Episode {episode + 1}:")
        print(f"  D values: {env.D_values}")
        
        bids = []
        for i in range(env.n_sellers):
            obs = env._get_observation(i)
            action = agents[i].get_action(obs)
            bids.append(float(action[0]))
        
        print(f"  Bids: {bids}")
        
        # Execute auction
        env.reset()
        for bid in bids:
            obs, reward, terminated, truncated, info = env.step(np.array([bid]))
        
        if info['winner'] >= 0:
            print(f"  Winner: Seller {info['winner']}")
            print(f"  Transaction price: {info['transaction_price']:.2f}")
            print(f"  Buyer utility: {info['buyer_utility']:.2f}")
            print(f"  Seller utilities: {info['seller_utilities']}")
            print(f"  Total utility: {info['buyer_utility'] + sum(info['seller_utilities']):.2f}")
        else:
            print("  No transaction")
        print()
    
    print("="*80)
    print("Demo completed!")
    print("="*80)


if __name__ == '__main__':
    args = parse_args()
    
    # Check if running demo mode
    if hasattr(args, 'demo') and args.demo:
        quick_demo()
    else:
        run_experiment(args)
