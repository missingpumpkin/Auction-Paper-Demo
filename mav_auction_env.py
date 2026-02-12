"""
Multi-Attribute Auction Environment (MAV)
Based on the paper: "An Ascending Bid Multi-Attribute Auction Method"
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional


class MultiAttributeAuctionEnv(gym.Env):
    """
    Multi-Attribute Auction Environment implementing the MAV method
    
    The auction involves:
    - One buyer with valuation function V(a)
    - Multiple sellers with cost functions C_i(a)
    - Attributes: continuous or discrete attribute space
    """
    
    def __init__(
        self,
        n_sellers: int = 3,
        n_attributes: int = 2,
        attribute_dims: List[int] = None,
        max_price: float = 100.0,
        buyer_type: str = 'linear',
        seller_types: List[str] = None
    ):
        super().__init__()
        
        self.n_sellers = n_sellers
        self.n_attributes = n_attributes
        self.max_price = max_price
        
        # Attribute space - for simplicity, use discrete attributes
        if attribute_dims is None:
            attribute_dims = [10] * n_attributes
        self.attribute_dims = attribute_dims
        
        # Action space for sellers: bid amount (0 to max_price)
        self.action_space = spaces.Box(
            low=0.0,
            high=max_price,
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation space for sellers: their cost function parameters + buyer's announced V'
        # Simplified: observe discrete attribute values and associated values
        obs_dim = n_attributes + 2  # attributes + buyer_valuation + own_cost
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize buyer and seller types
        self.buyer_type = buyer_type
        if seller_types is None:
            seller_types = ['linear'] * n_sellers
        self.seller_types = seller_types
        
        # Generate functions
        self._generate_buyer_function()
        self._generate_seller_functions()
        
        self.reset()
    
    def _generate_buyer_function(self):
        """Generate buyer's true valuation function V(a)"""
        if self.buyer_type == 'linear':
            # V(a) = sum(w_i * a_i) + bias
            self.buyer_weights = np.random.uniform(0.5, 2.0, self.n_attributes)
            self.buyer_bias = np.random.uniform(10, 30)
        elif self.buyer_type == 'nonlinear':
            # V(a) = sum(w_i * a_i^2) + bias
            self.buyer_weights = np.random.uniform(0.1, 0.5, self.n_attributes)
            self.buyer_bias = np.random.uniform(10, 30)
        
    def _generate_seller_functions(self):
        """Generate sellers' cost functions C_i(a)"""
        self.seller_weights = []
        self.seller_biases = []
        
        for seller_type in self.seller_types:
            if seller_type == 'linear':
                weights = np.random.uniform(0.3, 1.5, self.n_attributes)
                bias = np.random.uniform(5, 20)
            elif seller_type == 'nonlinear':
                weights = np.random.uniform(0.05, 0.3, self.n_attributes)
                bias = np.random.uniform(5, 20)
            else:
                weights = np.random.uniform(0.5, 1.5, self.n_attributes)
                bias = np.random.uniform(5, 20)
            
            self.seller_weights.append(weights)
            self.seller_biases.append(bias)
    
    def buyer_valuation(self, attributes: np.ndarray, true: bool = True) -> float:
        """
        Compute buyer's valuation for given attributes
        
        Args:
            attributes: Attribute vector
            true: If True, use true V, else use announced V'
        """
        if self.buyer_type == 'linear':
            return np.dot(self.buyer_weights, attributes) + self.buyer_bias
        elif self.buyer_type == 'nonlinear':
            return np.dot(self.buyer_weights, attributes**2) + self.buyer_bias
        return 0.0
    
    def seller_cost(self, seller_id: int, attributes: np.ndarray) -> float:
        """Compute seller's cost for given attributes"""
        weights = self.seller_weights[seller_id]
        bias = self.seller_biases[seller_id]
        
        if self.seller_types[seller_id] == 'linear':
            return np.dot(weights, attributes) + bias
        elif self.seller_types[seller_id] == 'nonlinear':
            return np.dot(weights, attributes**2) + bias
        return 0.0
    
    def compute_D_i(self, seller_id: int) -> Tuple[float, np.ndarray]:
        """
        Compute D_i(V') = max_a(V'(a) - C_i(a)) and optimal attribute a_i^*
        
        This is the key computation for the MAV method
        """
        best_value = -np.inf
        best_attributes = None
        
        # Search over attribute space (simplified grid search)
        n_samples = 20
        for _ in range(n_samples):
            # Random sampling of attribute space
            attributes = np.array([
                np.random.randint(0, dim) for dim in self.attribute_dims
            ])
            
            valuation = self.buyer_valuation(attributes, true=False)  # Use announced V'
            cost = self.seller_cost(seller_id, attributes)
            value = valuation - cost
            
            if value > best_value:
                best_value = value
                best_attributes = attributes
        
        return max(best_value, 0.0), best_attributes
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Current seller being considered
        self.current_seller = 0
        
        # Store bids and attributes
        self.bids = np.zeros(self.n_sellers)
        self.optimal_attributes = [None] * self.n_sellers
        self.D_values = np.zeros(self.n_sellers)
        
        # Compute D_i for all sellers
        for i in range(self.n_sellers):
            self.D_values[i], self.optimal_attributes[i] = self.compute_D_i(i)
        
        # Get observation for first seller
        obs = self._get_observation(self.current_seller)
        info = {}
        
        return obs, info
    
    def _get_observation(self, seller_id: int) -> np.ndarray:
        """Get observation for a specific seller"""
        # Observation includes:
        # - Optimal attributes for this seller
        # - Buyer's announced valuation for these attributes
        # - Seller's own cost
        
        attributes = self.optimal_attributes[seller_id]
        if attributes is None:
            attributes = np.zeros(self.n_attributes)
        
        valuation = self.buyer_valuation(attributes, true=False)
        cost = self.seller_cost(seller_id, attributes)
        
        obs = np.concatenate([
            attributes,
            [valuation],
            [cost]
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step: current seller makes a bid
        
        Args:
            action: Bid amount from current seller
        """
        bid = float(action[0])
        self.bids[self.current_seller] = bid
        
        # Move to next seller
        self.current_seller += 1
        
        # Check if all sellers have bid
        if self.current_seller >= self.n_sellers:
            # Determine winner and compute rewards
            reward, winner, transaction_price, transaction_attributes = self._determine_winner()
            terminated = True
            truncated = False
            
            info = {
                'winner': winner,
                'transaction_price': transaction_price,
                'transaction_attributes': transaction_attributes,
                'all_bids': self.bids.copy(),
                'buyer_utility': self._compute_buyer_utility(winner, transaction_price, transaction_attributes),
                'seller_utilities': self._compute_all_seller_utilities(winner, transaction_price, transaction_attributes)
            }
            
            obs = np.zeros_like(self._get_observation(0))
        else:
            # Get observation for next seller
            obs = self._get_observation(self.current_seller)
            reward = 0.0
            terminated = False
            truncated = False
            info = {}
        
        return obs, reward, terminated, truncated, info
    
    def _determine_winner(self) -> Tuple[float, int, float, np.ndarray]:
        """
        Determine auction winner based on MAV rules
        
        Returns:
            reward: Reward for the winning seller
            winner: Index of winning seller
            transaction_price: Final transaction price
            transaction_attributes: Final transaction attributes
        """
        # Find winning seller(s) with highest bid > 0
        positive_bids = self.bids > 0
        
        if not np.any(positive_bids):
            # No transaction
            return 0.0, -1, 0.0, None
        
        max_bid = np.max(self.bids[positive_bids])
        winners = np.where(self.bids == max_bid)[0]
        
        # Random tie-breaking
        winner = np.random.choice(winners)
        
        # Compute B* = max2(B_i) (second highest bid)
        sorted_bids = np.sort(self.bids[positive_bids])
        if len(sorted_bids) > 1:
            B_star = sorted_bids[-2]
        else:
            B_star = sorted_bids[-1]
        
        # Transaction attributes
        transaction_attributes = self.optimal_attributes[winner]
        
        # Transaction price: P* = V'(a*) - B*
        transaction_price = self.buyer_valuation(transaction_attributes, true=False) - B_star
        
        # Winner's utility: U_w = D_w - B_-w
        if self.D_values[winner] > B_star:
            reward = self.D_values[winner] - B_star
        else:
            reward = 0.0
        
        return reward, winner, transaction_price, transaction_attributes
    
    def _compute_buyer_utility(self, winner: int, price: float, attributes: np.ndarray) -> float:
        """Compute buyer's utility"""
        if winner == -1:
            return 0.0
        
        true_valuation = self.buyer_valuation(attributes, true=True)
        return true_valuation - price
    
    def _compute_all_seller_utilities(self, winner: int, price: float, attributes: np.ndarray) -> np.ndarray:
        """Compute all sellers' utilities"""
        utilities = np.zeros(self.n_sellers)
        
        if winner >= 0:
            cost = self.seller_cost(winner, attributes)
            utilities[winner] = price - cost
        
        return utilities
    
    def render(self):
        """Render the environment (optional)"""
        pass


class SellerAgent:
    """
    Individual seller agent that can be trained with RL
    Each seller has their own policy
    """
    
    def __init__(self, seller_id: int, observation_space, action_space):
        self.seller_id = seller_id
        self.observation_space = observation_space
        self.action_space = action_space
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Get action (bid) based on observation - to be overridden"""
        raise NotImplementedError
    

class TruthfulSellerAgent(SellerAgent):
    """
    Seller agent that follows the dominant strategy from the paper:
    Bid B_i^* = max(D_i(V'), 0)
    """
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Truthful bidding strategy
        
        Observation includes: [attributes, valuation, cost]
        The truthful bid is: valuation - cost (which equals D_i)
        """
        # Extract valuation and cost from observation
        valuation = observation[-2]
        cost = observation[-1]
        
        # Dominant strategy: bid D_i = V'(a*) - C_i(a*)
        bid = max(valuation - cost, 0.0)
        
        return np.array([bid], dtype=np.float32)
