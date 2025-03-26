import numpy as np
from typing import Optional

def is_recovered(state: np.ndarray) -> int:
    """
    Verify if agent has recovered to state where it can perform the original task.
    
    Args:
        state (np.ndarray): Environment state vector
        
    Returns:
        int: 1 if recovered, 0 if still recovering
    """
    # Define recovery criteria
    max_end_effector_distance = 0.1  # meters
    max_chair_tilt = 0.1  # radians

    # Check if the state meets recovery criteria
    if state[0] < max_end_effector_distance and abs(state[2]) < max_chair_tilt:
        return 1
    return 0

def calculate_reward(state: np.ndarray, action: np.ndarray) -> float:
    """
    Compute the reward for recovery behavior.
    
    Args:
        state (np.ndarray): Agent's state vector
        action (np.ndarray): Agent's action vector
        
    Returns:
        float: Reward value
    """
    # Initialize base reward
    reward = 0.0
    
    # Extract state components
    end_effector_distance = state[0]
    chair_tilt = state[2]
    
    # Define recovery criteria
    max_end_effector_distance = 0.1  # meters
    max_chair_tilt = 0.1  # radians

    # Logarithmic distance reward (encourages closer approach)
    log_dist_to_chair = np.log(end_effector_distance + 1e-5)
    reward += -end_effector_distance - np.clip(log_dist_to_chair, -10, 0)

    # Tilt-based reward (minimizing deviation from upright position)
    reward += -abs(chair_tilt)

    # Action penalty (encourages smooth control)
    action_norm = np.linalg.norm(action)
    reward -= action_norm * 1e-3  # Scale penalty by action dimensionality

    # Stage-based reward
    stage_reward = -10

    # Stage 1: Approach chair
    if end_effector_distance < max_end_effector_distance:
        stage_reward += 2  # Reward for reaching the chair
    
        # Stage 2: Stabilize chair
        if abs(chair_tilt) < max_chair_tilt:
            stage_reward += 8  # Reward for stabilizing the chair
    
    # Add stage-based reward to final reward
    reward += stage_reward

    return reward