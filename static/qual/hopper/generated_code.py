import numpy as np

def is_recovered(state: np.ndarray) -> int:
    """
    Verify if agent has recovered to state where it can perform the original task.
    
    Args:
        state (np.ndarray): Environment state vector
        
    Returns:
        int: 1 if recovered, 0 if still recovering
    """
    # Define recovery criteria
    min_height = 0.7  # Minimum height of the torso to be considered upright
    max_angle = np.pi / 6  # Maximum angle deviation from vertical

    # Check if the torso is upright and at a reasonable height
    if state[0] > min_height and abs(state[1]) < max_angle:
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
    # Define recovery criteria
    min_height = 0.7
    max_angle = np.pi / 6

    # Calculate distance from recovery criteria
    height_diff = max(0, min_height - state[0])
    angle_diff = max(0, abs(state[1]) - max_angle)

    # Calculate reward based on proximity to recovery criteria
    recovery_reward = - (height_diff + angle_diff)

    # Penalize large actions to encourage efficiency
    action_penalty = 0.1 * np.sum(np.square(action))

    # Total reward is the sum of recovery reward and action penalty
    total_reward = recovery_reward - action_penalty

    return total_reward