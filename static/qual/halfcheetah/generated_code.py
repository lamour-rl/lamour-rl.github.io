import numpy as np

def is_recovered(state: np.ndarray) -> int:
    """
    Verify if agent has recovered to state where it can perform the original task.
    
    Args:
        state (np.ndarray): Environment state vector
        
    Returns:
        int: 1 if recovered, 0 if still recovering
    """
    # Check if the z-coordinate of the head is close to 0 (upright position)
    # and the angle of the head is close to 0 (aligned with the torso)
    z_threshold = 0.1  # Allowable deviation in z-coordinate
    angle_threshold = 0.1  # Allowable deviation in angle

    z_position = state[0]
    head_angle = state[1]

    if abs(z_position) < z_threshold and abs(head_angle) < angle_threshold:
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
    # Define thresholds for recovery
    z_threshold = 0.1
    angle_threshold = 0.1

    z_position = state[0]
    head_angle = state[1]

    # Calculate distance from recovery criteria
    z_distance = abs(z_position)
    angle_distance = abs(head_angle)

    # Reward is more negative the further away from recovery criteria
    recovery_reward = - (z_distance + angle_distance)

    # Penalize large actions to encourage efficiency
    action_penalty_coefficient = 0.1
    action_penalty = action_penalty_coefficient * np.sum(np.square(action))

    # Total reward is the sum of recovery reward and action penalty
    total_reward = recovery_reward - action_penalty

    return total_reward