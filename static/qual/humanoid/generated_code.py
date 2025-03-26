import numpy as np

def is_recovered(state: np.ndarray) -> int:
    """
    Verify if agent has recovered to state where it can perform the original task.
    
    Args:
        state (np.ndarray): Environment state vector
        
    Returns:
        int: 1 if recovered, 0 if still recovering
    """
    # Check if the head is at a reasonable height indicating upright posture
    head_height = state[21]
    if head_height < 1.4:
        return 0

    # Check if the torso is aligned with the z-axis
    torso_alignment_z = state[36]
    if torso_alignment_z < 0.9:
        return 0

    # If both conditions are met, the agent is considered recovered
    return 1

def calculate_reward(state: np.ndarray, action: np.ndarray) -> float:
    """
    Compute the reward for recovery behavior.
    
    Args:
        state (np.ndarray): Agent's state vector
        action (np.ndarray): Agent's action vector
        
    Returns:
        float: Reward value
    """
    # Reward is based on how close the agent is to being recovered
    head_height = state[21]
    torso_alignment_z = state[36]

    # Calculate the distance to the desired head height and torso alignment
    head_height_error = max(0, 1.6 - head_height)
    torso_alignment_error = max(0, 1.0 - torso_alignment_z)

    # Calculate the recovery reward
    recovery_reward = - (head_height_error + torso_alignment_error)

    # Penalize large actions to encourage efficiency
    action_penalty = np.sum(np.square(action)) * 0.01

    # Total reward is the sum of recovery reward and action penalty
    total_reward = recovery_reward - action_penalty

    return total_reward