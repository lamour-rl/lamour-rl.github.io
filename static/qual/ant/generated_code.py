import numpy as np

def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw).
    
    Args:
        quaternion (np.ndarray): Quaternion [w, x, y, z]
        
    Returns:
        np.ndarray: Euler angles [roll, pitch, yaw]
    """
    w, x, y, z = quaternion
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp) if abs(sinp) <= 1 else np.pi / 2 * np.sign(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

def is_recovered(state: np.ndarray) -> int:
    """
    Verify if agent has recovered to state where it can perform the original task.
    
    Args:
        state (np.ndarray): Environment state vector
        
    Returns:
        int: 1 if recovered, 0 if still recovering
    """
    # Extract z-coordinate of the torso and quaternion for orientation
    z_position = state[0]
    quaternion = state[1:5]
    
    # Convert quaternion to Euler angles
    euler_angles = quaternion_to_euler(quaternion)
    pitch, roll = euler_angles[1], euler_angles[0]
    
    # Define recovery criteria
    upright_z_threshold = 0.25  # Minimum height for recovery
    angle_threshold = np.pi / 6  # 30 degrees in radians
    
    # Check if the ant is upright and within angle thresholds
    if z_position > upright_z_threshold and abs(pitch) < angle_threshold and abs(roll) < angle_threshold:
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
    # Extract z-coordinate of the torso and quaternion for orientation
    z_position = state[0]
    quaternion = state[1:5]
    
    # Convert quaternion to Euler angles
    euler_angles = quaternion_to_euler(quaternion)
    pitch, roll = euler_angles[1], euler_angles[0]
    
    # Define recovery criteria
    upright_z_threshold = 0.25
    angle_threshold = np.pi / 6
    
    # Calculate distance to recovery criteria
    z_distance = max(0, upright_z_threshold - z_position)
    pitch_distance = max(0, abs(pitch) - angle_threshold)
    roll_distance = max(0, abs(roll) - angle_threshold)
    
    # Calculate reward components
    recovery_reward = -(z_distance + pitch_distance + roll_distance)
    
    # Penalize large actions
    action_penalty = -0.1 * np.sum(np.square(action))
    
    # Total reward
    total_reward = recovery_reward + action_penalty
    
    return total_reward