import numpy as np
import pandas as pd
import sys
import time

# Q-Learning class
class QLearning():
    def __init__(self, S: list[int], A: list[int], gamma: float, Q: np.ndarray, alpha: float):
        self.A = A          # action space 
        self.gamma = gamma  # discount factor
        self.Q = Q           # The action value function Q[s, a] is a numpy array
        self.alpha = alpha  # learning rate
        self.S = S  # state space
    # Return the action to be taken in state s
    def lookahead(self, s: int, a: int):
        return self.Q[s, a]
    
    # Update the action value function Q[s, a] based on the reward r and the next state s_prime
    def update(self, s: int, a: int, r: float, s_prime: int):
        self.Q[s, a] += self.alpha * (r + (self.gamma * np.max(self.Q[s_prime])) - self.Q[s, a])     
    
    # Transform index to state
    def index_to_state(self, index: int) -> tuple:
        """Convert state index back to position and velocity"""
        vel = index // 500
        pos = index % 500
        return pos, vel
    
    def reward_shaped_update(self, s: int, a: int, r: float, s_prime: int):
        # Modified Q-learning update with reward shaping
        shaped_reward = self.shape_reward(s, r, s_prime)
        
        self.Q[s, a] += self.alpha * (shaped_reward + (self.gamma * np.max(self.Q[s_prime])) - self.Q[s, a])     

    def shape_reward(self, s: int, r: float, s_prime: int) -> float:
        """Shape the reward to provide better learning signals"""
        
        # Extract position and velocity from states
        pos, vel = self.index_to_state(s)
        next_pos, next_vel = self.index_to_state(s_prime)
        # Normalize position and velocity
        pos -= 250
        next_pos -= 250
        vel -= 50
        next_vel -= 50
        # Reward for moving right
        position_reward = (next_pos - pos) * 0.1
        
        # Reward for gaining velocity in useful directions
        velocity_reward = 0
        if pos < 0:  # Left side of track
            velocity_reward = abs(next_vel) * 0.1  # Reward building up velocity
        else:  # Right side of track
            velocity_reward = next_vel * 0.1 if next_vel > 0 else 0  # Reward moving right
            
        return r + position_reward + velocity_reward
    
def main():
    # Read the input file
    if sys.argv[1] == "small":
        A = [1,2,3,4]
        gamma = 0.95
        Q = np.zeros((100, 4))
        alpha = 0.2
        S = list(range(100))
        
    elif sys.argv[1] == "medium":
        A = [1,2,3,4,5,6,7]
        gamma = 1.0
        Q = np.zeros((50000, 7))
        alpha = 0.1
        S = list(range(50000))
        
    elif sys.argv[1] == "large":
        A = [1,2,3,4,5,6,7,8,9]
        gamma = 0.95
        Q = np.zeros((302020, 9))
        alpha = 0.15
        S = list(range(302020))
        
    else:
        print("Invalid input")
        return
    
    # Read the data file          
    data = pd.read_csv("data/" + sys.argv[1] + ".csv", skiprows=1)   
    q_learning = QLearning(S, A, gamma, Q, alpha)
    start_time = time.time()
    
    # Train the Q-Learning agent
    max_iter = 10
    for episdoe in range(max_iter):
        for index, row in data.iterrows():
            s, a, r, s_p = row
            # Update the action value function Q[s, a]
            # Convert the 1-based state and action to 0-based state and action
            if sys.argv[1] == "medium":
                q_learning.reward_shaped_update(s - 1, a - 1, r, s_p - 1)
            else:
                q_learning.update(s - 1, a - 1, r, s_p - 1)
    end_time = time.time()
    print("Training time with iter = ",max_iter ,", alpha = ",alpha , ":", end_time - start_time)
    
    # Generate the optimal policy
    policy = np.argmax(q_learning.Q, axis=1)
     # Write the policy to a file
    with open(sys.argv[1]+ ".policy", "w") as f:
        for action in policy:
            # Convert 0-based action to 1-based action
            f.write(f"{action + 1}\n")  

if __name__ == "__main__":
    main()        
          