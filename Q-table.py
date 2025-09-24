import numpy as np

# -------------------------
# Environment
# -------------------------
num_states = 5
num_actions = 2  # 0: Left, 1: Right

# Reward table
rewards = np.zeros((num_states, num_actions))
rewards[3, 1] = 1  # from state 3, moving right gives 1
rewards[4, :] = 0  # goal state

# -------------------------
# Initialize Q-table
# -------------------------
q_table = np.zeros((num_states, num_actions))

# -------------------------
# Hyperparameters
# -------------------------
alpha = 0.1     # learning rate
gamma = 0.9     # discount factor
epsilon = 0.8   # exploration
num_episodes = 20
max_steps = 10

# -------------------------
# Training Q-learning
# -------------------------
for episode in range(num_episodes):
    state = 0  # start at state 0
    for step in range(max_steps):
        # Epsilon-greedy action
        if np.random.rand() < epsilon:
            action = np.random.choice(num_actions)
        else:
            action = np.argmax(q_table[state])
        
        # Take action
        next_state = state + 1 if action == 1 else max(state - 1, 0)
        reward = rewards[state, action]
        
        # Q-learning update
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        
        state = next_state

print("Trained Q-table:")
print(q_table)

# -------------------------
# Evaluation
# -------------------------
state = 0
print("\nEvaluation:")
while state < 4:
    action = np.argmax(q_table[state])
    print(f"State {state} -> Action {action}")
    state = state + 1 if action == 1 else max(state - 1, 0)
print(f"Reached goal state {state}")
