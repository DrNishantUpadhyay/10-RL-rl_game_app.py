import streamlit as st
import numpy as np
import random
import time

st.set_page_config(page_title="AI Treasure Hunter", page_icon="🤖")
st.title("🕹️ Reinforcement Learning: Treasure Hunter")
st.write("Watch the AI learn to find the gold through trial and error!")

# 1. Setup the Environment (Grid Size 4x4)
grid_size = 4
goal_pos = (3, 3)
pit_pos = (1, 1)

# 2. The "Memory Bank" (Q-Table)
# Rows = States (16 squares), Columns = Actions (Up, Down, Left, Right)
if 'q_table' not in st.session_state:
    st.session_state.q_table = np.zeros((grid_size * grid_size, 4))

def get_state_index(pos):
    return pos[0] * grid_size + pos[1]

# 3. Training Logic (The AI Learns)
def train_agent(episodes=100):
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 0.3 # Exploration rate
    
    for _ in range(episodes):
        state = (0, 0) # Start at top-left
        while state != goal_pos and state != pit_pos:
            state_idx = get_state_index(state)
            
            # Choose Action (Exploration vs Exploitation)
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3) # Random move
            else:
                action = np.argmax(st.session_state.q_table[state_idx]) # Best known move
            
            # Move logic
            new_state = list(state)
            if action == 0 and state[0] > 0: new_state[0] -= 1 # Up
            elif action == 1 and state[0] < 3: new_state[0] += 1 # Down
            elif action == 2 and state[1] > 0: new_state[1] -= 1 # Left
            elif action == 3 and state[1] < 3: new_state[1] += 1 # Right
            new_state = tuple(new_state)
            
            # Rewards
            if new_state == goal_pos: reward = 10
            elif new_state == pit_pos: reward = -10
            else: reward = -1 # Walking cost
            
            # Update Q-Table (The Bellman Equation)
            old_value = st.session_state.q_table[state_idx, action]
            next_max = np.max(st.session_state.q_table[get_state_index(new_state)])
            st.session_state.q_table[state_idx, action] = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
            
            state = new_state

# UI Buttons
if st.button("Train AI (1000 rounds)"):
    train_agent(1000)
    st.success("Training Complete! The AI has updated its memory.")

if st.button("Watch AI Play"):
    state = (0, 0)
    path = [state]
    while state != goal_pos and state != pit_pos:
        state_idx = get_state_index(state)
        action = np.argmax(st.session_state.q_table[state_idx])
        
        # Movement
        new_state = list(state)
        if action == 0 and state[0] > 0: new_state[0] -= 1 
        elif action == 1 and state[0] < 3: new_state[0] += 1 
        elif action == 2 and state[1] > 0: new_state[1] -= 1 
        elif action == 3 and state[1] < 3: new_state[1] += 1 
        state = tuple(new_state)
        path.append(state)
        if len(path) > 20: break # Safety break

    # Display Grid
    for r in range(grid_size):
        cols = st.columns(grid_size)
        for c in range(grid_size):
            if (r, c) == goal_pos: cols[c].info("💰")
            elif (r, c) == pit_pos: cols[c].error("🕳️")
            elif (r, c) in path: cols[c].success("🤖")
            else: cols[c].write("⬜")
