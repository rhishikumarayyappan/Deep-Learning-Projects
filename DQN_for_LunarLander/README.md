# Reinforcement Learning: DQN for Autonomous Control

**Author:** Rhishi Kumar Ayyappan

---

## Project Overview

**Business Challenge:**  
Many real-world problems (e.g., robotics, ad-bidding, supply chains) are too complex for human-made rules and lack labeled datasets. Reinforcement Learning (RL) solves this by creating "agents" that learn optimal strategies on their own through trial-and-error.

This project implements a Deep Q-Network (DQN) **from scratch** to train an autonomous agent to master the complex physics of the `LunarLander` environment.

---

## Key Achievements & Metrics

-   **Solved the Benchmark:** Achieved a 100-episode average score of **241.71**, far exceeding the official "Solved" threshold of 200.
-   **Built DQN From Scratch:** Implemented the complete DQN algorithm, including:
    * **Replay Buffer:** For stable, de-correlated learning.
    * **Q-Network & Target Network:** A dual-network architecture to prevent unstable learning.
    * **Epsilon-Greedy Policy:** To balance exploration (learning) vs. exploitation (performing).
-   **High-Performance Code:** Utilized `tf.GradientTape` and `@tf.function` decorators to create a high-speed training loop, ensuring the model could be trained efficiently within a single Colab session.
-   **Tangible Result:** Produced a fully trained agent capable of perfectly landing the lunar module, as demonstrated in the final GIF.

---

## Methods Used

-   **Environment:** `Gymnasium "LunarLander-v3"`. The agent receives 8 state variables (position, velocity, etc.) and chooses from 4 discrete actions.
-   **Data:** **None.** The agent learns *tabula rasa* (from a blank slate) by interacting with the environment and storing `(state, action, reward, next_state)` transitions in a **Replay Buffer**.
-   **Architecture:** A Keras `Sequential` model (DQN) with 8 inputs (state) and 4 outputs (Q-values for each action).
-   **Core Algorithm:**
    1.  The agent **explores** using an Epsilon-Greedy strategy.
    2.  Experiences are stored in the `ReplayBuffer`.
    3.  The `Q-Network` learns by sampling mini-batches from the buffer.
    4.  It uses the **Bellman Equation** to calculate the target Q-value, using the frozen `Target-Network` for stability.
-   **Evaluation:** The 100-episode rolling average of the agent's total reward.

---

## Business Impact

-   **A Framework for Automation:** This project is a **reusable template** for solving complex, dynamic optimization problems where no "answer key" exists.
-   **Solves Real-World Problems:** The same logic can be applied to high-value business tasks:
    * **Robotics & Logistics:** Training a robotic arm to pick and place items efficiently.
    * **Finance & E-commerce:** Optimizing a trading algorithm or a dynamic pricing strategy.
    * **Marketing:** Creating an ad-bidding bot that learns to maximize ROI.
-   **Demonstrates Advanced AI Mastery:** Proves mastery of Reinforcement Learning, a third major paradigm of AI (distinct from Supervised and Unsupervised learning) that is critical for building autonomous, self-improving systems.

---

## Visuals

-   **Final Trained Agent (GIF):** A demonstration of the final, "intelligent" policy in action, showing the agent successfully landing the craft.
-   **Training Rewards Plot:** The technical proof of learning, showing the agent's average score (red line) starting from negative values and climbing to well above the "Solved" threshold (green line).

![Trained Agent Landing](images/lunar_lander_solved.gif)
![Training Reward Curve](images:dqn_reward_plot.png)

---

## How to Run

1.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```
    *(See requirements.txt file)*

2.  **Launch notebook:**
    ```bash
    jupyter notebook DQN_LunarLander.ipynb
    ```
    *(Notebook must be run in a GPU-accelerated environment like Google Colab)*

---

## Model Explainability & Monitoring

-   **Reward Curve:** In Reinforcement Learning, the reward curve *is* the primary explainability tool. It provides a real-time diagnostic of the agent's learning process, showing exactly *how* it learned to be intelligent.
-   **Epsilon (Exploration) Decay:** The `epsilon` value is monitored as it decays, showing the agent's shift from "learning" (random exploration) to "performing" (exploiting its knowledge).

---

## Tech Stack

-   Python
-   Gymnasium (for the environment)
-   TensorFlow 2.x & Keras
-   NumPy
-   Pandas (for rolling average)
-   Pygame & ImageIO (for rendering the GIF)
-   SWIG (build dependency)

---

***For the full code—including the from-scratch Replay Buffer and training loop—see the included Jupyter Notebook!***
