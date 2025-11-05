# Deep-Learning-Projects

**Author:** Rhishi Kumar Ayyappan

---

## Portfolio Overview

A curated collection of advanced, from-scratch deep learning projects. Each project explores a different AI paradigm (Generative, Sequential, and Reinforcement Learning) with a focus on deep architectural understanding, rigorous benchmarking, and business-value analysis—ready for recruiters and hiring managers.

---

## Projects

### 1. Generative AI: VAE for Face Generation (From Scratch)

**Challenge:** Solving data scarcity, a major bottleneck in AI. This project builds a generative model to *create new data* rather than just classify it.

**Impact:** Successfully built a VAE from scratch that can generate novel, synthetic 64x64 faces. This framework can be used for data augmentation, anonymization, and as a foundation for advanced generative AI.

**Highlights:** VAE (Encoder, Decoder, Sampler) built from scratch, Keras model subclassing, custom loss (Reconstruction + KL), and visual proof of a "learned" latent space via image morphing.

**[See the project folder for full details](https://github.com/rhishikumarayyappan/Deep-Learning-Projects/tree/main/VAE_for_Face_Generation)**

---

### 2. Time-Series Forecasting: Transformer vs. LSTM Benchmark

**Challenge:** Determining if a complex, state-of-the-art Transformer is *actually* better than a simpler, industry-standard LSTM for a real-world, multivariate forecasting problem.

**Impact:** **Prevented over-engineering.** The benchmark proved the simpler LSTM was **3.57% more accurate** and trained faster, saving compute costs and development time for what would have been a worse model.

**Highlights:** Rigorous head-to-head benchmarking, Transformer built from scratch, and critical **Worst-Case Error Analysis** to identify *when* the model fails (e.g., sudden, volatile events).

**[See the project folder for full details](https://github.com/rhishikumarayyappan/Deep-Learning-Projects/tree/main/Time_Series_Forecasting_Transformer_vs._LSTM_Benchmark)**

---

### 3. Reinforcement Learning: DQN for Autonomous Control

**Challenge:** Training an AI agent to master a complex physics environment (`LunarLander`) with **no labeled data**, learning purely from trial-and-error rewards.

**Impact:** Successfully trained an autonomous agent to "solve" the environment, achieving an average score of **241.71** (far exceeding the 200 benchmark). This framework is the foundation for robotics, ad-bidding, and game AI.

**Highlights:** DQN built from scratch (Replay Buffer, Q-Network, Target Network), high-speed `@tf.function` training loop, Epsilon-Greedy policy, and a final "solved" GIF of the agent landing perfectly.

**[See the project folder for full details](https://github.com/rhishikumarayyappan/Deep-Learning-Projects/tree/main/DQN_for_LunarLander)**

---

## Business Impact & Metrics

All projects include clear, provable metrics (e.g., MAE, reward scores, loss convergence) and visual analytics (e.g., generated images, forecast plots, "solved" GIFs).
Each folder details the business value, from providing quantitative benchmarks to creating autonomous agents and new data.
Every project includes best practices: reproducible Colab notebooks, `requirements.txt`, and clear visual results.

---

## How to Use

Browse each project folder for its detailed README, code, outputs, and key results.
Install and run using the instructions in each project’s README.
All notebooks are designed to run on the free tier of Google Colab (with GPU).

---

## Tech Highlights

Python, TensorFlow 2.x, Keras (Functional API & Model Subclassing), Gymnasium (Reinforcement Learning), DQN, Transformers, Generative AI (VAEs), Pandas, NumPy, Matplotlib, ImageIO, Pygame, SWIG

---

**Portfolio designed for clarity, business value, and deep technical mastery.**
**Contact for further details or demo requests!**
