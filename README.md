# YouTube-Style Recommendation System using Collaborative Filtering

## 1. Overview

This project implements a recommendation system inspired by platforms such as YouTube, where personalized content is suggested to users based on their past interactions. The system is built using collaborative filtering techniques and a neural network architecture implemented in PyTorch.

Instead of real YouTube data, the MovieLens 100K dataset is used as a proxy. In this context, movies are treated as videos and user ratings are interpreted as user engagement. The objective is to learn user preferences and recommend relevant content accordingly.

---

## 2. Problem Statement

Modern content platforms face the challenge of filtering vast amounts of data to provide users with relevant recommendations. The goal of this project is to design a machine learning model that predicts user preferences and generates personalized recommendations based on historical interaction data.

---

## 3. Methodology

The system is based on collaborative filtering using learned embeddings for users and items. The workflow is as follows:

- Data preprocessing: User IDs and item IDs are normalized and converted into index-based representations.
- Embedding layer: Each user and item is represented as a dense vector.
- Feature combination: User and item embeddings are concatenated.
- Neural network: Fully connected layers learn interaction patterns.
- Output layer: A sigmoid activation is used to predict ratings within a defined range.

The model is trained using Mean Squared Error loss and optimized using the Adam optimizer. A learning rate scheduler is used to improve convergence.

---

## 4. Dataset

The project uses the MovieLens 100K dataset, which contains:

- 943 users
- 1682 items (treated as videos)
- 100,000 ratings

Dataset Source:  
https://grouplens.org/datasets/movielens/100k/

---

## 5. Installation and Setup

### Prerequisites

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/recommendation-system.git
cd recommendation-system
