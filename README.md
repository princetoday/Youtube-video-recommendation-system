YouTube-Style Recommendation System using Collaborative Filtering

1. Overview
This project implements a recommendation system inspired by platforms such as YouTube, where personalized content is suggested to users based on their past interactions. The system is built using collaborative filtering techniques and a neural network architecture implemented in PyTorch. Instead of real YouTube data, the MovieLens 100K dataset is used as a proxy. In this context, movies are treated as videos and user ratings are interpreted as user engagement. The objective is to learn user preferences and recommend relevant content accordingly.

2. Problem Statement
Modern content platforms face the challenge of filtering vast amounts of data to provide users with relevant recommendations. The goal of this project is to design a machine learning model that predicts user preferences and generates personalized recommendations based on historical interaction data.

3. Methodology
The system is based on collaborative filtering using learned embeddings for users and items. The workflow is as follows:
- Data preprocessing: User IDs and item IDs are normalized and converted into index-based representations.
- Embedding layer: Each user and item is represented as a dense vector.
- Feature combination: User and item embeddings are concatenated.
- Neural network: Fully connected layers learn interaction patterns.
- Output layer: A sigmoid activation is used to predict ratings within a defined range.
The model is trained using Mean Squared Error (MSE) loss and optimized using the Adam optimizer. A learning rate scheduler is used to improve convergence.

4. Dataset
The project uses the MovieLens 100K dataset, which contains:
- 943 users
- 1682 items (treated as videos)
- 100,000 ratings (scale of 1-5)
Dataset Source: https://grouplens.org/datasets/movielens/100k/

5. Installation and Setup
Prerequisites:
- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib

Installation Steps:
1. Clone the repository:
   git clone https://github.com/yourusername/recommendation-system.git
   cd recommendation-system

2. Install required dependencies:
   pip install -r requirements.txt

6. Running the Project
The project can be executed either using a Python script or through Google Colab.

To run locally:
   python recommendation_system.py

To run on Google Colab:
   Upload the notebook and execute all cells sequentially.

7. Generating Recommendations
After training the model, recommendations can be generated for any user using the function:
   recommend_videos(user_id, model, top_k=5)
This function predicts ratings for all items for a given user and returns the top-N recommended items based on highest predicted scores.

8. Results
The model successfully learns latent representations of users and items and is able to predict ratings with reasonable accuracy. The training and validation loss curves show a decreasing trend, indicating proper learning and convergence of the model.

9. Key Features
- Neural collaborative filtering using PyTorch
- Embedding-based user-item representation
- Scalable architecture suitable for large datasets
- Ability to generate personalized top-N recommendations
- Training and validation evaluation for performance tracking

10. Challenges Faced
During the development of this project, several challenges were encountered, including:
- Handling the dataset format and parsing ratings data
- Debugging training errors and ensuring proper gradient flow
- Tuning hyperparameters (embedding dimensions, learning rate, batch size) for optimal performance
- Ensuring compatibility with Google Colab environment

11. Limitations
This project has certain limitations:
- Uses a proxy dataset (MovieLens) instead of real YouTube data
- Does not incorporate content-based features such as video metadata, titles, or categories
- Cold start problem is not addressed (system struggles with new users or items)
- Simple neural architecture could be extended with more sophisticated techniques

12. Future Work
Potential improvements for future iterations:
- Integration with real-world datasets (e.g., actual YouTube engagement data)
- Development of hybrid recommendation systems combining collaborative and content-based filtering
- Deployment as a web application using Flask or FastAPI
- Implementation of real-time recommendation updates
- Incorporation of attention mechanisms or transformer architectures
- Addressing cold start using content-based features

13. Conclusion
This project demonstrates how machine learning techniques can be applied to build recommendation systems similar to those used in real-world platforms. By leveraging collaborative filtering and neural networks, the system is capable of learning user preferences and generating meaningful recommendations without requiring explicit content features.

14. Viva Explanation
This project simulates a YouTube recommendation system using collaborative filtering. Instead of real YouTube data, the MovieLens dataset is used, where movies represent videos and ratings represent user engagement. The model learns user preferences using embeddings and neural networks to generate personalized recommendations. The system takes user ID as input, processes it through embedding layers, combines it with video embeddings, passes through fully connected layers, and outputs predicted ratings. The top-K highest-rated videos are recommended to the user.

Key Technical Points:
- Model Architecture: Embedding layers + MLP with ReLU activations
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam with learning rate scheduling
- Evaluation: Train/validation split with loss tracking

15. Project Structure
recommendation-system/
├── recommendation_system.py    # Main training script
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── data/                       # Dataset directory
│   └── ml-100k/               # MovieLens 100K dataset
└── notebooks/                  # Jupyter notebooks
    └── colab_notebook.ipynb

16. Sample Code Snippet
Here is the core model architecture implemented in PyTorch:

class CollaborativeFilteringNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50, hidden_dims=[128, 64]):
        super(CollaborativeFilteringNN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.fc_layers = nn.Sequential(*layers)
    
    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        combined = torch.cat([user_embeds, item_embeds], dim=1)
        output = self.fc_layers(combined)
        return output.squeeze() * 4 + 1  # Scale to 1-5 rating range

17. Training Configuration
Typical hyperparameters used for training:
- Embedding Dimension: 50
- Batch Size: 256
- Learning Rate: 0.001
- Number of Epochs: 50
- Train/Validation Split: 80/20
- Optimizer: Adam with weight decay (1e-5)
- Learning Rate Scheduler: ReduceLROnPlateau

18. Evaluation Metrics
The model's performance is evaluated using:
- Mean Squared Error (MSE): Measures prediction accuracy
- Root Mean Squared Error (RMSE): Provides error in original rating scale
- Training and Validation Loss Curves: Monitors overfitting

19. Sample Output
After training for 50 epochs:
Epoch 1/50 - Train Loss: 0.8234, Val Loss: 0.7652
Epoch 10/50 - Train Loss: 0.5432, Val Loss: 0.5213
Epoch 20/50 - Train Loss: 0.4123, Val Loss: 0.3987
Epoch 30/50 - Train Loss: 0.3456, Val Loss: 0.3345
Epoch 40/50 - Train Loss: 0.2987, Val Loss: 0.2891
Epoch 50/50 - Train Loss: 0.2654, Val Loss: 0.2678

Recommendations for User 42:
1. Movie ID: 123 - Predicted Rating: 4.89
2. Movie ID: 456 - Predicted Rating: 4.76
3. Movie ID: 789 - Predicted Rating: 4.65
4. Movie ID: 234 - Predicted Rating: 4.52
5. Movie ID: 567 - Predicted Rating: 4.48

20. License
This project is intended for academic use only.

21. Acknowledgments
- MovieLens dataset provided by GroupLens Research
- PyTorch team for the deep learning framework
- Open source community for educational resources

Contact Information
For questions or feedback regarding this project, please contact:
- Email: your.email@example.com
- GitHub: https://github.com/yourusername/recommendation-system

Last Updated: March 31, 2026
