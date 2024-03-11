# Movie Recommendation System

## Overview

This Movie Recommendation System is built using the MovieLens dataset, collected by the GroupLens Research Project at the University of Minnesota. Developed in Python, this project leverages correlation similarity, cosine similarity, and matrix factorization techniques to recommend movies to users based on their viewing history and preferences.

## Features

- **Data Source**: Utilizes the rich MovieLens dataset, providing a comprehensive collection of user ratings and movie metadata for analysis.
- **Correlation Similarity**: Employs user rating patterns to find movies similar to the user's preferences based on Pearson correlation.
- **Cosine Similarity**: Implements cosine similarity to measure the cosine of the angle between two vectors, offering recommendations by identifying users with similar tastes.
- **Matrix Factorization**: Applies matrix factorization techniques for latent factor analysis, enhancing recommendation accuracy and efficiency.
- **Dynamic Recommendations**: Generates personalized movie recommendations for users, adapting to their evolving tastes and preferences.



### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/Movie-Recommendation-System.git
   cd Movie-Recommendation-System
   ```


3. **Run the Recommendation System**
   Follow the instructions in the `recommendation_engine.py` script to start generating movie recommendations.

## Usage Example

To generate movie recommendations for a user, execute the recommendation engine script and follow the prompts to input user preferences.

```bash
python recommendation_engine.py
```

## Results

The Movie Recommendation System effectively generates personalized movie suggestions by analyzing user ratings, employing similarity measures, and utilizing matrix factorization techniques. Users can discover new movies aligned with their tastes, enhancing their viewing experience.

## Acknowledgments

- GroupLens Research Project at the University of Minnesota for providing the MovieLens dataset.
- The open-source community for the Python packages that facilitated the development of this project.

For any questions or further information, please feel free to reach out.
