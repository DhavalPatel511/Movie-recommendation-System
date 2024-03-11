#!/usr/bin/env python
# coding: utf-8

# **Business Problem**
# 
# MovieLens data sets were collected by the GroupLens Research Project at the University of Minnesota.      
# The dataset can be downloaded from here  -- (https://grouplens.org/datasets/movielens/100k/)
# This data set consists of: 
# 	* 100,000 ratings (1-5) from 943 users on 1682 movies. 
# 	* Each user has rated at least 20 movies. 
#     * Simple demographic info for the users (age, gender, occupation, zip)
# 
# The data was collected through the MovieLens web site (movielens.umn.edu) during the seven-month period from September 19th,1997 through April 22nd, 1998.

# **Task and Approach:**
# 
# We need to work on the MovieLens dataset and build a model to recommend movies to the end users

# **Step 1 :** Importing Libraries and Understanding Data

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# To make data visualisations display in Jupyter Notebooks 
import numpy as np   # linear algebra
import pandas as pd  # Data processing, Input & Output load
import matplotlib.pyplot as plt # Visuvalization & plotting
import seaborn as sns # Also for Data visuvalization 

from sklearn.metrics.pairwise import cosine_similarity  # Compute cosine similarity between samples in X and Y.
from scipy import sparse  #  sparse matrix package for numeric data.
from scipy.sparse.linalg import svds # svd algorithm

import warnings   # To avoid warning messages in the code run
warnings.filterwarnings("ignore")


# **Step 2 :** Loading Data  & Corss chekcing 

# In[3]:


Rating = pd.read_csv('Ratings.csv') 
Movie_D = pd.read_csv('Movie details.csv',encoding='latin-1') ##Movie details 
User_Info = pd.read_csv('user level info.csv',encoding='latin-1') ## if you have a unicode string, you can use encode to convert


# In[4]:


Rating.shape


# In[5]:


Rating.head()


# * Item id means it is Movie id 
# * Item_ID chnaged as Movie id for the better redability pupose 
# 

# In[6]:


Rating.columns = ['user_id', 'movie_id', 'rating', 'timestamp'] 


# Renaming the columns to avoid the space in the column name text 

# In[7]:


Movie_D.shape


# In[8]:


Movie_D.head()


# In[9]:


Movie_D.columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date ',
       'IMDb_URL', 'unknown', 'Action ', 'Adventure', 'Animation',
       'Childrens', 'Comedy ', 'Crime ', ' Documentary ', 'Drama',
       ' Fantasy', 'Film-Noir ', 'Horror ', 'Musical', 'Mystery',
       ' Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']


# Renaming the columns to avoid the space in the column name text 

# **To get our desired information in a single dataframe, we can merge the two dataframes objects on the movie_Id column since it is common between the two dataframes.**
# 
# **We can do this using merge() function from the Pandas library**

# In[10]:


Movie_Rating = pd.merge(Rating ,Movie_D,on = 'movie_id')
Movie_Rating.describe()


# In[11]:


Movie_Rating.shape


# **We can see the Average rating for all the movie is 3.5**              
# **We can also see 25 percentile also indicating avaerage is 3 highest is 5**

# In[12]:


n_users = Movie_Rating.user_id.unique().shape[0]
n_items = Movie_Rating.movie_id.unique().shape[0]
print(n_items,n_users)


# No of unique users & No of unique Movies 

# In[13]:


# Calculate mean rating of all movies 
Movie_Stats = pd.DataFrame(Movie_Rating.groupby('movie_title')['rating'].mean())
Movie_Stats.sort_values(by = ['rating'],ascending=False).head()


# **Let's now plot the total number of ratings for a movie**

# In[14]:


# Calculate count rating of all movies 

Movie_Stats['Count_of_ratings'] = pd.DataFrame(Movie_Rating.groupby('movie_title')['rating'].count())
Movie_Stats.sort_values(by =['Count_of_ratings'], ascending=False).head()


# **Now we know that both the average rating per movie and the number of ratings per movie are important attributes**

# **Plot a histogram for the number of ratings**

# In[15]:


Movie_Stats['Count_of_ratings'].hist(bins=50)


# **From the output, you can see that most of the movies have received less than 50 ratings.**
# It is evident that the data has a weak normal distribution with the mean of around 3.5. There are a few outliers in the data

# In[16]:


sns.jointplot(x='rating', y='Count_of_ratings', data=Movie_Stats)


# * The graph shows that, in general, movies with higher average ratings actually have more number of ratings, compared with movies that have lower average ratings.

#  ### Finding Similarities Between Movies

# * We will use the correlation between the ratings of a movie as the similarity metric.
# * To see the corrilation we will create Pivot table between user_id ,movies, ratings

# In[17]:


User_movie_Rating = Movie_Rating.pivot_table(index='user_id', columns='movie_title', values='rating')
User_movie_Rating.head()


# In[18]:


##We can achieve this by computing the correlation between these two movies ratings and the ratings of the rest of the movies in the dataset. 
##The first step is to create a dataframe with the ratings of these movies 

# Example pick up one movie related rating  
User_movie_Rating['Air Force One (1997)']


# ## Correlation Similarity

# * We can find the correlation between the user ratings for the **given movie**  and all the other movies using corrwith() function as shown below:

# In[19]:


Similarity = User_movie_Rating.corrwith(User_movie_Rating['Air Force One (1997)'])
Similarity.head()


# In[20]:


corr_similar = pd.DataFrame(Similarity, columns=['Correlation'])
corr_similar.sort_values(['Correlation'], ascending= False).head(10)


# #### We will add the count of rating also to see why many movies are exactly correlating for the single movie 

# In[21]:


corr_similar_num_of_rating = corr_similar.join(Movie_Stats['Count_of_ratings'])
corr_similar_num_of_rating.sort_values(['Correlation'], ascending= False).head(10)


# * We can able to see  that a movie cannot be declared similar to the another movie based on just 2 or 3  ratings. 
# 
# * This is why we need to filter  movies correlated to given movie  that have more than 30/50 ratings

# In[22]:


corr_similar_num_of_rating[corr_similar_num_of_rating ['Count_of_ratings']>50].sort_values('Correlation', ascending=False).head()


# **Creation the user defined function to get the similar movies to recommend**
# * All the above steps created as one UDF so that we can pass the movie title and get the recomendations
# 

# In[23]:


def get_recommendations(title):
    # Get the movie ratings of the movie that matches the title
    Movie_rating = User_movie_Rating[title]

    # Get the  similarity corrilated  scores of all movies with that movie
    sim_scores = User_movie_Rating.corrwith(Movie_rating)

    # Sort the movies based on the similarity scores
    corr_title = pd.DataFrame(sim_scores, columns=['Correlation'])
    
    # Removing na values 
    corr_title.dropna(inplace=True)
    
    corr_title = corr_title.join(Movie_Stats['Count_of_ratings'])
    
    # Return the top 10 most similar movies
    return corr_title[corr_title ['Count_of_ratings']>50].sort_values('Correlation', ascending=False).head()


# In[24]:


# Usage of the above function
get_recommendations('Air Force One (1997)')


# In[25]:


get_recommendations('Star Wars (1977)')


# =======================================================================================
# ## Cosine Similarties

# * Untill now we have seen the correlation wise now we are going to use  cosine similariy to find the similar movies
# * Filter out required columns from the dataset 

# In[26]:


Movie_cosine = Movie_Rating[['user_id','movie_id','rating']]
Movie_cosine.head()


# * Sparse matrix we are going to create using above data      
# * A sparse matrix in Coordinate format this is also called as triplet format

# In[27]:


data = Movie_cosine.rating
col = Movie_cosine.movie_id
row = Movie_cosine.user_id

R = sparse.coo_matrix((data, (row, col))).tocsr()
print ('{0}x{1} user by movie matrix'.format(*R.shape))


# * Keeping data ,col, row we call it as Triplet Format of Matrix
# 
# * The individual elements of the matrix can be listed in any order, and if there are multiple items for the same nonzero position, the values provided for those positions are added.
# 
# * Using the **cosine similarity** to measure the similarity between a pair of vectors
# 
# * With the cosine similarity, we are going to evaluate the similarity between two vectors based on the angle between them. The smaller the angle, the more similar the two vectors are
# 
# * If you recall from trigonometry, the range of the cosine function goes from -1 to 1. Some important properties of cosine to recall:
# 
# >+ Cosine(0°) = 1
# + Cosine(90°) = 0
# + Cosine(180°) = -1
# 
# 
# * If we restrict our vectors to non-negative values (as in the case of movie ratings, usually going from a 1-5 scale), then the angle of separation between the two vectors is bound between 0° and 90°

# In[28]:


find_similarities = cosine_similarity(R.T) # We are transposing the matrix 
print (find_similarities.shape)


# In[29]:


def Get_Top5_Similarmovies(model, movie_id, n=5):
    return model[movie_id].argsort()[::-1][:n].tolist()  # Here movie id is index


# * index is started with 0 and movie id is started with 1  

# In[30]:


Movie_D.head()


# In[31]:


Movie_D.iloc[4] 


# * Here Index 4 means movie id 5 

# In[32]:


Movie_D.iloc[Get_Top5_Similarmovies(find_similarities, 4)]


# =======================================================================================
# ## Matrix Factorization

# ### Singular Value Decomposition (Matrix Factorization)
# 
# $$ df\_array = U\Sigma V^T $$
# 
# Now using Singular Value Decomposition, we will decompose our dataframe into smaller matrices. This can be done directly with the help of `scipy` using `scipy.sparse.linalg.svds`.

# For Matrix factorization, we need normalized values.

# In[33]:


df_matrix = User_movie_Rating.fillna(0)


# In[34]:


df_matrix


# In[35]:


normalized_values = df_matrix.values - np.mean(df_matrix.values, axis=1).reshape(-1,1)
normalized_values


# In[36]:


normalized_values.shape


# In[37]:


U, SIGMA, VT = svds(normalized_values)
U.shape, SIGMA.shape, VT.shape


# Since `SIGMA` is not the same shape as `U` or `VT`, we make it into a diagonal matrix.

# In[38]:


SIGMA = np.diag(SIGMA)


# In[39]:


U.shape, SIGMA.shape, VT.shape


# Now we can see the values of $U, \Sigma, V$. Notice that $\Sigma$ is a diagonal matrix now.

# In[40]:


df_matrix


# In[41]:


reconstructed_df = np.dot(np.dot(U, SIGMA), VT) + np.mean(df_matrix.values, axis=1).reshape(-1, 1)
predictions_df = pd.DataFrame(reconstructed_df, columns = df_matrix.columns)
predictions_df


# As you can see above, the `predictions_df` shows values of ratings very close to the actual ratings given by users to the movies in the `df_matrix` dataframe. The remaining values in `predictions_df` of which corresponding cells are 0 in `df_matrix`, are the predicted ratings from the multiplication of the factors that we decomposed the `df_matrix` into.
# <br><br>
# Let's create some helper functions to find out what a user's ratings and the genres which the user rates the highest.
# 

# In[42]:


def get_genres(in_df):
    df1 = in_df.iloc[:, 5:24]
    df1 = df1.eq(1).dot(' ' + df1.columns.values).apply(lambda x: x.replace(' ', '|').replace('|', '', 1))
    return df1

# x.replace(' ' , "|") in entire dataset
# eq() is used to compared & eq(1) is used compared 1 in dataset if match then pick that column only

def user_ratings(user_id, df):
    df1 = df[df['user_id'] == user_id].sort_values(['rating'], ascending=False).copy() # if user id match thn sor
    df2 = get_genres(df1)  # genres
    df3 = pd.concat([df1[['movie_title', 'rating']], df2], axis=1)  # select 2 column & df2 data
    df3 = df3.rename(columns = {0:'genres'})   ## Rename the column
    return df3


# Now let's see what the user 100 likes. Movie_Rating is data set

# In[43]:


user_ratings(100, Movie_Rating)


# We see that user 100 likes a lot of **Drama** movies. The movies he has rated highest have a Drama element in them.
# 
# Now let's write a helper function, to get the recommendations for the user, from the matrix factors that we created earlier.
# 
# The process is as follows.
# 
# * We select movies which are not rated by the user.
# * We merge the resultant (non rated) movies dataframe with the `predictions_df` matrix on the movie title.
# * Now we have the predicted values for the movie ratings.
# * We select the movies with the highest predicted ratings, and recommend that to the user.

# In[44]:


def get_recommendations(user_id, number_of_recommendations=10):
    df1 = Movie_D[~Movie_D['movie_id'].isin(Movie_Rating[Movie_Rating['user_id'] == user_id]                                         .sort_values(['rating'], ascending=False)['movie_id'])]                                         .merge(pd.DataFrame(predictions_df.iloc[user_id - 1]                                                             .sort_values(ascending=False))                                                .reset_index(),                                                how='left', left_on='movie_title', right_on='movie_title')                                         .sort_values(user_id - 1, ascending=False)                                         .iloc[:number_of_recommendations, :]
    
    df2 = pd.concat([df1[['movie_title']], get_genres(df1)], axis=1)
    
    df2 = df2.rename(columns = {0:'genres'})
    
    return df2


# In[46]:


get_recommendations(99)


# Thus, from the prediction we see that the user will be recommended a lot of movies with an element of Drama in the genre.
