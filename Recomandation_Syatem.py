import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample data - User-item matrix
data = {
    'User': ['User1', 'User2', 'User3', 'User4', 'User5'],
    'Item1': [5, 4, 0, 2, 1],
    'Item2': [0, 5, 4, 0, 2],
    'Item3': [3, 0, 0, 4, 5],
    'Item4': [0, 2, 5, 1, 0],
}

df = pd.DataFrame(data)
df.set_index('User', inplace=True)

# Calculate similarity between users
user_similarity = cosine_similarity(df)

# Convert the similarity matrix to a DataFrame for better understanding
user_similarity_df = pd.DataFrame(user_similarity, index=df.index, columns=df.index)

# Function to get recommendations for a user
def get_recommendations(user):
    # Get the user's preferences
    user_preferences = df.loc[user]

    # Calculate the weighted average of similar users' preferences
    weighted_preferences = user_similarity_df[user] * user_preferences
    weighted_average = weighted_preferences.sum(axis=1) / user_similarity_df[user].sum()

    # Filter out items the user has already rated
    recommendations = weighted_average[~user_preferences.index.isin(user_preferences[user_preferences > 0].index)]

    # Sort recommendations in descending order
    recommendations = recommendations.sort_values(ascending=False)

    return recommendations

# Get recommendations for a specific user
user_to_recommend = 'User1'
recommendations = get_recommendations(user_to_recommend)

# Display the recommendations
print(f"Recommendations for {user_to_recommend}:\n{recommendations}")
