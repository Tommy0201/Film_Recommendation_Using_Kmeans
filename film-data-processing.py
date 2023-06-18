import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Load u.item file and convert it to a dataframe called movies_df
movie_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 
              'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 
              'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
              'FilmNoir', 'Horror', 'Musical', 'Mystery', 'Romance', 
              'SciFi', 'Thriller', 'War', 'Western']
movies_df = pd.read_csv('ml-100k/ml-100k/u.item', sep='|', names=movie_cols, encoding='latin-1', usecols=[0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

# Load u.data file, remove redundant columns and merged it with movies_df to form a dataframe called merged_df
ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings_df = pd.read_csv('ml-100k/ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')
merged_df = pd.merge(ratings_df, movies_df, on='item_id')
merged_df = merged_df.drop(columns=['timestamp'])

#Filtered movies with average rating score of 2.5 or higher
dict_total_high_rating={}
for index, row in merged_df.iterrows():
    if row['title'] not in dict_total_high_rating:
        dict_total_high_rating[row['title']]= [5,row['rating']]
    else:
        dict_total_high_rating[row['title']][0]+= 5
        dict_total_high_rating[row['title']][1]+= row['rating']
filtered_movie = [key for key, value in dict_total_high_rating.items() if value[1] / value[0] >= 0.6]


#Create a data frame that contain filtered_movie and store it to a new csv file using the two dataframe aboved
filtered_df = merged_df[merged_df['title'].isin(filtered_movie)]
filtered_df = filtered_df.drop_duplicates(subset=['title'])
filtered_df = filtered_df.drop(columns=['user_id','rating'])

filtered_df.to_csv('liked_movies.csv',index=False)

