import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



#Create a matrix which rows are filtered movies and columns are genre affliation
df_movies = pd.read_csv('liked_movies.csv')
pivot_table = pd.pivot_table(df_movies, index='item_id', values=['Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'FilmNoir', 'Horror', 'Musical', 'Mystery', 'Romance', 'SciFi', 'Thriller', 'War', 'Western'], aggfunc='sum')


# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pivot_table)


#Elbow method for optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# Run the K-means algorithm with the optimal number of clusters which is 5 based on the elbow method
n_clusters = 5  
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(scaled_data)

# Visualize the results with PCA
# Principle Component Analysis - a stats method help reduce the dimensionality of a dataset and retain most of its important information
# Data are transformed into princial components, which are linear combinations of the original variables

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans.labels_)
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red')

plt.title('K-means Clustering Results')
plt.show()

#Add a cluster column to the dataframe
df_movies['cluster'] = kmeans.predict(scaled_data)
df_movies = df_movies.drop(['Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'FilmNoir', 'Horror', 'Musical', 'Mystery', 'Romance', 'SciFi', 'Thriller', 'War', 'Western'], axis=1)

#Create a dataframe called df_merged that include users'id and users'fav movies:
df_favorites = pd.read_csv('user_fav.csv')
df_merged = pd.merge(df_movies, df_favorites, on='item_id')
df_merged = df_merged.sort_values(by='user_id')

print(df_merged)

def fav_cluster(id):
    new_dict={}
    for index,row in df_merged.iterrows():
        if row['user_id'] == id: 
            if row['cluster'] not in new_dict:
                new_dict[row['cluster']] = 1
            else:
                new_dict[row['cluster']] += 1
    return  max(new_dict,key=new_dict.get)


def recommend_movies(id): 
    fav = fav_cluster(id)
    count = 1
    out = "Recommend Movies:\n"
    for index,row in df_merged.iterrows():
        if row['cluster'] ==fav and count<11:
            out += str(count) + ')' + row['title'] + '\n'
            count+=1
    return out

id_input = int(input("What's your user ID? "))
print(recommend_movies(id_input))



            


