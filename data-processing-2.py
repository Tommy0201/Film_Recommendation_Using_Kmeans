import pandas as pd
import numpy as np

#Create a dataframe consisting userid, item id, and rating only from the u.data file)


fav_cols = ['user_id', 'item_id', 'rating', 'timestamp']
fav_df = pd.read_csv('ml-100k/ml-100k/u.data', sep='\t', names=fav_cols, encoding='latin-1')
fav_df = fav_df.drop(columns=['timestamp'])

#Drop all rating that is below 4 (out of 5)
fav_df = fav_df.drop(index=fav_df[fav_df['rating'] <= 3].index)
fav_df = fav_df.drop(columns=['rating'])
print(fav_df)

#Save the file as user's favorite
fav_df.to_csv('user_fav.csv',index=False)