import numpy as np
import pandas as pd
import imageio
import glob

# import kaggle dataset 
df = pd.read_csv('data/fashion_data/styles.csv', error_bad_lines=False)
df.set_index('id', inplace=True)

# get list of file names for images
path = r'/Users/Kelly/galvanize/week8/data/fashion_data/data_images/images'
files = glob.glob(path + "/*.jpg")
idx_series = pd.Series(files, dtype=object).str.replace('/Users/Kelly/galvanize/week8/data/fashion_data/images/', '').str.replace('.jpg', '')

pic_arr = np.array(files)

def get_pixels(files):
    r = np.zeros((len(files),14400))
    for i in range(len(files)):
        img = imageio.imread(files[i])
        flat = np.array(img).flatten()
        if len(flat) == 14400:   
            r[i,:] = flat  
    return r

def problematic_images(files):
    bad_list = []
    for i in range(len(files)):
        img = imageio.imread(files[i])
        flat = np.array(img).flatten()
        if len(flat) != 14400:   
            bad_list.append(files[i])
    return bad_list

# make clothing df
images_arr = get_pixels(pic_arr)

pic_df0 = pd.DataFrame(images_arr, index=idx_series.astype(int), dtype='int')
pic_df = pic_df0[pic_df0.notnull()]

pic_idx = set(pic_df.index)
df_idx = set(df.index)

# include only items present in BOTH indices
df = df[df.index.isin(pic_idx)]
pic_df = pic_df[pic_df.index.isin(df_idx)]

'''
#double check this is True:
pic_idx = df_idx
'''

# make a wearable subset
wearable_list = ['Apparel', 'Accessories', 'Footwear']
wearable_df = df[df['masterCategory'].isin(wearable_list)]

wear_idx = list(wearable_df.index)
wear_pics = pic_df[pic_df.index.isin(wear_idx)]

# double check this is True
# set(wear_pics.index) == set(wearable_df.index)

from sklearn.cluster import AgglomerativeClustering

clust = AgglomerativeClustering(n_clusters=3).fit(wear_pics)


a = wear_pics.index
b = wearable_df.loc[a,:]
c = clust.labels_

b['category_group'] = b['masterCategory'].apply(lambda x: 0 if x =='Apparel' else (1 if x=='Accessories' else 2))
b['cluster_group'] = c
d = b[['category_group', 'cluster_group']]

d[d['category_group']==0].groupby('cluster_group').count()
d[d['category_group']==1].groupby('cluster_group').count()
d[d['category_group']==2].groupby('cluster_group').count()