import numpy as np
import pandas as pd
import imageio
import glob

# import kaggle dataset 
df = pd.read_csv('/home/ubuntu/Clothing-Clusters/data/styles.csv', error_bad_lines=False)
df.set_index('id', inplace=True)

# get list of file names for images
path = r'/home/ubuntu/Clothing-Clusters/data/images'
#### path = r'/Users/Kelly/galvanize/week8/data/images'

files = glob.glob(path + "/*.jpg")
idx_series = pd.Series(files, dtype=object).str.replace('/home/ubuntu/Clothing-Clusters/data/images/', '').str.replace('.jpg', '')

pic_arr = np.array(files)
#### pic_arr = np.array(files)[0:300]

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
#### pic_df0 = pd.DataFrame(images_arr, index=idx_series[0:300].astype(int), dtype='int')
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

print(wear_pics.head())
print(images_arr.shape)
# double check this is True
# set(wear_pics.index) == set(wearable_df.index)