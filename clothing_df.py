import numpy as np
import pandas as pd
import imageio
import glob

# import kaggle dataset 
df = pd.read_csv('data/fashion_data/styles.csv', error_bad_lines=False)

# get list of file names for images
path = r'/Users/Kelly/galvanize/week8/data/fashion_data/images'
files = glob.glob(path + "/*.jpg")
idx_series = pd.Series(files).str.replace('/Users/Kelly/galvanize/week8/data/fashion_data/images/', '').str.replace('.jpg', '')
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

images_arr = get_pixels(pic_arr)
idx_series = file_series.str.replace('.jpg', "").astype(int)
pix_series = pd.Series(images_col)

pic_df0 = pd.DataFrame(pix_series, index=idx_series, columns=['Pix'])
pic_df = pic_df[pic_df['Pix'].notnull()]

clothing_df = df.merge(pic_df, how="inner", left_on='id', right_on=pic_df.index)