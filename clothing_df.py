import numpy as np
import pandas as pd
import imageio
from os import listdir
from os.path import isfile, join

# import kaggle dataset 
df = pd.read_csv('/Users/Kelly/galvanize/week8/data/fashion-product-images-small/styles.csv', error_bad_lines=False)

# get list of file names for images
pic_files = [f for f in listdir('data/fashion-product-images-small/images') if isfile(join('data/fashion-product-images-small/images', f))]
pic_arr = np.array(pic_files)

def get_pixels(files):
    pix_list = []
    for each in files:
        img = imageio.imread(('data/fashion-product-images-small/images/' + each))
        flat = np.array(img).flatten()
        pix_list.append(flat)
    return pix_list

images_col = get_pixels(pic_arr)
idx_series = file_series.str.replace('.jpg', "").astype(int)
pix_series = pd.Series(images_col)

pic_df0 = pd.DataFrame(pix_series, index=idx_series, columns=['Pix'])
pic_df = pic_df[pic_df['Pix'].notnull()]

clothing_df = df.merge(pic_df, how="inner", left_on='id', right_on=pic_df.index)