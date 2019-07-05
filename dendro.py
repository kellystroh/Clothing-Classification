from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list

from full_df import df, bw_img
from subset_df import wearable_subset

wear_df, bw_wear = wearable_subset(df, bw_img)

Z = linkage(bw_wear[0:3000], 'complete')
Z_idx = wear_df[0:3000].index

plt.figure(figsize=(20,7))
den = dendrogram(Z, truncate_mode='lastp', p=30, 
           leaf_rotation=45, leaf_font_size=15, 
           show_contracted=True, show_leaf_counts=True)

sizes = np.concatenate([np.ones(Z.shape[0]+1), Z[:,3]]).astype(int)
count_leaves = sizes[den['leaves']]

leaf_groups = []
for i in range(len(count_leaves)):
    start = 0
    if i!=0:
        start += count_leaves[i-1]
    end = start + count_leaves[i] 
    each_list = list(leaves_list(Z))[start:end]
    leaf_groups.append(each_list)

leaf_categories_by_group = []
for g in leaf_groups:
    g_cat_list = []
    for each in g: 
        g_cat_list.append(wearable_df.loc[Z_idx[each], 'masterCategory'])
    leaf_categories_by_group.append(g_cat_list)

num_g = len(leaf_groups)

