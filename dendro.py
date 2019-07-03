from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list

from full_df import df, bw_img
from subset_df import wearable_subset

wear_df, bw_wear = wearable_subset(df, bw_img)

Z = linkage(bw_wear[0:3000], 'ward')