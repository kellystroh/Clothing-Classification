def wearable_subset(df, pic_df):
    wearable_list = ['Apparel', 'Accessories', 'Footwear']
    wearable_df = df[df['masterCategory'].isin(wearable_list)]

    wearable_img = pic_df[ wearable_df.index.values ].copy()
    return (wear_pics, wear_df)

def apparel_subset(df, pic_df):
    apparel_df = df[df['masterCategory']=='Apparel']
    apparel_img = pic_df[ apparel_df.index.values ].copy()

def accessory_subset(df, pic_df):
    accessory_df = df[df['masterCategory']=='Accessories']
    accessory_img = pic_df[ accessory_df.index.values ].copy()

def shoe_subset(df, pic_df):
    shoe_df = df[df['masterCategory']=='Footwear']
    shoe_img = all_img[ shoe_df.index.values ].copy()
