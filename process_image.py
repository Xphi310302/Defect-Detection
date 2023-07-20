import pandas as pd
import glob
import numpy as np
import cv2
import os


'''
def get_data(path='train.csv'):
    train_df = pd.read_csv(path)
    train_df = train_df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
    train_df = train_df.reset_index()
    train_df2 = pd.DataFrame({'ImageId': glob.glob('data'+'/train_images/*')})
    train_df2['ImageId'] = train_df2['ImageId'].apply(lambda x: x.split('/')[-1])
    train_df = pd.merge(train_df, train_df2, how='outer', on='ImageId')
    return train_df
train_df = get_data(path = 'data/train.csv')

def make_mask_binary(row_id):
    fname = train_df.iloc[row_id].ImageId

    labels = train_df.iloc[row_id][1:5]
    masks = np.zeros((256, 1600, 1), dtype=np.uint8)    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 255
            masks[:, :,0] = mask.reshape(256, 1600, order='F')
    return fname, masks
'''
def binarize_custom(masks, th = 0.1):
    # Maximum value of each channel per pixel
    m = masks
    # Binarization
    m = (m>th) * 255
    return m

def predict(path, model, show_img = False):
    # name = path.split('/')[-1]

    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_ = img_gray[..., np.newaxis]    # Add channel axis
    img_ = img_[np.newaxis, ...]    # Add batch axis
    img_ = img_ / 255.              # 0～1
    
    masks = model.predict(img_)
    pred_mask = masks[0,:,:,0]
    for i in range(1,4):
        pred_mask +=  masks[0,:,:,i]
    pred_mask = binarize_custom(pred_mask, 0.1)
    if show_img:
        img = cv2.imread(path)
        return img, pred_mask
    else: 
        return pred_mask