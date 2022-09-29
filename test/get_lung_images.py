import os
import numpy as np
import pandas as pd
from scipy.ndimage import label

def remove_small_objects(vol, min_area):
    ret_vol = np.copy(vol)
    s = np.ones(shape=(3, 3, 3))
    labels, num_ft = label(vol, structure=s)
    for i in range(1, num_ft + 1):
        num = np.sum(labels == i)
        if num < min_area:
            ret_vol[labels == i] = 0
    return ret_vol


def bounding_cube(vol, offset=0):
    a = np.where(vol != 0)
    bbox = np.min(a[0]) - offset, np.min(a[1]) - offset, np.min(a[2]) - offset, \
           np.max(a[0]) + 1 + offset, np.max(a[1]) + 1 + offset, np.max(a[2]) + 1 + offset
    return bbox


def extract_lung_area(npy_lung_mask):
    # cleaning lung mask
    # lungs = np.copy(npy_lung_mask)
    lungs = remove_small_objects(npy_lung_mask, 200000)
    roi = bounding_cube(lungs, offset=0)
    # r, c, d = lungs.shape
    # for i in range(d):
    #    im = lungs[:, :, i]
    #    cv2.imshow("Lungs", (255*im).astype(np.uint8))
    #    cv2.waitKey()
    return roi


def main():
    cohort1 = "/home/ubuntu/local-s3-bucket/cohort1/"
    img_path = os.listdir(cohort1)

    roi_dict = {}

    for n, name in enumerate(os.listdir(cohort1)):
        #name = img_path[index].split('cohort1')[-1]
        #name = "Patient 1.npy"

        mask_dir = "/home/ubuntu/local-s3-bucket/lung_masks/"  # make generic later
        mask = np.load(mask_dir + name)
        roi_limits = extract_lung_area(mask)
        roi_dict["{}".format(name)] = roi_limits
        if n > 10:
            break

    df = pd.DataFrame(roi_dict)
    df.to_csv("roi_lung.csv")
        #img = np.load(img_path[index])  # self.loader(img_path[index])
        #img = img[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]]



if __name__ == "__main__":
    main()