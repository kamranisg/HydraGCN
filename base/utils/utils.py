import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, roc_auc_score
import pandas as pd

import matplotlib # added by alex
#matplotlib.use('TkAgg') # added by alex
import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import label
from scipy import ndimage

try:
    import seaborn as sns
except Exception as E:
    print(E)


def calc_adj(meta_data, threshold=10, sigma=1):
    """
    Calculate graph using a thresholded Gaussian kernel weighting.

    Args:
        meta_data (torch.float32): N x M tensor with N samples and M columns
        threshold (int): Theshold to use in eucledian distance.
        sigma (int): standard deviation for gaussian kernel weighting.

    Returns:
        cur_adj torch (torch.long): Adjacency matrix as N x N tensor.
    """
    dist = meta_data[:, None, :] - meta_data[None, :, :]
    dist = torch.sum(dist, dim=2)
    dist = torch.sqrt(dist ** 2)
    mask = dist < threshold
    dist = dist/(2 * (sigma ** 2))
    cur_adj = torch.exp(-dist)
    cur_adj = cur_adj * mask
    cur_adj = cur_adj.long()
    return cur_adj


def to_dataframe(y_list, estimator_name, n_class=None, average='binary',
                 multi_class=None):
    """Given a list of output results calculate metrics and return as
    dataframe."""
    # Calculate metrics
    acc = [accuracy_score(x[0], x[2]) for x in y_list]
    mat_coef = [matthews_corrcoef(x[0], x[2]) for x in y_list]
    if n_class == 2:
        fscore = [f1_score(x[0], x[2], average=average) for x in
                  y_list]
        auc_ = [roc_auc_score(x[0], x[1][:, 1:], multi_class=multi_class)
                for x in y_list]
    else:
        fscore = [f1_score(x[0], x[2], average=average) for x in y_list]
        auc_ = [roc_auc_score(x[0], x[1], multi_class=multi_class) for x in
                y_list]
    acc = pd.DataFrame(acc)
    fscore = pd.DataFrame(fscore)
    auc_ = pd.DataFrame(auc_)
    mat_coef = pd.DataFrame(mat_coef)

    # Combine all dataframe
    new_df = pd.concat([acc, fscore, auc_, mat_coef], 1)
    new_df.columns = ['Accuracy', 'F1_score', 'ROC_AUC',
                      'Matthews_Coefficient']
    new_df = new_df.round(5)

    # Prettify dataframe for easy plotting
    new_df.reset_index(inplace=True)
    cols_to_melt = new_df.columns
    new_df = new_df.melt('index', list(cols_to_melt[1:]))
    new_df['estimator'] = estimator_name
    return new_df


def pretty_df_to_boxplots(res_df, order=None, color_pal=None):
    """Given long dataframe plot boxplots."""
    plt.figure(figsize=(20, 6))
    plt.axis('tight')

    metric_names = res_df.variable.unique()
    len_ = len(metric_names)
    list_of_fig = []
    for k, cur_variable in enumerate(metric_names):
        plt.subplot(1, len_, k + 1)
        ax = sns.boxplot(x="variable", y="value", hue="estimator",
                         data=res_df[res_df.variable == cur_variable])
        ax.set(xticklabels=[])
        ax.legend(title='Models')
        plt.xlabel('')
        plt.ylabel(cur_variable)
        plotter = sns.categorical._BoxPlotter(x="variable", y="value",
                                              hue="estimator",
                                              data=res_df.copy(),
                                              order=None,
                                              hue_order=order,
                                              palette=color_pal,
                                              orient=None,
                                              width=.8, color=None,
                                              saturation=.75, dodge=True,
                                              fliersize=5, linewidth=None)
        list_of_fig.append((plotter, ax))
    plt.show()
    return list_of_fig


def df_drop(df, p):
    """ #TODO
    Drop a column of a dataframe if elements in that column are less than
    given percentage.

    Args:
        df (pandas.DataFrame):
            Dataframe containing all data
        p (float):
            Percentage of how much a column should contain. If p is
            0.75, columns with elements greather than 75% will be included.

    Returns:
        res_df (tuple of pandas.DataFrame):
            First element contains the dataframe of interest,
            Second element contains the dataframe of all columns which were
            excluded
    """
    assert p < 1.0, 'p must be less than 1.0'
    df = df.copy()
    # Percentage of elements of every column
    col_p = 1.0 - df.isnull().sum().values / df.shape[0]

    # Column names of interest
    col_of_interest_idx = col_p > p
    main_df = df[df.columns[col_of_interest_idx]]
    del_df = df[df.columns[~col_of_interest_idx]]
    res_df = main_df, del_df
    return res_df


###############################################################
# Added by Ekin

#  ------------------- Extracr ROI based on lungs -----------------
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

