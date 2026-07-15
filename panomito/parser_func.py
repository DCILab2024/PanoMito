import os
import numpy as np
import matplotlib
from PIL import Image
from scipy.ndimage import uniform_filter

def parser_20260519_fig4_NM_Revision_gsf(filename):
    condition = {}

    if '_mdivi-1_CCCP_' in filename:
        condition['Species'] = '_mdivi-1_CCCP_'
    elif '_cccp_' in filename:
        condition['Species'] = 'cccp'
    elif '_mdivi-1_' in filename:
        condition['Species'] = 'mdivi-1'
    elif '_wt_' in filename:
        condition['Species'] = 'wt'
    else:
        raise ValueError(f"Unable to parse treatment from filename: {filename}")
    
    return condition