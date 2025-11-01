import copy
import numpy as np

def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None  # No object found in the mask
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return x_min, y_min, x_max, y_max

def pad_bbox(bbox, width, height, value=10):
    bbox_pad = copy.deepcopy(bbox)
    left_space  = bbox[0]
    up_space    = bbox[1]
    right_space = width  - bbox[2]
    down_space  = height - bbox[3] 
    
    bbox_pad[0] -= min(value, left_space)
    bbox_pad[1] -= min(value, up_space)
    bbox_pad[2] += min(value, right_space)
    bbox_pad[3] += min(value, down_space)

    return bbox_pad

def crop_foreground_from_mask(image, mask, padding_size=10):
    bbox = get_bbox_from_mask(mask)
    if bbox is None:
        return image  # No object found in the mask

    bbox_pad = pad_bbox(list(bbox), image.shape[1], image.shape[0], padding_size) if padding_size > 0 else bbox
    foreground_image = image[bbox_pad[1]:bbox_pad[3], bbox_pad[0]:bbox_pad[2]]
    foreground_mask = mask[bbox_pad[1]:bbox_pad[3], bbox_pad[0]:bbox_pad[2]]
    return foreground_image, foreground_mask
