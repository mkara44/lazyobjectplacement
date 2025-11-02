import os
import copy
import torch
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

def save_checkpoint(work_dir,
                    epoch,
                    global_context_encoder,
                    pixart_decoder,
                    pixart_decoder_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    keep_last=False,
                    step=None,
                    ):
    os.makedirs(work_dir, exist_ok=True)
    pixart_decoder_state_dict = dict(state_dict=pixart_decoder.state_dict())
    global_context_encoder_state_dict = dict(state_dict=global_context_encoder.state_dict())
    if pixart_decoder_ema is not None:
        pixart_decoder_state_dict['state_dict_ema'] = pixart_decoder_ema.state_dict()
    if optimizer is not None:
        pixart_decoder_state_dict['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        pixart_decoder_state_dict['scheduler'] = lr_scheduler.state_dict()

    if epoch is not None:
        pixart_decoder_state_dict['epoch'] = epoch
        pixart_decoder_file_path = os.path.join(work_dir, f"pixart_decoder_epoch_{epoch}.pth")
        global_context_encoder_file_path = os.path.join(work_dir, f"global_context_encoder_epoch_{epoch}.pth")
        if step is not None:
            pixart_decoder_file_path = pixart_decoder_file_path.split('.pth')[0] + f"_step_{step}.pth"
            global_context_encoder_file_path = global_context_encoder_file_path.split('.pth')[0] + f"_step_{step}.pth"

    torch.save(pixart_decoder_state_dict, pixart_decoder_file_path)
    torch.save(global_context_encoder_state_dict, global_context_encoder_file_path)

    if keep_last:
        for i in range(epoch):
            previous_pixart_decoder_ckpt = pixart_decoder_file_path.format(i)
            previous_global_context_encoder_ckpt = global_context_encoder_file_path.format(i)
            if os.path.exists(previous_pixart_decoder_ckpt):
                os.remove(previous_pixart_decoder_ckpt)
            if os.path.exists(previous_global_context_encoder_ckpt):
                os.remove(previous_global_context_encoder_ckpt)