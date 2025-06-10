import torch
import numpy as np
import logging
import os
import torch.nn.functional as F
import torchvision.utils as utils
import math
import cv2
import statistics
import time

from tqdm import tqdm
from skimage import img_as_ubyte
from math import log10
from utils.utils_metrics import psnr, ssim

def create_dir(file_path):
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)

def tensor2img(tensor, rgb2bgr=False, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')

def batch_PSNR_SSIM(restore, gt):
    # following FFANet and MSBNet
    psnr_list = [psnr(restore[i].unsqueeze(0), gt[i].unsqueeze(0)) for i in range(gt.shape[0])]
    ssim_list = [ssim(restore[i].unsqueeze(0), gt[i].unsqueeze(0)) for i in range(gt.shape[0])]

    return psnr_list, ssim_list

def save_image(dehaze, image_name, save_path):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)
    for ind in range(batch_num):
        # 以下两种方式保存的图像，测试后的PSNR/SSIM结果是一样的
        # utils.save_image(dehaze_images[ind], filename)
        imwrite(tensor2img(dehaze_images[ind], rgb2bgr=True), save_path + '/' + image_name[ind][:-3] + 'png') # 需要从RGB图像转化为BGR进行保存，so 参数rgb2bgr=True

def validation(net, val_data_loader, device, save_tag, save_path, multi_supervised):
    psnr_list = []
    ssim_list = []
    tqdm_test = tqdm(val_data_loader, ncols=80)

    med_time = []
    net.eval()
    with torch.no_grad():
        for batch_id, val_data in enumerate(tqdm_test):
            start_time = time.perf_counter()
            haze, gt, image_name = val_data
            haze = haze.to(device)
            gt = gt.to(device)
            restore = net(haze)
            if multi_supervised:
                # restore = restore[0]
                restore = restore[3]
            torch.cuda.synchronize()#wait for CPU & GPU time syn
            evalation_time = time.perf_counter() - start_time#---------finish an image
            med_time.append(evalation_time)

            # 使用函数库 compare_ssim 计算SSIM 
            batch_psnr, batch_ssim = batch_PSNR_SSIM(restore, gt)

            psnr_list.extend(batch_psnr)
            ssim_list.extend(batch_ssim)

            # --- Save image --- #
            if save_tag:
                save_image(restore, image_name, save_path)

    median_time = statistics.median(med_time)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim, median_time


