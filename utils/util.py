import os, subprocess
import numpy as np
from matplotlib import pyplot as plt
import torch
import cv2

import matplotlib.pyplot as plt
import pickle

with open('/home/lh/lihao/FaceDiffuser-main/data/damm_rig_equal/scaler_192.pkl', 'rb') as f:
    RIG_SCALER = pickle.load(f)


def plot_losses(train_losses, val_losses, save_name="losses"):
    print(train_losses)
    print(val_losses)
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Losses")
    plt.savefig(f"{save_name}.png")
    plt.close()


def create_gaussian_diffusion(args):
    # default params
    sigma_small = True
    predict_xstart = False  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diff_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule("cosine", steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def imagesc(nd_array):
    plt.figure(figsize=(10, 10))
    plt.imshow(nd_array)
    plt.colorbar()
    plt.show()


def imread(img):
    # return RGB image
    try:
        im = cv2.imread(img)
    except Exception as e:
        print(e)
        return None
    im = im[..., ::-1] / 255.
    return im


def imwrite(img, name):
    # write RGB image, img is in RGB format
    cv2.imwrite(name, img[..., ::-1] * 255.)


def stereoRead(img: str):
    im = imread(img)
    H, W, _ = im.shape
    assert W % 2 == 0, '%s is not a side-by-side stereo image' % img
    left = im[:, :W // 2, ...]
    right = im[:, W // 2:, ...]
    return left, right


def stereoWright(left: np.ndarray, right: np.ndarray, name: str):
    im = np.concatenate((left, right), axis=1)
    imwrite(im, name)


def exeCmd(cmd=None):
    print(cmd)
    return subprocess.check_call(cmd, shell=True)


def img2tensor(img, cuda=True):
    img_t = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img_t = torch.from_numpy(img_t.astype(np.float32))
    if cuda:
        img_t = img_t.cuda(non_blocking=True)
    return img_t


def tensor2img(img_t):
    if len(img_t.shape) == 4:
        img = img_t[0].detach().cpu().numpy()
    elif len(img_t.shape) == 3:
        img = img_t.detach().cpu().numpy()
    else:
        raise NotImplementedError
    img = img.transpose(1, 2, 0)
    return img


def cal_padding(L, window_size):
    padding_left, padding_right = 0, 0
    if L%window_size != 0:
        total = window_size-L%window_size
        padding_left = total//2
        padding_right = total - padding_left
    return padding_left, padding_right

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
