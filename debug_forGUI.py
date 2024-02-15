from IPython.utils import io
import torch
import PIL
import imageio
import pickle
import os
import numpy as np
import random
import ipywidgets as widgets
import matplotlib.pyplot as plt
from PIL import Image
from models import get_instrumented_model
from decomposition import get_or_compute
from config import Config
from skimage import img_as_ubyte
from ipywidgets import fixed

import glob
#import cv2



if __name__ == '__main__':
        
    # Speed up computation
    torch.autograd.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    # Specify model to use
    config = Config(
    model='StyleGAN2',
    layer='style',
    output_class='egg',
    components=80,
    use_w=True,
    n=1_000_000,
    batch_size=10_000, # style layer quite small
    )

    # config = Config(
    #   model='StyleGAN2',
    #   layer='style',
    #   output_class='ffhq',
    #   components=80,
    #   use_w=True,}
    #   n=1_000_000,
    #   batch_size=10_000, # style layer quite small
    # )


    inst = get_instrumented_model(config.model, config.output_class,
                                config.layer, torch.device('cuda'), use_w=config.use_w)

    path_to_components = get_or_compute(config, inst)

    model = inst.model
    with open(str(os.path.split(path_to_components)[0]) + '/pca_model.pkl', mode='rb') as f:
        transformer = pickle.load(f)

    named_directions = {} #init named_directions dict to save directions

    str(os.path.split(path_to_components)[0]) + '/pca_model.pkl'


    comps = np.load(path_to_components)
    lst = comps.files
    latent_dirs = []
    latent_stdevs = []
    comp_dir = []
    comp_dir_stdev = []

    load_activations = True
    for item in lst:
        if load_activations:
            if item == 'act_comp':
                for i in range(comps[item].shape[0]):
                    latent_dirs.append(comps[item][i])
        if item == 'act_stdev':
            for i in range(comps[item].shape[0]):
                latent_stdevs.append(comps[item][i])
        if item == 'act_mean':
            comp_mean = comps[item]
        else:
            if item == 'lat_comp':
                for i in range(comps[item].shape[0]):
                    latent_dirs.append(comps[item][i])
            if item == 'lat_stdev':
                for i in range(comps[item].shape[0]):
                    latent_stdevs.append(comps[item][i])



    num = 8

    for i in range(num):
        comp_dir.append(latent_dirs[i])
        comp_dir_stdev.append(latent_stdevs[i])
        
    print(f'Loaded Component No. 1~{num}')

    w = comp_mean
    out = model.sample_np()
    # print(out.shape)
    # print(out)