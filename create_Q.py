from IPython.utils import io
import torch
import cv2
import PIL
import imageio
import pickle
import os
import numpy as np
import random
import ipywidgets as widgets
import matplotlib.pyplot as plt
import datetime
from PIL import Image
from models import get_instrumented_model
from decomposition import get_or_compute
from config import Config
from skimage import img_as_ubyte
from ipywidgets import fixed
from pathlib import Path


if __name__ == '__main__':
    global max_batch, sample_shape, feature_shape, inst, args, layer_key, model

    # Load config
    args = Config().from_args()
    t_start = datetime.datetime.now()
    timestamp = lambda : datetime.datetime.now().strftime("%d.%m %H:%M")
    print(f'[{timestamp()}] {args.model}, {args.layer}, {args.estimator}')

    # Ensure reproducibility
    torch.manual_seed(0) # also sets cuda seeds
    np.random.seed(0)

    # Speed up backend
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_grad_enabled(False)

    # Load model
    inst = get_instrumented_model(args.model, args.output_class,
                              args.layer, torch.device('cuda'), use_w=args.use_w)
    path_to_components = get_or_compute(args, inst,force_recompute=True)#Path("/home/taki/ganspace/cache/components/stylegan2-egg_style_ipca_c80_n300000_w.npz")#
    model = inst.model
    with open(str(os.path.split(path_to_components)[0]) + '/pca_model.pkl', mode='rb') as f:
        transformer = pickle.load(f)

    # Load components
    print("path_to_components: ", path_to_components)
    print("m", args.m)
    comps = np.load(path_to_components)
    lst = comps.files
    latent_dirs = []
    latent_stdevs = []

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
                latent_mean = comps[item]

        else:
            if item == 'lat_comp':# 主成分軸
                for i in range(comps[item].shape[0]):
                    latent_dirs.append(comps[item][i])
            if item == 'lat_stdev':
                for i in range(comps[item].shape[0]):
                    latent_stdevs.append(comps[item][i])
            if item == 'lat_mean':
                latent_mean = comps[item]

    #w_dir = np.array(latent_dirs)

    # for i in range(args.m):
    #     for j in range(5): # 主成分の数
    #         rand = random.uniform(-2*latent_stdevs[0], 2*latent_stdevs[0])
    #         if(j==0): w = latent_mean + rand * w_dir[j]
    #         else: w = w + rand * w_dir[j]
    #     out = model.sample_np(w)
    #     img = Image.fromarray((out * 255).astype(np.uint8)).resize((256,256),Image.LANCZOS)
    #     img.save(f'out/StyleGAN2-pre-process-final/img/{i:05}.png')

    #latents = model.sample_latent(args.m, seed=0).cpu().numpy()
    #w = np.empty((latents.shape[0], latents.shape[1]), dtype=np.float32)
    w = np.array([latent_mean[0].copy() for _ in range(args.m)]) # 潜在空間の平均座標をM個用意
    w_dir = np.array(latent_dirs)
    for i in range(args.m):
        for j in range(5):
            rand = random.uniform(-2*latent_stdevs[j], 2*latent_stdevs[j])
            w[i:i+1] = w[i:i+1]+rand*w_dir[j]
        out = model.sample_np(w[i:i+1])
        img = Image.fromarray((out * 255).astype(np.uint8)).resize((256,256),Image.LANCZOS)
        img.save(f'/home/taki/ganspace/gen_img_lat-mean-to-comp_-2to2sigma_batch64x4_310k-iter_fid=38-30437_compnum=5/{i:05}.png')

    param = transformer.transform(w)
    
    np.save('/home/taki/ganspace/gen_img_lat-mean-to-comp_-2to2sigma_batch64x4_310k-iter_fid=38-30437_compnum=5/w', w)
    np.save('/home/taki/ganspace/gen_img_lat-mean-to-comp_-2to2sigma_batch64x4_310k-iter_fid=38-30437_compnum=5/param', param)
    np.save('/home/taki/ganspace/gen_img_lat-mean-to-comp_-2to2sigma_batch64x4_310k-iter_fid=38-30437_compnum=5/pca_score', param[:,:5])

    print('Done in', datetime.datetime.now() - t_start)

