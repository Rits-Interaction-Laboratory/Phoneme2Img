#Loss関数をまとめたpythonファイル不要なものもインポートしてるかも
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import sys
import pandas as pd
from PIL import Image
from natsort import natsorted
import unicodedata
import glob
import os
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from net import VGGFeatures
from utils import gram_matrix
def criterion_VAE(target, ave, log_dev, ave_p, log_dev_p): #VAEの再構成とKLDを含めたLoss

    num_batch=target.size(1) #バッチサイズを取得
    num_sample=target.size(0) #複数のサンプルをアウトプットしているのでその数を取得
    # --- 再構成誤差（ガウス仮定の負の対数尤度） ---
    recon_element = 0.5 * (log_dev_p+(target - ave_p)**2 / torch.exp(log_dev_p)) 
    recon_loss = recon_element.sum()
    # --- KLダイバージェンス ---
    kl_element = 0.5 * (torch.exp(log_dev) + ave**2 - 1.0 - log_dev)
    kl_loss = kl_element.sum()/num_batch

    #log σ^2→σ^2
    var=torch.exp(log_dev_p)
    #(x-μ)^2
    sq_error = (target - ave_p)**2
    nll_per=(sq_error/var).sum()
    nll_per=nll_per/(num_batch*num_sample)
    var=var.sum()/(num_batch*num_sample)
    log_var=log_dev_p.sum()/(num_batch*num_sample)
    sq_error=sq_error.sum()/(num_batch*num_sample)
    
    mse_loss=F.mse_loss(target,ave_p,reduction="sum")
    mse_loss=mse_loss/(num_batch*num_sample)
    loss = mse_loss+kl_loss  
    #lossは1サンプルの再構成誤差とKLDの値を足したものになり、それぞれのLossの値も1サンプルの誤差を表した値となる
    
    return loss,recon_loss,kl_loss,sq_error,log_var,var,nll_per,mse_loss

def criterion_PCAVAE(x, mean, logvar,z_mean,z_logvar):
    """
    多変量ガウスの対角共分散を仮定した負の対数尤度を要素単位で計算。
    -log p(x|z) ~ 0.5 * sum_over_i( ((x_i - mu_i)^2 / var_i) + log(var_i) )
    """
    var = torch.exp(logvar)
    nll_elem = (x - mean)**2 / var + logvar  # shape: (batch_size, 100)
    nll = 0.5 * torch.sum(nll_elem, dim=1)   # shape: (batch_size,)
    kl = 0.5 * torch.sum(torch.exp(z_logvar) + z_mean**2 - 1. - z_logvar, dim=1)
    loss=nll+kl
    return loss,nll,kl

def style_loss_and_diffs(recon, orig,device): #グラム行列の誤差を計算
    model=VGGFeatures().to(device)
    recon_features = model(recon)
    orig_features = model(orig)
    loss = 0
    layer_weights = [1, 1e-1, 1e-2]  # 各層に適用する重み
    diffs = {}
    criterion=nn.MSELoss()
    for layer, (rf, of) in enumerate(zip(recon_features.values(), orig_features.values())):
        diff=criterion(gram_matrix(rf),gram_matrix(of))
        # import pdb;pdb.set_trace()
        # gram_map(gram_matrix(of),layer,"origi") #グラム行列を可視化する関数
        # gram_map(gram_matrix(rf),layer,"recon") #グラム行列を可視化する関数
        diffs[f'Layer_{layer}_diff'] = diff
        loss += layer_weights[layer] * diff  # 重みを適用して損失に加算
    return loss, diffs

