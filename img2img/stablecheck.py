

import torch
import torch.nn as nn
from torchvision import models, transforms,datasets
from torch.utils.data import DataLoader, Dataset
import tqdm
from torch import optim
import sys
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import os 
import numpy as np
import cv2

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(parent_dir, 'phoneme2img'))
from pipe import StableDiffusionPipeline
from net import Encoder,Decoder,TextureNet,PromptEncoder
from dataset import ImageDataset,Lang,tensorFromSentence
from lossfunc import style_loss_and_diffs

def generate_image(image_model,prompt_converter,device,pipe,train_dataloader):
    criterion=nn.MSELoss()
    for batch_idx, (imgs, path) in tqdm.tqdm(enumerate(train_dataloader)):
        imgs=imgs.to(device)
        hidden=image_model(imgs)
        for data_num in range(train_dataloader.batch_size):
            img=imgs[data_num].to(device)
            img=img.unsqueeze(0)
            img_hidden= image_model(img)
            my_hidden=prompt_converter(img_hidden)
            my_hidden=my_hidden.to(dtype=torch.bfloat16).requires_grad_(True)
            # # -----------PromptLatentSpaceを保存
            # filename = os.path.splitext(os.path.basename(path[0]))[0]
            # dirname = os.path.basename(os.path.dirname(path[0]))
            # result = f"{dirname}/{filename}"
            # if not os.path.exists(f"image_hidden/model29/{dirname}/"):
            #     os.makedirs(f"image_hidden/model29/{dirname}/")
            # torch.save(my_hidden,f"image_hidden/model29/{result}.pt")
            #------------------
            #-----------画像to画像生成
            image,torchimage = pipe(prompt_embeds=my_hidden)
            dirname=os.path.basename(os.path.dirname(path[0]))
            if not os.path.exists(f"output/{model_save_path}/{dirname}/"):
              os.makedirs(f"output/{model_save_path}/{dirname}/")
            image_path=os.path.basename(path[0]) #拡張子あり
            image_path=os.path.splitext(image_path)[0]
            image[data_num].save(f"output/{model_save_path}/{dirname}/{image_path}.png")
            #--------------------------

            
            
            #-------------任意の画像とのグラム行列の差を計算したいとき
            # torchimage=torchimage.float()
            # torchimage=transform(torchimage).to(device)
            # recon_loss=criterion(torchimage,img)
            # img2 = Image.open("../code/dataset/image/train/あみあみ/woven_0001.jpg").convert("RGB") #img_pathの画像を開く
            # img2 = transform(img2) #transformする
            # img2=img2.unsqueeze(0)
            # img2=img2.to(device)
            # s_loss, _ = style_loss_and_diffs(torchimage, img2, image_model)
            # print(s_loss,"s_loss")
            # print(recon_loss,"recon_loss")

if __name__ == '__main__':
  with torch.no_grad():
    device = "cuda:0" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model=TextureNet().to(device)
    prompt_converter=PromptEncoder2().to(device)
    model_id = "dream-textures/texture-diffusion"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16) #from_pretrainedは/pipelines/pipelines_utils.py内で定義されているクラス
    pipe = pipe.to(device)
    num=29.1 #モデル番号
    model_save_path=f"model/prompt_converter_{num}.pth"
    imgfile=f"model/img_model_{num}.pth"
    image_model.load_state_dict(torch.load(imgfile))
    prompt_converter.load_state_dict(torch.load(model_save_path))

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_lang=ImageDataset( "texturedata/image/train/",transform)
    train_dataloader=DataLoader(train_lang,batch_size=1, shuffle=False,drop_last=True)
    generate_image(image_model,prompt_converter,device,pipe,train_dataloader)

