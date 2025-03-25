

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
from torch import optim
import sys,os
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from torchvision import models, transforms
from torchviz import make_dot
from torch.cuda.amp import autocast
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(parent_dir, 'phoneme2img'))
from dataset import ImageLang, ImageDataset
from lossfunc import style_loss_and_diffs
from net import TextureNet,Encoder,Decoder,PromptEncoder
from pipe import StableDiffusionPipeline
device = "cuda:1" 
image_model=TextureNet().to(device)
prompt_converter=PromptEncoder().to(device)

# imgfile="model/imgmodelcosine"
# image_model.load_state_dict(torch.load(imgfile))
dtype=torch.bfloat16

model_id = "dream-textures/texture-diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype) #from_pretrainedは/pipelines/pipelines_utils.py内で定義されているクラス
pipe = pipe.to(device)



transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
resize_transform=transforms.Resize((64,64))
batch_size=8
# train_lang=ImageLang( '../code/dataset/imageono/onomatope/train/train_image_onomatope.csv',"../code/dataset/imageono/image/train",transform)
# train_dataloader=DataLoader(train_lang,batch_size=batch_size, shuffle=True,drop_last=True)
# valid_lang=ImageLang( '../code/dataset/imageono/onomatope/train/train_image_onomatope.csv',"../code/dataset/imageono/image/valid",transform)
# valid_dataloader=DataLoader(valid_lang,batch_size=batch_size, shuffle=False,drop_last=True)
train_dataset=ImageDataset("texturedata/images/train",transform)
train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
valid_dataset=ImageDataset("texturedata/images/valid",transform)
valid_dataloader=DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,drop_last=True)

learning_rate=1e-3

prompt_optimizer=optim.Adam(prompt_converter.parameters(), lr=learning_rate) #学習率は標準で10e-4
img_optimizer=optim.Adam(image_model.parameters(), lr=learning_rate) #学習率は標準で10e-4
# loadnum=29
num=0 #1,2,3は学習率1e-6 1,2はStyleLossのみ 4はグラム行列を正しい方法で計算 5はバッチ平均のロスをとって学習率をe-6→e-11に変更 6はu-netの処理をcheckpointでラッピング、float32で学習、ランダムノイズではなくノイズ固定、prompt_converterの出力を20倍
#7はlearning_rate 1e-5→1e-3 8は7の続き、学習率を1e-3から1e-6にして、recon_lossの重みを1e-3→1e-2 9はデータセットをjigに変更1e-4の学習率 10は本番データセットで学習+validationを追加 11はノイズを固定
#12はbfloat16,13はprompt_converterに20倍の正規化,14 Denoising Loop is 40 15 Not using checkpoint 16 learning_rate=1e-6 17 bfloat16,checkpointなし,checkpointなし,Denoising40,ランダムノイズ,lr=1e-3,本番データ
#18はbfloat16,lr=1e-3,ランダムノイズ、本番データセット,19からweightを加えた後のグラム行列のMSE_Lossのグラフを追加 19.1 le=1e-3,19.2 lr=1e-4
#20はランダムノイズ
#21は固定していたencoderも学習
#25はVGG19のconv3_1から得られるグラム行列単一を最小化 25.1は学習率をe-4からe-5にして再学習
#26はグラム行列を3つに、Texturenetの中身が活性化関数を使ってなかったので変更
#27はTexturenetの最終層に正規化を加えた
#28はTextureNetの構造を変え、L2ノルムを1に正規化 28.1からTextureデータを増強
#29はTextureNetにLayerNormを行う作業を追加、出力はtanhで-1~1に
save=True
if save:    
    prompt_model_save_path=f"model/prompt_converter_{num}"
    img_model_save_path=f"model/img_model_{num}"
    writer = SummaryWriter(log_dir=f"log/stable{num}")
    
try:
    if loadnum is not None:  # loadnum が定義されていて None でないことを確認
        image_model.load_state_dict(torch.load(f"model/img_model_{loadnum}.pth"))
        prompt_converter.load_state_dict(torch.load(f"model/prompt_converter_{loadnum}.pth"))
except NameError:
    pass  # loadnum が定義されていない場合は何もしない

epochs=2000
w=0 #重み
criterion=nn.MSELoss()
i=0
save_loss=10000
for epoch in range(epochs):
        batch_s_loss=0
        batch_gram_loss={} #こいつには3つのグラム行列のMSEが入るので名前と値を持った辞書となる
        batch_recon_loss=0
        batch_total_loss=0
        vbatch_s_loss=0
        vbatch_gram_loss={} #こいつには3つのグラム行列のMSEが入るので名前と値を持った辞書となる
        vbatch_recon_loss=0
        vbatch_total_loss=0
        for batch_idx, (imgs, path) in enumerate(tqdm.tqdm(train_dataloader)):
            
            image_model.train()
            prompt_converter.train()
            img=imgs.to(device)

            prompt_optimizer.zero_grad()
            img_hidden= image_model(img)

            my_hidden=prompt_converter(img_hidden)
            my_hidden=my_hidden.to(dtype=dtype).requires_grad_(True)
            image, torchimage = pipe(prompt_embeds=my_hidden)
            # save_path=f"img_process/model_{num}/train/{epoch}epoch/"
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)            
            # for i in range(train_dataloader.batch_size):
            #     image_path=os.path.basename(path[i]) #拡張子あり
            #     image_path=os.path.splitext(image_path)[0] #拡張子なし
            #     image[i].save(f"{save_path}{image_path}.png")            
            torchimage2=torchimage.to(torch.float32)
            torchimage2=resize_transform(torchimage2).to(device)

            
            s_loss, gram_diffs = style_loss_and_diffs(torchimage2, img, device)
            batch_s_loss+=s_loss
            
            recon_loss=criterion(torchimage2,img)
            batch_recon_loss+=recon_loss
            recon_lossw=recon_loss*w
            
            total_loss=recon_lossw+s_loss
            batch_total_loss+=total_loss
            total_loss.backward()
            for name, diff in gram_diffs.items():
                if name not in batch_gram_loss:
                    batch_gram_loss[name]=0
                batch_gram_loss[name] +=diff.item()


            prompt_optimizer.step() 
        save_path=f"img_process/model_{num}/train/{epoch}epoch/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)            
        for i in range(train_dataloader.batch_size):
            image_path=os.path.basename(path[i]) #拡張子あり
            image_path=os.path.splitext(image_path)[0] #拡張子なし
            image[i].save(f"{save_path}{image_path}.png")  
        if save: 
            for name in batch_gram_loss:
                writer.add_scalars('GramDiffs/', {f"{name}" + "train":batch_gram_loss[name]/len(train_dataloader)}, epoch)
                if name=="Layer_0_diff":
                    writer.add_scalars('GramDiffs_addweight/', {f"{name}" + "train":batch_gram_loss[name]/(len(train_dataloader)*1)}, epoch)
                elif name=="Layer_1_diff":
                    writer.add_scalars('GramDiffs_addweight/', {f"{name}" + "train":batch_gram_loss[name]/(len(train_dataloader)*100)}, epoch)
                elif name=="Layer_2_diff":
                    writer.add_scalars('GramDiffs_addweight/', {f"{name}" + "train":batch_gram_loss[name]/(len(train_dataloader)*1000)}, epoch)
        with torch.no_grad():
                
            for batch_idx, (imgs, path) in enumerate(tqdm.tqdm(valid_dataloader)):
                
                prompt_converter.eval()
                img=imgs.to(device)
                img_hidden= image_model(img)
                my_hidden=prompt_converter(img_hidden)
                my_hidden=my_hidden.to(dtype=dtype)
                image, torchimage = pipe(prompt_embeds=my_hidden)
                # save_path=f"img_process/model_{num}/valid/{epoch}epoch/"
                # if not os.path.exists(save_path):
                #     os.makedirs(save_path)            
                # for i in range(train_dataloader.batch_size):
                #     image_path=os.path.basename(path[i]) #拡張子あり
                #     image_path=os.path.splitext(image_path)[0] #拡張子なし
                #     image[i].save(f"{save_path}{image_path}.png")  
                torchimage2=torchimage.to(torch.float32)
                torchimage2=resize_transform(torchimage2).to(device)

                
                s_loss, gram_diffs = style_loss_and_diffs(torchimage2, img, device)
                vbatch_s_loss+=s_loss
                
                recon_loss=criterion(torchimage2,img)
                vbatch_recon_loss+=recon_loss
                recon_lossw=recon_loss*w
                
                total_loss=recon_lossw+s_loss
                vbatch_total_loss+=total_loss
                for name, diff in gram_diffs.items():
                    if name not in vbatch_gram_loss:
                        vbatch_gram_loss[name]=0
                    vbatch_gram_loss[name] +=diff.item()








                if save==True:
                    torch.save(prompt_converter.state_dict(), f"{prompt_model_save_path}"+ "_checkpoint.pth")
                    torch.save(image_model.state_dict(), f"{img_model_save_path}"+ "_checkpoint.pth")
                for name in vbatch_gram_loss:
                    writer.add_scalars('GramDiffs/', {f"{name}" + "valid":vbatch_gram_loss[name]/len(valid_dataloader)}, epoch)
                    if name=="Layer_0_diff":  
                        writer.add_scalars('GramDiffs_addweight/', {f"{name}" + "valid":vbatch_gram_loss[name]/(len(valid_dataloader)*1)}, epoch)
                    elif name=="Layer_1_diff":            
                        writer.add_scalars('GramDiffs_addweight/', {f"{name}" + "valid":vbatch_gram_loss[name]/(len(valid_dataloader)*100)}, epoch)
                    elif name=="Layer_2_diff":
                        writer.add_scalars('GramDiffs_addweight/', {f"{name}" + "valid":vbatch_gram_loss[name]/(len(valid_dataloader)*1000)}, epoch)
        print("------------------------------------")
        print(f"epoch:{epoch}train_style_loss:{batch_s_loss/len(train_dataloader)}")
        # print(f"epoch:{epoch}train_recon_loss:{batch_recon_loss/len(train_dataloader)}")
        # print(f"epoch:{epoch}train_total_loss:{batch_total_loss/len(train_dataloader)}")   
        print(f"epoch:{epoch}valid_style_loss:{vbatch_s_loss/len(valid_dataloader)}")
        # print(f"epoch:{epoch}valid_recon_loss:{vbatch_recon_loss/len(valid_dataloader)}")
        # print(f"epoch:{epoch}valid_total_loss:{vbatch_total_loss/len(valid_dataloader)}")    
        if save:
            if save_loss >= vbatch_total_loss: 
                torch.save(image_model.state_dict(), f"{img_model_save_path}"+".pth")
                torch.save(prompt_converter.state_dict(), f"{prompt_model_save_path}"+".pth")
                save_loss=vbatch_total_loss
                print(f"Model saved to {prompt_model_save_path}")
                            
            writer.add_scalars('style_loss/',{"train":batch_s_loss/len(train_dataloader)},epoch)
            # writer.add_scalars('recon_loss/',{"train":batch_recon_loss/len(train_dataloader)},epoch)
            # writer.add_scalars('total_loss/',{"train":batch_total_loss/len(train_dataloader)},epoch)
            writer.add_scalars('style_loss/',{"valid":vbatch_s_loss/len(valid_dataloader)},epoch)
            # writer.add_scalars('recon_loss/',{"valid":vbatch_recon_loss/len(valid_dataloader)},epoch)
            # writer.add_scalars('total_loss/',{"valid":vbatch_total_loss/len(valid_dataloader)},epoch)                    
       
writer.close()
if save==True:
    torch.save(prompt_converter.state_dict(), f"{prompt_model_save_path}" + ".pth")



#pipeを呼び出したとき、pipelines/stable_diffusion/pipeline_stable_diffusion.py内にある__call__以下の処理が呼び出される(約777行目あたり)
    