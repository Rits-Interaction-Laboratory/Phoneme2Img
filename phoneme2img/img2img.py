import numpy as np
from sklearn.decomposition import PCA
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
from utils import draw_pca_plot3
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
batch_size=1
# train_lang=ImageLang( '../code/dataset/imageono/onomatope/train/train_image_onomatope.csv',"../code/dataset/imageono/image/train",transform)
# train_dataloader=DataLoader(train_lang,batch_size=batch_size, shuffle=True,drop_last=True)
# valid_lang=ImageLang( '../code/dataset/imageono/onomatope/train/train_image_onomatope.csv',"../code/dataset/imageono/image/valid",transform)
# valid_dataloader=DataLoader(valid_lang,batch_size=batch_size, shuffle=False,drop_last=True)
train_dataset=ImageLang('dataset/img2img/onomatope/train/train_image_onomatope.csv',"dataset/img2img/image/train","dataset/img2img/image_hidden/train",transform)
train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
valid_dataset=ImageLang('dataset/img2img/onomatope/valid/valid_image_onomatope.csv',"dataset/img2img/image/valid","dataset/img2img/image_hidden/valid",transform)
valid_dataloader=DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,drop_last=True)

learning_rate=1e-2

prompt_optimizer=optim.Adam(prompt_converter.parameters(), lr=learning_rate) #学習率は標準で10e-4
img_optimizer=optim.Adam(image_model.parameters(), lr=learning_rate) #学習率は標準で10e-4
# loadnum=11111
num=11111 #1,2,3は学習率1e-6 1,2はStyleLossのみ 4はグラム行列を正しい方法で計算 5はバッチ平均のロスをとって学習率をe-6→e-11に変更 6はu-netの処理をcheckpointでラッピング、float32で学習、ランダムノイズではなくノイズ固定、prompt_converterの出力を20倍
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
    prompt_model_save_path=f"/workspace/mycode/aihara/aihara/phoneme2img/model/prompt_converter_{num}"
    img_model_save_path=f"/workspace/mycode/aihara/aihara/phoneme2img/model/image_model_{num}"
    writer = SummaryWriter(log_dir=f"/workspace/mycode/aihara/aihara/phoneme2img/log/stable_{num}")
    
try:
    if loadnum is not None:  # loadnum が定義されていて None でないことを確認
        image_model.load_state_dict(torch.load(f"/workspace/mycode/aihara/aihara/phoneme2img/model/{loadnum}/image_model_{loadnum}.pth"))
        prompt_converter.load_state_dict(torch.load(f"/workspace/mycode/aihara/aihara/phoneme2img/model/{loadnum}/prompt_converter_{loadnum}.pth"))
except NameError:
    pass  # loadnum が定義されていない場合は何もしない

epochs=101
w=0 #重み
criterion=nn.MSELoss()
i=0
save_loss=10000
size=64

for epoch in range(epochs):
        
        train_iterator=iter(train_dataloader)
        valid_iterator=iter(valid_dataloader)
        train_max_iter=len(train_iterator)
        valid_max_iter=len(valid_iterator)

        train_loss=0
        batch_s_loss=0
        batch_gram_loss={} #こいつには3つのグラム行列のMSEが入るので名前と値を持った辞書となる
        batch_recon_loss=0
        
        valid_loss=0
        vbatch_s_loss=0
        vbatch_gram_loss={} #こいつには3つのグラム行列のMSEが入るので名前と値を持った辞書となる
        vbatch_recon_loss=0
        
        shown_onos = set()
        all_img_features = []
        valid_img_features = []
        images_list = []
        valid_images_list = []
        all_labels = []
        valid_all_labels = []

        #---Train---
        image_model.train()
        prompt_converter.train() 

        for batch_idx in tqdm.tqdm(range(train_max_iter)):  

            prompt_optimizer.zero_grad()
            
            img,path,ono,phoneme,img_hidden,hiddenpath=next(train_iterator)

            all_labels.append(ono[0])
            images_list.append(img_hidden.to(torch.float32).reshape(-1).detach().cpu().numpy())
                
            img_hidden=img_hidden[0].to(device)
            img_tensor=img.to(device)
            img_input=img_tensor.view(-1,3,size,size)
                
            loss_img=0

            hidden= image_model(img_input)
            my_hidden=prompt_converter(hidden)
            all_img_features.append(my_hidden.reshape(-1).detach().cpu().numpy())
            my_hidden=my_hidden.to(dtype=dtype)

            loss_img=F.mse_loss(my_hidden,img_hidden,reduction="mean")


            ono_str = ono[0]
            # if epoch < 5:
            #     if ono_str not in shown_onos:
            #         with torch.no_grad():  # 勾配不要
            #             image2, torch_images = pipe(prompt_embeds=my_hidden.detach()) #SDに通す
            #             # 保存ファイル名を決定
            #         save_path = os.path.join(f"output/{num}/hidden2img", f"{ono[0]}.png") # trainingdata
                                    
            #                     # 画像を保存（pipeの返り値はリストなので [0] を取り出す）
            #         image2[0].save(save_path)
            #         shown_onos.add(ono_str)

            # else:
            if epoch % 100 ==0:
                if ono_str not in shown_onos:
                    with torch.no_grad():
                        image2, torch_images = pipe(prompt_embeds=my_hidden.detach()) #SDに通す
                    # 保存ファイル名を決定
                    save_path = os.path.join(f"output/{num}", f"epoch{epoch+1}_{ono[0]}.png") # trainingdata
                                        
                    # 画像を保存（pipeの返り値はリストなので [0] を取り出す）
                    image2[0].save(save_path)

                    with open(f"/workspace/mycode/aihara/aihara/phoneme2img/output/{num}/{num}_img2img_batch_loss.txt", "a", encoding="utf-8") as f:
                        if batch_idx == 0:
                            f.write(f"\n== epoch {epoch+1} ==\n")
                        f.write(f"{loss_img.item():.4f}------{path[0]}------{ono[0]}\n")
                        
                    shown_onos.add(ono_str)
                                
            

            loss_img.backward()
            prompt_optimizer.step() 
            train_loss+=loss_img.item()


        #---Valid---
        prompt_converter.eval()

        with torch.no_grad():
            for batch_idx in tqdm.tqdm(range(valid_max_iter)):
                vbatch_total_loss=0

                img,path,ono,phoneme,img_hidden,hiddenpath=next(valid_iterator)

                valid_all_labels.append(ono[0])
                valid_images_list.append(img_hidden.to(torch.float32).reshape(-1).detach().cpu().numpy())
                  
                img_hidden=img_hidden[0].to(device)
                img_tensor=img.to(device)
                img_input=img_tensor.view(-1,3,size,size)


                hidden= image_model(img_input)
                my_hidden=prompt_converter(hidden)
                valid_img_features.append(my_hidden.reshape(-1).detach().cpu().numpy())
                my_hidden=my_hidden.to(dtype=dtype)

                valid_loss_img=F.mse_loss(my_hidden,img_hidden,reduction="mean")
                valid_loss+=valid_loss_img.item()

        X_reference = np.array(images_list) # 全教師データの配列
        X_reference = X_reference / (np.linalg.norm(X_reference, axis=1, keepdims=True) + 1e-8)

        shared_pca = PCA(n_components=2)
        shared_pca.fit(X_reference)

        # 基準となる範囲(limits)を計算しておく
        ref_pca = shared_pca.transform(X_reference)
        x_min, x_max = ref_pca[:, 0].min(), ref_pca[:, 0].max()
        y_min, y_max = ref_pca[:, 1].min(), ref_pca[:, 1].max()
        margin = 0.2
        common_limits = ([-0.3,0.4], [-0.2,0.6])


        if epoch < 5:
            #utils.pyのsave_path変える！！
            draw_pca_plot3(epoch, num, all_labels,       all_img_features,   images_list,        pca_model=shared_pca, limits=common_limits, dir="",mode="trainimage")
            draw_pca_plot3(epoch, num, all_labels,       all_img_features,   target=None,        pca_model=shared_pca, limits=common_limits, dir="",mode="my")

            draw_pca_plot3(epoch, num, valid_all_labels, valid_img_features, valid_images_list,  pca_model=shared_pca, limits=common_limits, dir="",mode="validimage")
           
        else:
            if epoch % 10 == 0:
                # myPCA(epoch, nums,all_img_features,images_list,all_ono_features,all_labels)
                draw_pca_plot3(epoch, num, all_labels,       all_img_features,   images_list,        pca_model=shared_pca, limits=common_limits, dir="",mode="trainimage")
                draw_pca_plot3(epoch, num, all_labels,       all_img_features,   target=None,        pca_model=shared_pca, limits=common_limits, dir="",mode="my")

                draw_pca_plot3(epoch, num, valid_all_labels, valid_img_features, valid_images_list,  pca_model=shared_pca, limits=common_limits, dir="",mode="validimage")

        print("------------------------------------")
        print(f"epoch:{epoch} train_loss:{train_loss/len(train_dataloader)}")
        # print(f"epoch:{epoch}train_recon_loss:{batch_recon_loss/len(train_dataloader)}")
        # print(f"epoch:{epoch}train_total_loss:{batch_total_loss/len(train_dataloader)}")   
        print(f"epoch:{epoch} valid_loss:{valid_loss/len(valid_dataloader)}")
        # print(f"epoch:{epoch}valid_recon_loss:{vbatch_recon_loss/len(valid_dataloader)}")
        # print(f"epoch:{epoch}valid_total_loss:{vbatch_total_loss/len(valid_dataloader)}")    
        if save:
            torch.save(image_model.state_dict(), f"model/{num}/image_model_{num}.pth")
            torch.save(prompt_converter.state_dict(), f"model/{num}/prompt_converter_{num}.pth")
            print(f"Model saved")
                            
            writer.add_scalars('loss/IMG2',{"train":train_loss/len(train_dataloader)},epoch)
            writer.add_scalars('loss/IMG2',{"valid":valid_loss/len(valid_dataloader)},epoch)

writer.close()
if save==True:
    torch.save(prompt_converter.state_dict(), f"model/{num}/prompt_converter_{num}.pth")



#pipeを呼び出したとき、pipelines/stable_diffusion/pipeline_stable_diffusion.py内にある__call__以下の処理が呼び出される(約777行目あたり)
    