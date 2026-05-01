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
import gc
import cv2
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter
import datetime
import pytz
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from torchviz import make_dot
from torch.cuda.amp import autocast

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from pipe import StableDiffusionPipeline
from net import PromptEncoder,Encoder,Decoder,TextureNet,PhonemeVAE
from dataset import ImageLang,Lang,ImageLang2
from lossfunc import style_loss_and_diffs,criterion_VAE,criterion_PCAVAE
from utils import set_seed,draw_pca_plot3,select_top_k_outputs,tensorFromSentence,select_random_output,draw_pca_plot_vae_samples

from transformers import CLIPTokenizer
import pdb
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgba
from collections import defaultdict # データをグループ化するために使用


np.set_printoptions(threshold=np.inf)

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1) #s2の音素数＋1をpreviousにする

    for i, c1 in enumerate(s1): 
        current_row = [i + 1]

        # s2の各文字 c2 についてループ
        for j, c2 in enumerate(s2): 
            insertions = previous_row[j + 1] + 1 
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2) # Trueなら1をFalseなら0を返す
            current_row.append(min(insertions, deletions, substitutions))
        
        previous_row = current_row

    return previous_row[-1]

def is_close_match(s1, s2, tolerance=3): #tolerance
    return levenshtein_distance(s1, s2) <= tolerance #levenshteinの距離、値


def Train(epoch,nums,encoder,decoder,image_model,prompt_converter,phonemevae, pipe, lang,imageono_dataloader,ono_dataloader,shared_pca,common_limits,device):
    encoder.train()
    decoder.train()
    image_model.train()
    prompt_converter.train()
    phonemevae.train()

    #--------音素列復元のLoss
    train_ono_loss=0
    train_ONO_loss=0
    train_img_loss=0
    #--------画像復元のLoss
    train_s_loss=0
    train_recon_loss=0
    train_recon_hidden_loss=0
    #------二つのモーダルを近づけるLoss
    train_imgono_loss=0
    #-----VAEのLoss
    train_kl_loss=0
    train_recon_loss=0
    train_sq_error_loss=0
    train_log_var_loss=0
    train_var_loss=0
    train_nll_per_loss=0
    train_mse_loss=0

    train_total_loss=0

    score=0

    SOS_token = 0
    EOS_token = 1
    learning_rate = 1e-3
    # ono_weight=1e-6 #オノマトペ音素列復元のLossに対する重み
    ono_weight=0
    imgono_weight=1 #画像復元のLossに対する重み
    # imgono_weight=0
    # img_weight=1 #比率上げてみる
    img_weight=0
    size=64 #画像のサイズ

    criterion= nn.CrossEntropyLoss() #これをバッチサイズ分繰り返してそれをエポック回数分まわす？
    mse=nn.MSELoss()
    cos=nn.CosineEmbeddingLoss()

    encoder_optimizer = optim.Adam( encoder.parameters(), lr=learning_rate )
    decoder_optimizer = optim.Adam( decoder.parameters(), lr=learning_rate )
    image_optimizer=optim.Adam(image_model.parameters(), lr=learning_rate) #学習率は標準で1e-4
    prompt_optimizer=optim.Adam(prompt_converter.parameters(), lr=learning_rate)
    phonemevae_optimizer=optim.Adam(phonemevae.parameters(),lr=learning_rate)
    
    phoneme_iterator=iter(ono_dataloader) #オノマトペの音素のみが詰まったイテレータ
    imageono_iterator=iter(imageono_dataloader) #画像と音素がバインドされたデータセットのイテレータ
    max_iter=max(len(phoneme_iterator),len(imageono_iterator)) #max_iterは長さが大きいデータセットにあわされる今回だとimageono_dataloaderであわされる
    resize_transform=transforms.Resize((64,64))

    all_decoded = {}
    ld_values=[]
    shown_onos = set()  # すでに表示済みのオノマトペを記録
    shown_onos2 = set()
    all_img_features = []
    all_ono_features = []
    all_labels = []
    batch_losses = []
    geneimg_labels = []
    all_vae_logvars = []
    geneimg_features = []
    target_epochs = [0, 1, 2, 3,4, 5, 6, 7, 8, 9, 10, 50, 100] 

    # dataloader=imageono_dataloader
    # img_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,128))
    # mu_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,128))
    # log_var_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,128))
    # ono_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,128))
    # hidden_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,78848))
    # phoneme_hidden_numpy=np.zeros((len(dataloader),dataloader.batch_size,78848))

    os.makedirs(f"output/{nums}/train", exist_ok=True)

    for idx in tqdm.tqdm(range(max_iter)):
        batch_total_loss=0  
        
        # encoder_optimizer.zero_grad()
        # decoder_optimizer.zero_grad()
        # image_optimizer.zero_grad()
        # prompt_optimizer.zero_grad()
        phonemevae_optimizer.zero_grad()
# #--------------------------------------------------------------------------音素復元単体(imageono_dataloaderの音素列のみでは14単語しかないので297単語学習するためのコード)            
        # try:
        #     _,phoneme=next(phoneme_iterator)
        #     phoneme2_tensor=tensorFromSentence(lang,phoneme[0],EOS_token,device)
        #     encoder_hidden=encoder.initHidden().to(device)

        #     input_length  = phoneme2_tensor.size(0)  
        #     for i in range( input_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
        #         encoder_output, encoder_hidden = encoder( phoneme2_tensor[ i ], encoder_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！  
        #     # Decoder phese
        #     loss_ono = 0 #seq2seqのloss
        #     decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to(device)
        #     decoder_hidden = encoder_hidden

        #     decoded_words=[]

        #     for i in range( input_length ):
        #         decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden ) 
                        
        #         decoder_input = phoneme2_tensor[ i ] #次の音素（インデックス）をセットしておく
        #         if random.random() < 0.5: 
        #             topv, topi                     = decoder_output.topk( 1 )
        #             decoder_input                  = topi.squeeze().detach() # detach from history as input
        #         loss_ono += criterion( decoder_output, phoneme2_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算
        #         topv, topi = decoder_output.data.topk(1)

        #         if topi.item() == EOS_token: 
        #             decoded_words.append('<EOS>')
        #             all_decoded[phoneme[0]] = decoded_words.copy()
        #             break #decoder_inputの中がEOSだったらここで終了
        #         else:
        #             decoded_words.append(lang.index2word[topi.item()])

        #     #ここは精度評価---------------------------------------------------------          
        #     word=[x for x in decoded_words if x != '<EOS>']
        #     word=' '.join(word)
        #     if is_close_match(phoneme[0],word):
        #         score+=1
        #     #ここは精度評価---------------------------------------------------------  



        #     # loss_ono.backward()
        #     # encoder_optimizer.step()
        #     # decoder_optimizer.step()
        #     train_ono_loss += loss_ono.item()
        # except StopIteration:
        #     phoneme_iterator=iter(ono_dataloader)
        #     loss_ono=0
    
    #----------------------------------------------------------　音素復元
        try:
            IMG,PATH,ONO,PHONEME,IMG_HIDDEN, HIDDENPATH=next(imageono_iterator)
            all_labels.append(ONO[0])
            # images_list.append(IMG_HIDDEN.to(torch.float32).reshape(-1).detach().cpu().numpy())
            # _,phoneme=next(phoneme_iterator)
            PHONEME2_tensor=tensorFromSentence(lang,PHONEME[0],EOS_token,device)
            ENCODER_hidden=encoder.initHidden().to(device)
            INPUT_length  = PHONEME2_tensor.size(0)  

            for i in range( INPUT_length ): #INPUT_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
                ENCODER_output, ENCODER_hidden = encoder( PHONEME2_tensor[ i ], ENCODER_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！  
            
            # Decoder phese
            loss_ONO = 0 #seq2seqのloss
            DECODER_input  = torch.tensor( [ [ SOS_token ] ] ).to(device)
            DECODER_hidden = ENCODER_hidden

            decoded_words=[]
            
            for i in range( INPUT_length ):
                DECODER_output, DECODER_hidden = decoder( DECODER_input, DECODER_hidden )  
                DECODER_input = PHONEME2_tensor[ i ] #次の音素（インデックス）をセットしておく
                
                if random.random() < 0.5: 
                    topV, topI                     = DECODER_output.topk( 1 )
                    'topkはpytorchのライブラリにある。'
                    'topvはtopkで選ばれた音素、topiはその時のインデックス!!'
                    DECODER_input                  = topI.squeeze().detach() # detach from history as input
                loss_ONO += criterion( DECODER_output, PHONEME2_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算
                topV, topI = DECODER_output.data.topk(1)
                
                if topI.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    all_decoded[PHONEME[0]] = decoded_words.copy()
                    break #decoder_inputの中がEOSだったらここで終了
                else:
                    decoded_words.append(lang.index2word[topI.item()])


            # ここは精度評価---------------------------------------------------------          
            word=[x for x in decoded_words if x != '<EOS>']
            word=' '.join(word)
            if is_close_match(PHONEME[0],word):
                score+=1
            # ここは精度評価---------------------------------------------------------  

            train_ONO_loss += loss_ONO.item()


            # レーベンシュタイン距離の計算
            ld = levenshtein_distance(PHONEME[0], word)
            ld_values.append(ld)

    #-----------------------------------------------------------      画像復元  
            IMG_HIDDEN=IMG_HIDDEN[0].to(device) # IMG_HIDDEN は教師データとして使う「正解の画像特徴（ベクトル）」です。
            'IMG_HIDDENは長さcompressed=compressed/(torch.norm(compressed))のようにされてる'           
            IMG_tensor=IMG[0].to(device) #画像のテンソル
            IMG_input=IMG_tensor.view(-1,3,size,size) # バッチ1個分の画像を (B, 3, H, W) の形に整形します。
            loss_img=0 #画像のloss

            hidden=image_model(IMG_input) # 画像から特徴（hidden）を抽出。hiddenは128次元

            
            my_hidden=prompt_converter(hidden) # テキスト的なプロンプトベクトルに変換（my_hidden）。
            all_img_features.append(my_hidden.reshape(-1).detach().cpu().numpy())
            my_hidden=my_hidden.to(dtype=torch.bfloat16).requires_grad_(True) # my_hidden を勾配が流れるように設定し、型を省メモリな bfloat16 に変換。
            # print(torch.norm(my_hidden, p=2, dim=-1))
            # print("IMG_HIDDEN", IMG_HIDDEN)
            # print("IMG_gram")

            loss_img=F.mse_loss(my_hidden,IMG_HIDDEN,reduction="mean") # ここでいったん my_hidden と教師 IMG_HIDDEN との差を MSE Loss で計算。
            '長さ1なのでmseじゃなくてcosで計算。dim=-1で、最後の次元だけ計算される'
             
            'layer_normじゃないとなぜか絵がつぶれる'
            '↑prompt_converter内でLayerNormかけてるから必要ない'

            # my_hidden = torch.nn.functional.layer_norm(my_hidden, my_hidden.shape[-1:])

            # print("===IMG_HIDDEN===")
            # print("shape:", IMG_HIDDEN.shape)                # 形状（例: torch.Size([1, 77, 1024]))
            # print("mean:", torch.mean(IMG_HIDDEN).item())    # 全要素の平均
            # print("std:", torch.std(IMG_HIDDEN).item())      # 標準偏差
            # print("norm:", torch.norm(IMG_HIDDEN).item())    # ベクトル全体のL2ノルム（長さ）


            # # これは 生成するプロンプトが「正解のプロンプトベクトル」とどれだけ近いかを測る。
               
        
            # 画像をprompt_converterに通したやつの復元
            # 1. 保存したいエポックをリストに定義（0始まりなので、実際の表示エポック-1）

            # # 2. if判定を一箇所にまとめる
            # if epoch in target_epochs:
            #     ono_str = ONO[0]
                
            #     # すでに処理した単語でないかチェック
            #     if ono_str not in shown_onos2:
            #         with torch.no_grad():
            #             # # image, torch_image = pipe(prompt_embeds=my_hidden.detach()) #SDに通す
            #             # image, torch_image = pipe(prompt_embeds=my_hidden)
            #             # # print("my_hidden:",my_hidden)
            #             my_hidden=my_hidden.to(dtype=torch.float32)
            #             # geneimg_features.append(my_hidden.reshape(-1).detach().cpu().numpy())
            #             # geneimg_labels.append(ono_str)
                            
            #         # 保存ディレクトリとパス
            #             save_dir = f"output/{nums}/train/img2img"
            #             os.makedirs(save_dir, exist_ok=True) # ディレクトリがない場合に作成
            #             save_path = os.path.join(save_dir, f"epoch{epoch+1}_{ono_str}.png")

            #         # 画像を保存（pipeの返り値はリストなので [0] を取り出す）
            #         # image[0].save(save_path)
                                
            #     #         # # hidden確認
            #     #         # print("ONO          : ", ono_str)
            #     #         # print("IMAGE  PATH  : ", PATH[0])  # PATH もタプルなら [0]
            #     #         # print("HIDDEN PATH  : ", HIDDENPATH[0])  # PATH もタプルなら [0]
            #     #         # print("ENCODER_hidden: ", ENCODER_hidden)
            #             # print("Vector: ", my_hidden)
            #     #         # print("ING_input:", IMG_input)
            #     #         # print("-" * 50)
                            
            #     #         # print("img Batch", idx, "mean:", my_hidden.mean().item(), "std:", my_hidden.std().item())
            #         shown_onos.add(ono_str)


        #---------これより下はStableDiffusionに通します、計算重いです
            # image, torchimage = pipe(prompt_embeds=my_hidden) # pipe(prompt_embeds=...) に my_hidden を入れて、画像を生成（または再構成）。
            # torchimage=torchimage.to(torch.float32) # torchimage は生成された画像（Tensor形式）。
            # torchimage=resize_transform(torchimage).to(device) # サイズを調整してから、元画像（IMG_input）と比較。
            # s_loss, _ = style_loss_and_diffs(torchimage, IMG_input, device) # style_loss_and_diffs によってスタイル的な差異を数値化（s_loss）。
            # loss_img=s_loss *img_weight # loss_img をここで最終的に計算
            #  # この loss_img は「生成された画像が元画像にどれだけ似ているか（スタイル的に）」を表す損失。

            # train_s_loss +=s_loss.item() # .item() を呼ぶことで計算グラフから切り離されるので安全
            train_img_loss +=loss_img.item() # item()はloss_imgがスカラーテンソルの場合 
            batch_losses.append(loss_img.item())
            
#----------------------------------------------------------------　２つを近づける
            
            loss_imgono=0 #特徴ベクトルを近づけるloss
            
            # ENCODER_hidden = torch.nn.functional.layer_norm(ENCODER_hidden, ENCODER_hidden.shape[-1:])
            ENCODER_hidden = ENCODER_hidden.squeeze(1)
            mu_p,log_var_p,z,mu,log_var=phonemevae(ENCODER_hidden, epoch, nums, idx, ONO[0])
            #log_varが大きい→分散が大きい
           
            # 指数関数で元の分散（または標準偏差）に戻してから平均
            current_var_avg = torch.exp(log_var).mean().item()
            # print(f"平均分散（線形スケール）: {current_var_avg:.4f}")
            all_vae_logvars.append(current_var_avg)


            #batch毎のサンプルベクトルの推移-----------------------------------------------------------------------------------------------
            # if PATH[0] == 'dataset/imageono/image/train/あみあみ/woven_0001.jpg':
            if idx % 10 == 0:
                with torch.no_grad():
                    amiami_features = []
                    amiami_img_features = []
                    amiami_img_features.append(my_hidden.to(torch.float32).reshape(-1).detach().cpu().numpy())

                #     # print(epoch+1, "エポック目の", idx,"バッチ目")
                    for i in range(mu_p.size(0)):
                        single_sample = mu_p[i:i+1]
                        amiami_features.append(single_sample.to(torch.float32).reshape(-1).detach().cpu().numpy())
                    draw_pca_plot_vae_samples(epoch,nums,idx,amiami_features,amiami_img_features,pca_model=shared_pca, limits=common_limits, dir="VAEsampling_KL",mode="KL1e2_std0.2",ono=f"{ONO[0]}")

            #---------------------------------------------------------------------------------------------------------------------------

            best_outputs,my_hidden2,log_var_p2=select_top_k_outputs(my_hidden,mu_p,log_var_p,top_k=10)
            # best_outputs,my_hidden3,log_var_p2=select_random_output(mu_p,log_var_p,my_hidden)

            
            best_outputs=best_outputs.squeeze(1)
            best_outputs = torch.nn.functional.layer_norm(best_outputs, best_outputs.shape[-1:])
            
            # 1. 保存したいエポックをリストに定義（0始まりなので、実際の表示エポック-1）

            # 2. if判定を一箇所にまとめる
            if epoch in target_epochs:
                # ono_str = ONO[0]
                
            #     # すでに処理した単語でないかチェック
                # if ono_str not in shown_onos2:
                    # with torch.no_grad():
                        # ランダムに1つ抽出
                random_idx = torch.randint(0, best_outputs.size(0), (1,)).item()
                random_output_keepdim = best_outputs[random_idx : random_idx + 1]
                        
            #             # 画像生成
            #             # image2, torch_images = pipe(prompt_embeds=random_output_keepdim.detach())

            #             # 特徴量保存
            #             geneimg_features.append(random_output_keepdim.to(torch.float32).reshape(-1).detach().cpu().numpy())
            #             geneimg_labels.append(ono_str)

                all_ono_features.append(random_output_keepdim.to(torch.float32).reshape(-1).detach().cpu().numpy())


            #             # 保存ディレクトリとパス
            #             save_dir = f"output/{nums}/train/phoneme2img"
            #             os.makedirs(save_dir, exist_ok=True) # ディレクトリがない場合に作成
            #             save_path = os.path.join(save_dir, f"epoch{epoch+1}_{ono_str}.png")
                        
            #             # 画像保存
            #             # image2[0].save(save_path)
                        
                    # 処理済みリストに追加
                    # shown_onos2.add(ono_str)

            #         with open(f"/workspace/mycode/aihara/aihara/phoneme2img/output/{nums}/{nums}_batch_loss.txt", "a", encoding="utf-8") as f:
            #             if idx == 0:
            #                 f.write(f"\n== epoch {epoch+1} ==\n")
            #             f.write(f"{loss_img.item():.4f}------{PATH[0]}------{decoded_words}\n")
                        
                                
                # with torch.no_grad():  # 勾配不要
                # #     best_outputs2=best_outputs.squeeze(1)
                # #     best_outputs2=F.normalize(best_outputs2, p=2, dim=-1) 
                # #     best_outputs2 = torch.nn.functional.layer_norm(best_outputs2, best_outputs2.shape[-1:])
                #     image2, torch_images = pipe(prompt_embeds=best_outputs.detach()) #SDに通す
                            

                #     # image2がリストで返ってくる場合を考慮
                #     if isinstance(image2, list):
                #         img = image2[0]
                #     else:
                #         img = image2  # 単一画像の場合
                #     # PIL.Imageとしてそのまま保存リストに追加
                #     generated_images.append(img)

                #     if idx == 19:  # 0からカウントされるので20枚目
                #         cols, rows = 5, 4
                #         w, h = generated_images[0].size
                #         grid = Image.new('RGB', size=(cols * w, rows * h))
                #         for i, img in enumerate(generated_images):
                #             x = (i % cols) * w
                #             y = (i // cols) * h
                #             grid.paste(img, (x, y))

                #     os.makedirs(f"output/{nums}/train", exist_ok=True)
                #     save_path = os.path.join(f"output/{nums}/train", f"epoch{epoch+1}_aaa_phoneme2img.png")
                #     img.save(save_path)

            loss_imgono,recon_loss,kl_loss,sq_error,log_var,var,nll_per,mse_loss=criterion_VAE(my_hidden,mu,log_var,best_outputs,log_var_p2)
            'loss_imgono→画像の特徴ベクトルmy_hiddenとVAEの出力best_outputs2の損失cos_loss+KLloss'

            # current_kl_loss_avg = torch.exp(kl_loss).mean().item()
            # print(f"KL Loss: {current_kl_loss_avg:.4f}")

            train_recon_loss +=recon_loss.item() #バッチの値をため込む。.item()を使用しているので、計算グラフには影響しない
            train_sq_error_loss +=sq_error.item()
            train_log_var_loss +=log_var.item()
            train_var_loss += var.item()
            train_nll_per_loss+=nll_per.item()                         
            train_imgono_loss += loss_imgono.item()
            train_kl_loss+=kl_loss.item()
            train_mse_loss+=mse_loss.item()

            loss=loss_imgono*imgono_weight 
            total_loss=loss + loss_img*img_weight + loss_ONO*ono_weight
                        
            # print("imgono",(loss_imgono * imgono_weight).mean())
            # print("ono  ", loss_ONO.mean())
            # print("align", (lambda_align * align_loss).mean())
            # print("loss ", loss.mean())
            # print("img  ", (loss_img*img_weight).mean())
            # print("total", total_loss.mean())

            train_total_loss += total_loss.item()
            batch_total_loss+=total_loss #こいつは1バッチ処理するたびに0に初期化される
            total_loss=0
            loss=0

            batch_total_loss.backward()

            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(image_model.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(prompt_converter.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(phonemevae.parameters(), max_norm=1.0)

 
            # hidden_batch_numpy[idx]=my_hidden.view(1,-1).to("cpu").to(torch.float32).detach().requires_grad_(False)
            # phoneme_hidden_numpy[idx]=best_outputs.view(1,-1).to("cpu").to(torch.float32).detach().requires_grad_(False)
            # mu_batch_numpy[idx]=mu.view(-1,128).to('cpu').detach().requires_grad_(False)
            # log_var_batch_numpy[idx]=log_var.view(-1,128).to('cpu').detach().requires_grad_(False)


        except StopIteration:
            imageono_iterator=iter(imageono_dataloader)
            IMG,PATH,ONO,PHONEME,IMG_HIDDEN,HIDENPATH=next(imageono_iterator)

        # encoder_optimizer.step()
        # decoder_optimizer.step()
        # image_optimizer.step()
        # prompt_optimizer.step()
        phonemevae_optimizer.step()

        train_losses={
            "ono":         train_ono_loss/(max_iter*(imageono_dataloader.batch_size)), #ミニバッチ*イテレータの数で割ることで1データ当たりのLossの値を算出
            "ONO":         train_ONO_loss/(max_iter*(imageono_dataloader.batch_size)),
            "img":         train_img_loss/(max_iter*(imageono_dataloader.batch_size)),
            "recon_hidden":train_recon_hidden_loss/(max_iter*(imageono_dataloader.batch_size)),
            "imgono":      train_imgono_loss/(max_iter*(imageono_dataloader.batch_size)),
            "kl_loss":     train_kl_loss/(max_iter*(imageono_dataloader.batch_size)),
            "total":       train_total_loss/(max_iter*(imageono_dataloader.batch_size)),
            "recon":       train_recon_loss/(max_iter*(imageono_dataloader.batch_size)),
            "sq_error":    train_sq_error_loss/(max_iter*(imageono_dataloader.batch_size)),
            "log_var":     train_log_var_loss/(max_iter*(imageono_dataloader.batch_size)),
            "var_loss":    train_var_loss/(max_iter*(imageono_dataloader.batch_size)),
            "nll_per":     train_nll_per_loss/(max_iter*(imageono_dataloader.batch_size)),
            "mse":         train_mse_loss/(max_iter*(imageono_dataloader.batch_size)),
            # "score":       score/len(ld_values),
        }

    return train_losses, encoder,decoder,image_model,prompt_converter,phonemevae , all_decoded, ld_values, all_labels, batch_losses, all_ono_features, all_img_features, geneimg_labels, geneimg_features, all_vae_logvars, amiami_features, amiami_img_features


def Valid(encoder, decoder, image_model, prompt_converter, phonemevae, pipe, lang, imageono_valid_dataloader, ono_valid_dataloader, device):
    encoder.eval()
    decoder.eval()
    image_model.eval()
    prompt_converter.eval()
    phonemevae.eval()


    #--------音素列復元のLoss
    valid_ono_loss=0
    valid_ONO_loss=0
    valid_img_loss=0
    #--------画像復元のLoss
    valid_s_loss=0
    valid_recon_loss=0
    valid_recon_hidden_loss=0
    #------二つのモーダルを近づけるLoss
    valid_imgono_loss=0
    #-----VAEのLoss
    valid_kl_loss=0
    valid_recon_loss=0
    valid_sq_error_loss=0
    valid_log_var_loss=0
    valid_var_loss=0
    valid_nll_per_loss=0
    valid_mse_loss=0

    valid_total_loss=0

    score=0
    
    SOS_token = 0
    EOS_token = 1

    # ono_weight=1e-6
    ono_weight=0
    imgono_weight=1 #画像復元のLossに対する重み
    # imgono_weight=0
    # img_weight=1 #比率上げてみる
    img_weight=0
    size = 64

    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    cos = nn.CosineEmbeddingLoss()

    phoneme_valid_iterator=iter(ono_valid_dataloader)
    imageono_valid_iterator = iter(imageono_valid_dataloader)
    max_valid_iter=max(len(phoneme_valid_iterator),len(imageono_valid_iterator)) #max_iterは長さが大きいデータセットにあわされる今回だとimageono_dataloaderであわされる
    resize_transform=transforms.Resize((64,64))

    valid_all_decoded = {}
    valid_ld_values=[]
    valid_img_features=[]
    valid_ono_features=[]
    valid_all_labels=[]
    valid_images_list=[]

    with torch.no_grad():
        for __ in tqdm.tqdm(range(max_valid_iter)):
            try:
                # 音素列テンソル化
                IMG, PATH, ONO, PHONEME, IMG_HIDDEN,HIDDENPATH = next(imageono_valid_iterator)
                valid_all_labels.append(ONO[0])
                valid_images_list.append(IMG_HIDDEN.to(torch.float32).reshape(-1).detach().cpu().numpy())
                # _,PHONEME=next(phoneme_iterator)
                PHONEME2_tensor = tensorFromSentence(lang, PHONEME[0], EOS_token, device)
                ENCODER_hidden = encoder.initHidden().to(device)
                INPUT_length = PHONEME2_tensor.size(0)

                for i in range(INPUT_length):
                    encoder_output, ENCODER_hidden = encoder(PHONEME2_tensor[i], ENCODER_hidden)

                # デコーダーの音素列復元 loss 計算
                loss_ono = 0
                decoder_input = torch.tensor([[SOS_token]]).to(device)
                decoder_hidden = ENCODER_hidden

                valid_decoded_words  =[]


                # valid関数内のデコードループ（Teacher Forcingなしの修正版）
                for i in range(INPUT_length):
                    
                    # 1. 前のステップの入力（最初はSOSトークン、2回目以降は予測結果）をデコーダーへ
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    decoder_input= PHONEME2_tensor[i]

                    # 3. 予測: デコーダーの出力から、最も確率の高いトークン（topi）を取得
                    topv, topi = decoder_output.topk(1)
                    
                    # 4. 次の入力の設定: モデルの予測結果（topi）を次のステップの入力とする
                    # これにより、予測に基づくデコード（Greedy Decoding）が実現される
                    decoder_input = topi.squeeze().detach()
                    
                    # 2. 損失計算: このステップの出力（ロジット）と、このステップの正解ラベルで計算
                    # 損失計算の位置と引数はこれで正しい
                    loss_ono += criterion(decoder_output, PHONEME2_tensor[i]) 
                    
                    # 5. デコード結果の格納
                    # ここでtopiを格納し、EOS判定を行う
                    if topi.item() == EOS_token:
                        valid_decoded_words.append('<EOS>')
                        valid_all_decoded[PHONEME[0]] = valid_decoded_words.copy()
                        break # EOSが出たらループを抜ける
                    else:
                        valid_decoded_words.append(lang.index2word[topi.item()])

                
                #ここは精度評価---------------------------------------------------------          
                word=[x for x in valid_decoded_words if x != '<EOS>']
                word=' '.join(word)
                if is_close_match(PHONEME[0],word):
                    score+=1
                 #ここは精度評価---------------------------------------------------------  

                valid_ono_loss += loss_ono.item()
                # loss_ONO = loss_ONO * 0  # ここはTrainで0だったので0に合わせる

                ld =levenshtein_distance(PHONEME[0], word)
                valid_ld_values.append(ld)
            except StopIteration:
                imageono_valid_iterator = iter(imageono_valid_dataloader)
                # _,PHONEME=next(phoneme_iterator)
                IMG, PATH, ONO, PHONEME, IMG_HIDDEN,HIDDENPATH = next(imageono_valid_iterator)
                continue

            try:
                IMG_HIDDEN = IMG_HIDDEN[0].to(device)
                IMG_tensor = IMG[0].to(device)
                IMG_input = IMG_tensor.view(-1, 3, size, size)
                loss_img=0

                # 画像モデルの出力と特徴変換
                hidden = image_model(IMG_input)
                my_hidden = prompt_converter(hidden)
                valid_img_features.append(my_hidden.reshape(-1).detach().cpu().numpy())
                my_hidden = my_hidden.to(dtype=torch.bfloat16) # 勾配計算なし


                loss_img=F.mse_loss(my_hidden,IMG_HIDDEN,reduction="mean") # ここでいったん my_hidden と教師 IMG_HIDDEN との差を MSE Loss で計算。
                # print(loss_img)
                # # これは 生成するプロンプトが「正解のプロンプトベクトル」とどれだけ近いかを測る。

                # my_hidden = torch.nn.functional.layer_norm(my_hidden, my_hidden.shape[-1:]) # layer_normじゃないとなぜか絵がつぶれる


                #---------これより下はStableDiffusionに通します、計算重いです
                # image, torchimage = pipe(prompt_embeds=my_hidden)
                # torchimage=torchimage.to(torch.float32)
                # torchimage=resize_transform(torchimage).to(device)
                # s_loss, _ = style_loss_and_diffs(torchimage, IMG_input, device)
                # loss_img=s_loss
                # valid_s_loss +=s_loss.item()
                valid_img_loss +=loss_img.item()
                loss_imgono=0                

                # 2つのモーダルの距離を縮める VAE loss

                # ENCODER_hidden = torch.nn.functional.layer_norm(ENCODER_hidden, ENCODER_hidden.shape[-1:])
                ENCODER_hidden = ENCODER_hidden.squeeze(1)
                mu_p, log_var_p, z, mu, log_var = phonemevae(ENCODER_hidden)
                best_outputs, my_hidden2, log_var_p2 = select_top_k_outputs(my_hidden,mu_p,log_var_p,top_k=1)
                
                valid_ono_features.append(best_outputs.to(torch.float32).reshape(-1).detach().cpu().numpy())

                best_outputs=best_outputs.squeeze(1)
                best_outputs = torch.nn.functional.layer_norm(best_outputs, best_outputs.shape[-1:])

                # align_loss = F.mse_loss(my_hidden.squeeze(1).to(torch.bfloat16), mu_p.mean(dim=0).to(torch.bfloat16), reduction="mean")
                
                loss_imgono, recon_loss, kl_loss, sq_error, log_var_l, var, nll_per, mse_loss = criterion_VAE(my_hidden, mu, log_var, best_outputs, log_var_p2)

                # --- 🔽ここから追加 ---
                # # [B,77,1024] → [B,1024]
                # img_vec = F.normalize(my_hidden.mean(dim=1).to(torch.float32), dim=-1)
                # ono_vec = F.normalize(best_outputs.squeeze(0).mean(dim=1).to(torch.float32), dim=-1)

                # # cosine 類似度を最大化したい → loss = 1 - cos
                # align_loss = (1 - F.cosine_similarity(img_vec, ono_vec, dim=-1)).mean()

                # shape: [B, 77, 1024]
                # img_vec = F.normalize(my_hidden.to(torch.float32), dim=-1)
                # ono_vec = F.normalize(best_outputs.squeeze(0).to(torch.float32), dim=-1)

                # # トークン単位のコサイン類似度 → [B, 77]
                # cos = F.cosine_similarity(img_vec, ono_vec, dim=-1)

                # # 平均を取って損失に
                # align_loss = (1 - cos.mean())

                # lambda_align = 0  # 重みは調整推奨

                # --- 🔼ここまで追加 ---

                valid_recon_loss += recon_loss.item()
                valid_sq_error_loss += sq_error.item()
                valid_log_var_loss += log_var_l.item()
                valid_var_loss += var.item()
                valid_nll_per_loss += nll_per.item()
                valid_imgono_loss += loss_imgono.item()
                valid_kl_loss += kl_loss.item()
                valid_mse_loss += mse_loss.item()

                loss=loss_imgono*imgono_weight
                total_loss = loss+loss_img*img_weight+loss_ono*ono_weight # +lambda_align*align_loss

                valid_total_loss += total_loss.item()
                total_loss=0
                loss=0

            except StopIteration:
                imageono_iterator = iter(imageono_valid_dataloader)
                IMG,PATH,ONO,PHONEME,IMG_HIDDEN,HIDENPATH=next(imageono_iterator)

    # 平均を計算（バッチサイズは1の想定ならこのまま）
    valid_losses={
            "ono":         valid_ono_loss/(max_valid_iter*(imageono_valid_dataloader.batch_size)), #ミニバッチ*イテレータの数で割ることで1データ当たりのLossの値を算出
            "ONO":         valid_ONO_loss/(max_valid_iter*(imageono_valid_dataloader.batch_size)),
            "img":         valid_img_loss/(max_valid_iter*(imageono_valid_dataloader.batch_size)),
            "recon_hidden":valid_recon_hidden_loss/(max_valid_iter*(imageono_valid_dataloader.batch_size)),
            "imgono":      valid_imgono_loss/(max_valid_iter*(imageono_valid_dataloader.batch_size)),
            "kl_loss":     valid_kl_loss/(max_valid_iter*(imageono_valid_dataloader.batch_size)),
            "total":       valid_total_loss/(max_valid_iter*(imageono_valid_dataloader.batch_size)),
            "recon":       valid_recon_loss/(max_valid_iter*(imageono_valid_dataloader.batch_size)),
            "sq_error":    valid_sq_error_loss/(max_valid_iter*(imageono_valid_dataloader.batch_size)),
            "log_var":     valid_log_var_loss/(max_valid_iter*(imageono_valid_dataloader.batch_size)),
            "var_loss":    valid_var_loss/(max_valid_iter*(imageono_valid_dataloader.batch_size)),
            "nll_per":     valid_nll_per_loss/(max_valid_iter*(imageono_valid_dataloader.batch_size)),
            "mse":         valid_mse_loss/(max_valid_iter*(imageono_valid_dataloader.batch_size)),
            # "score":       score/len(valid_ld_values)
        }

    return valid_losses, valid_all_decoded , valid_ld_values, valid_img_features, valid_ono_features, valid_all_labels, valid_images_list



def main():
    # 実行時に最初に呼び出す
    set_seed(42)
    
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    nums=20101 #モデル番号
    #107はlr=1e-3,画像のオートエンコーダも一緒に学習
    #108はlr=1e-3,画像のオートエンコーダは固定
    #109はStableDiffusionに通して画像のオートエンコーダも学習
    #110は100個アウトプットを出してベストな10個の平均をバックワード,画像のオートエンコーダは固定
    #111は100個アウトプットを出してベストな30個の平均をバックワード、画像のオートエンコーダは固定
    #112は100個アウトプットを出してベストな50個の平均をバックワード、画像のオートエンコーダは固定
    #113は100個アウトプットを出してベストな80個の平均をバックワード、画像のオートエンコーダは固定
    image_model=TextureNet().to(device)
    prompt_converter=PromptEncoder().to(device)
    phonemevae=PhonemeVAE(num_samples=100).to(device)

    epochs=1
    save_loss=10000
    embedding_size = 128
    hidden_size   = 128
    phoneme_num=40 #入出力として使える音素の数=データセット内の.n_wordsに等しい
    encoder           = Encoder( phoneme_num, embedding_size, hidden_size ).to( device )
    decoder           = Decoder( hidden_size, embedding_size, phoneme_num ).to( device )
    img_size=64
    batch_size=1
    augment=True
    transform = transforms.Compose([transforms.Resize((img_size, img_size)),transforms.ToTensor()])

    # os.makedirs(f"model/{nums}", exist_ok=True)
    # os.makedirs(f"output/{nums}", exist_ok=True)
        #os.makedirs(f"model/{nums}/phonemedecoder", exist_ok=True)
        #os.makedirs(f"model/{nums}/image_model", exist_ok=True)
        #os.makedirs(f"model/{nums}/prompt_converter", exist_ok=True)
        #os.makedirs(f"model/{nums}/phonemevae", exist_ok=True)

    prompt_converter.load_state_dict(torch.load(f"/workspace/mycode/aihara/aihara/phoneme2img/model/prompt_converter.pth"))
    image_model.load_state_dict(torch.load(f"/workspace/mycode/aihara/aihara/phoneme2img/model/img_model.pth"))
    encoder.load_state_dict( torch.load( "/workspace/mycode/aihara/aihara/phoneme2img/model/phonemeencoder.pth" ) ) #読み込み
    decoder.load_state_dict( torch.load( "/workspace/mycode/aihara/aihara/phoneme2img/model/phonemedecoder.pth" ) )  

    # prompt_converter.load_state_dict(torch.load(f"/workspace/mycode/aihara/aihara/phoneme2img/model/{nums}/prompt_converter_{nums}.pth"))
    # image_model.load_state_dict(torch.load(f"/workspace/mycode/aihara/aihara/phoneme2img/model/{nums}/image_model_{nums}.pth"))
    # encoder.load_state_dict( torch.load( f"/workspace/mycode/aihara/aihara/phoneme2img/model/{nums}/phonemeencoder_{nums}.pth" ) ) #読み込み
    # decoder.load_state_dict( torch.load( f"/workspace/mycode/aihara/aihara/phoneme2img/model/{nums}/phonemedecoder_{nums}.pth" ) )  
    # phonemevae.load_state_dict(torch.load(f"/workspace/mycode/aihara/aihara/phoneme2img/model/{nums}/phonemevae_{nums}.pth"))


    #--------データセットの用意    
    lang  = Lang( 'dataset/onomatope/dictionary.csv') #オノマトペ音素の辞書
    # imageono_train_dataset=ImageLang2('dataset/newds/onomatope/train/train_image_onomatope.csv',"dataset/newds/image/train","dataset/newds/image_hidden/train",transform)
    # imageono_train_dataloader = DataLoader(imageono_train_dataset, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=0) #drop_lastをtrueにすると最後の中途半端に入っているミニバッチを排除してくれる
    imageono_train_dataset=ImageLang('dataset/imageono/onomatope/train/train_image_onomatope.csv',"dataset/imageono/image/train","dataset/image_hidden/model29/train",transform)
    imageono_train_dataloader=DataLoader(imageono_train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

    imageono_valid_dataset=ImageLang2('dataset/newds/onomatope/valid/valid_image_onomatope.csv',"dataset/imageono/image/valid5","dataset/image_hidden/model29/valid",transform)
    imageono_valid_dataloader=DataLoader(imageono_valid_dataset, batch_size=batch_size, shuffle=False,drop_last=True,num_workers=0)

    #--------------オノマトペ音素単体のデータセット    
    ono_train_dataset  = Lang( 'dataset/onomatope/dictionary.csv',augment)
    ono_train_dataloader=DataLoader(ono_train_dataset,batch_size=batch_size, shuffle=True,drop_last=True,num_workers=0)
    ono_valid_dataset=Lang('dataset/onomatope/onomatopeunknown.csv')
    ono_valid_dataloader=DataLoader(ono_valid_dataset,batch_size=batch_size,shuffle=False,drop_last=True,num_workers=0)    
    writer=SummaryWriter(log_dir=f"log/crossmodalstable_{nums}")
    all_batch_losses = []

    gc.collect()
    torch.cuda.empty_cache()

    images_list = []
    imageono_iterator=iter(imageono_train_dataloader)
    for idx in tqdm.tqdm(range(len(imageono_iterator))):
        IMG,PATH,ONO,PHONEME,IMG_HIDDEN, HIDDENPATH=next(imageono_iterator)
        images_list.append(IMG_HIDDEN.to(torch.float32).reshape(-1).detach().cpu().numpy())

    X_reference = np.array(images_list) # 全教師データの配列
    X_reference = X_reference / (np.linalg.norm(X_reference, axis=1, keepdims=True) + 1e-8)

    shared_pca = PCA(n_components=2)
    shared_pca.fit(X_reference)
        
    # 基準となる範囲(limits)を計算しておく
    ref_pca = shared_pca.transform(X_reference)
    x_min, x_max = ref_pca[:, 0].min(), ref_pca[:, 0].max()
    y_min, y_max = ref_pca[:, 1].min(), ref_pca[:, 1].max()
    margin = 0.2
    common_limits = ([x_min - margin, x_max + margin], [y_min - margin, y_max + margin])
    
    
    #---------stablediffusionのパイプラインの用意 
    model_id = "dream-textures/texture-diffusion"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16) 
    pipe = pipe.to(device)

    os.makedirs(f"model/{nums}", exist_ok=True)
    os.makedirs(f"output/{nums}", exist_ok=True)
    logfile = f"/workspace/mycode/aihara/aihara/phoneme2img/output/{nums}/{nums}_decoded_words_log.txt"

    with open(logfile, "w") as f:
        f.write("")

    for epoch in range(epochs):
       
        #--------train
        train_losses,encoder,decoder,image_model,prompt_converter,phonemevae , all_decoded , ld_values, all_labels, batch_losses, all_ono_features, all_img_features, geneimg_labels, geneimg_features, all_vae_logvars, amiami_features, amiami_img_features = Train(epoch,nums,encoder,decoder,image_model,prompt_converter,phonemevae,pipe,lang,imageono_train_dataloader,ono_train_dataloader,shared_pca,common_limits,device)

        #--------validation
        # valid_losses, valid_all_decoded ,valid_ld_values ,valid_img_features, valid_ono_features, valid_all_labels, valid_images_list= Valid(encoder, decoder, image_model, prompt_converter, phonemevae, pipe, lang, imageono_valid_dataloader, ono_valid_dataloader, device)
        

        X_reference = np.array(images_list) # 全教師データの配列
        X_reference = X_reference / (np.linalg.norm(X_reference, axis=1, keepdims=True) + 1e-8)

        shared_pca = PCA(n_components=2)
        shared_pca.fit(X_reference)

        # ami_reference = np.array(amiami_features) # 全教師データの配列
        # ami_reference = ami_reference / (np.linalg.norm(ami_reference, axis=1, keepdims=True) + 1e-8)

        # amiami_pca = PCA(n_components=2)
        # amiami_pca.fit(ami_reference)
        
        # 基準となる範囲(limits)を計算しておく
        ref_pca = shared_pca.transform(X_reference)
        x_min, x_max = ref_pca[:, 0].min(), ref_pca[:, 0].max()
        y_min, y_max = ref_pca[:, 1].min(), ref_pca[:, 1].max()
        margin = 0.2
        common_limits = ([x_min - margin, x_max + margin], [y_min - margin, y_max + margin])

        std = 0.2
        target_epochs = [0,1,2,3,4,5,6,7,8,9,10,50,100] 
        if epoch in target_epochs:
            # ---train (my_hidden|IMG_HIDDEN, generated, my_hidden, VAE|my_hidden)
            # draw_pca_plot3(epoch, nums, all_labels,       all_img_features,   images_list,        pca_model=shared_pca, limits=common_limits, dir="IMG",mode="trainimage")
            draw_pca_plot3(epoch, nums, all_labels,       all_ono_features,   all_img_features,   pca_model=shared_pca, limits=common_limits, dir="NewVAE",mode="trainVAE")
            # draw_pca_plot3(epoch, nums, geneimg_labels,   geneimg_features,   target=None,        pca_model=shared_pca, limits=common_limits, dir="VAE",mode="geneimg")
            draw_pca_plot3(epoch, nums, all_labels,       all_ono_features,   target=None,        pca_model=shared_pca, limits=common_limits, dir="NewVAE",mode="onlyVAE")
            # draw_pca_plot3(epoch, nums, all_labels,       all_ono_features,   all_img_features,   pca_model=shared_pca, limits=common_limits, dir="VAE",mode="amiami")
            # draw_pca_plot3(epoch, nums, all_labels,       all_ono_features,   all_img_features,   pca_model=shared_pca, limits=common_limits, dir="VAE",mode="shimashima")
            # draw_pca_plot_vae_samples(epoch,nums,           amiami_features,    amiami_img_features,pca_model=shared_pca, limits=common_limits, dir="VAEsampling",mode=f"{std}_amiami")
            # draw_pca_plot_vae_samples(epoch,nums,           amiami_features,    amiami_img_features,pca_model=amiami_pca, limits=None, dir="NewVAE",mode=f"{std}_onlyamiami")
            #---valid (my_hidden|IMG_HIDDEN, VAE|my_hidden)
            # draw_pca_plot3(epoch, nums, valid_all_labels, valid_img_features, valid_images_list,  pca_model=shared_pca, limits=common_limits, dir="IMG",mode="validimage")
            # draw_pca_plot3(epoch, nums, valid_all_labels, valid_ono_features, valid_img_features, pca_model=shared_pca, limits=common_limits, dir="VAE",mode="validVAE")
           
        #--------print
        # print( "[epoch num %d ] [ train_mse: %.6f] [val_mse: %.6f] [val_total_loss: %.6f]" % ( epoch+1, train_losses["imgono"], valid_losses["imgono"], valid_losses["total"]) )
        print( "[epoch num %d ] [ train_mse: %.6f]" % ( epoch+1, train_losses["imgono"]) )
        
        # #--------decoded_words  
        # if epoch % 10 == 0:

        #     #--------decoded_words
        #     output_lines = []
        #     output_lines.append(f"=== Epoch{epoch+1} decoded words ===")

        #     output_lines.append("---- Train")
        #     # ランダムに10個だけ選択（ただし最大10個）
        #     sampled_train_keys = random.sample(list(all_decoded.keys()), min(10, len(all_decoded)))
        #     for phonemet in sorted(sampled_train_keys):
        #         output_lines.append(f"phoneme: {phonemet}")
        #         output_lines.append(f"decoded: {all_decoded[phonemet]}")

        #     output_lines.append("---- Valid")
        #     sampled_valid_keys = random.sample(list(valid_all_decoded.keys()), min(10, len(valid_all_decoded)))
        #     for phoneme in sorted(sampled_valid_keys):
        #         output_lines.append(f"phoneme: {phoneme}")
        #         output_lines.append(f"decoded: {valid_all_decoded[phoneme]}")

        #     output_lines.append("=" * 40)

            # with open(logfile, "a") as f:
            #     f.write("\n".join(output_lines) + "\n")

        all_batch_losses.extend(batch_losses)

        loss_mapping = {
            "ono":       "loss/ono2",     "ONO":          "loss/ono",
            "img":       "loss/img",      "recon_hidden": "loss/recon_hidden",
            "imgono":    "loss/imgono",   "kl_loss":      "loss/kl",
            "total":     "loss/total",    "recon":        "loss/recon",
            "sq_error":  "loss/sq_error", "log_var":      "loss/log_var",
            "var_loss":  "loss/var",      "nll_per":      "loss/nll_per",
            "mse":       "loss/mse",      
            # "score":        "loss/accu"
        }

        #train
        for key, path in loss_mapping.items():
            if key in train_losses:
                writer.add_scalars(path, {'train': train_losses[key]}, epoch + 1)
        #valid
        # for key, path in loss_mapping.items():
        #     if key in valid_losses:
        #         writer.add_scalars(path, {'valid': valid_losses[key]}, epoch + 1)

        
        os.makedirs(f"model/{nums}/phonemeencoder", exist_ok=True)
        os.makedirs(f"model/{nums}/phonemedecoder", exist_ok=True)
        os.makedirs(f"model/{nums}/image_model", exist_ok=True)
        os.makedirs(f"model/{nums}/prompt_converter", exist_ok=True)
        os.makedirs(f"model/{nums}/phonemevae", exist_ok=True)

        torch.save(encoder.state_dict(), f'model/{nums}/phonemeencoder_{nums}.pth') #state_dictでencoderの状態を保存
        torch.save(decoder.state_dict(), f'model/{nums}/phonemedecoder_{nums}.pth')
        torch.save(image_model.state_dict(),f"model/{nums}/image_model_{nums}.pth")
        torch.save(prompt_converter.state_dict(),f"model/{nums}/prompt_converter_{nums}.pth")
        torch.save(phonemevae.state_dict(),f"model/{nums}/phonemevae_{nums}.pth")

    writer.close()
        

    # #1バッチごとのloss
    # plt.plot(all_batch_losses)
    # plt.xlabel("Batch")
    # plt.ylabel("loss_img")
    # plt.title("loss_img")
    # if not os.path.exists(f"figure/{nums}/train"):
    #     os.makedirs(f"figure/{nums}/train")
    # plt.savefig(f"figure/{nums}/train/batch_loss_img.png")
    # print("    plot saved: .../batch_loss_img.png")

    # #1バッチごとの分散
    # plt.plot(all_vae_logvars)
    # plt.xlabel("Batch")
    # plt.ylabel("log_var")
    # plt.title("平均分散")
    # if not os.path.exists(f"figure/{nums}/train"):
    #     os.makedirs(f"figure/{nums}/train")
    # plt.savefig(f"figure/{nums}/train/batch_logvars.png")
    # print("    plot saved: .../batch_logvars.png")

    # # ヒストグラム
    # # 3. ヒストグラムの描画
    # plt.figure(figsize=(12, 10)) # グラフのサイズを設定 (幅, 高さ)

    # # ld_values: 描画するデータ
    # # bins=range(10): 0, 1, ..., 9 の境界を持つビンを作成。これにより、0のデータは0-1のビンに、1のデータは1-2のビンに...8のデータは8-9のビンに入ります。
    # # align='left': 棒がビンの左端に揃うようにします。これにより、x軸の目盛りが棒の真下に来ます。
    # # rwidth=0.8: 棒の相対的な幅。0.8にすると棒間に隙間ができて見やすいです。
    # # color: 棒の色
    # # edgecolor: 棒の縁の色
    # # plt.hist(ld_values, alpha=0.5, bins=range(10), align='left', rwidth=0.8, color='skyblue',label='Train')
    # # plt.hist(valid_ld_values, alpha=0.5, bins=range(10), align='left', rwidth=0.8, color='r', label='Valid')
    # plt.hist([ld_values, valid_ld_values],  bins=range(18), ec='black', label=['train', 'valid'])

    # plt.legend(loc="upper right", fontsize=13) # (5)凡例表
    # # グラフのタイトルと軸ラベル
    # plt.title('Distribution of Levenshtein Distances (ld)')
    # plt.xlabel('Levenshtein Distance (ld) Value')
    # plt.ylabel('Frequency (Count)')

    # # x軸の目盛りを0から8の整数にする
    # plt.xticks(range(18)) # range(9) は 0, 1, ..., 8 を生成します

    # # グリッドの表示（任意、可視性を高めます）
    # plt.grid(axis='y', alpha=0.75) # y軸方向に薄いグリッドを表示
    # # bbox_inches='tight': 余白を自動的に調整して、グラフ全体が画像に収まるようにします。
    # if not os.path.exists(f"figure/{nums}/train"):
    #     os.makedirs(f"figure/{nums}/train")
    # plt.savefig(f"figure/{nums}/train/onohistogram.png")

    # target_numbers=range(18)
    
    # print("--- ld_values 内の要素数 ---")
    # print("ld_values: ", len(ld_values))
    # print("valid_ld_values: ", len(valid_ld_values))

    # # 0から18までの各数字についてループ
    # for number in target_numbers:
    #     count = ld_values.count(number)
    #     valid_count = valid_ld_values.count(number)
    #     print(f"ld={number}  train: {count} 個, valid: {valid_count} 個")

    print("--------------------------")
    
    # 日本時間のタイムゾーンを指定
    jst = pytz.timezone('Asia/Tokyo')
    # 現在の日本時間を取得
    now = datetime.datetime.now(jst)

    print(f"model {nums} epoch {epochs} completed (", now.strftime('%Y-%m-%d %H:%M:%S'), ")")

if __name__ == '__main__':
    main()
    

"""
num,phoneme,word
1,a m i a m i,あみあみ
2,g a ch a g a ch a,がちゃがちゃ
3,k i r a k i r a,きらきら
4,g i z a g i z a,ぎざぎざ
5,s a r a s a r a,さらさら
6,s a r a r i,さらり
7,z a r a z a r a,ざらざら
8,sh i m a sh i m a,しましま
9,sh u w a sh u w a,しゅわしゅわ
10,j i g u z a g u,じぐざぐ
11,j a r a j a r a,じゃらじゃら
12,ts u y a ts u y a,つやつや
13,d e k o b o k o,でこぼこ
14,m o y a m o y a,もやもや

"""