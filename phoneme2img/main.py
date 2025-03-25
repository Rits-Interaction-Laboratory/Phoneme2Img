
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
from torchviz import make_dot
from torch.cuda.amp import autocast

from pipe import StableDiffusionPipeline
from net import PromptEncoder,Encoder,Decoder,TextureNet,PhonemeVAE
from dataset import ImageLang,Lang
from lossfunc import style_loss_and_diffs,criterion_VAE,criterion_PCAVAE
from utils import select_top_k_outputs,tensorFromSentence

def Train(encoder,decoder,image_model,prompt_converter,phonemevae,pipe,lang,imageono_dataloader,ono_dataloader,device):
    encoder.train()
    decoder.train()
    image_model.train()
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

    SOS_token = 0
    EOS_token = 1
    learning_rate = 1e-3
    ono_weight=0 #オノマトペ音素列復元のLossに対する重み
    image_weight=1 #画像復元のLossに対する重み

    size=64 #画像のサイズ
    criterion= nn.CrossEntropyLoss() #これをバッチサイズ分繰り返してそれをエポック回数分まわす？
    mse=nn.MSELoss()
    cos=nn.CosineEmbeddingLoss()

    encoder_optimizer = optim.Adam( encoder.parameters(), lr=learning_rate )
    decoder_optimizer = optim.Adam( decoder.parameters(), lr=learning_rate )
    image_optimizer=optim.Adam(image_model.parameters(), lr=learning_rate) #学習率は標準で10e-4
    prompt_optimizer=optim.Adam(prompt_converter.parameters(), lr=learning_rate)
    phonemevae_optimizer=optim.Adam(phonemevae.parameters(),lr=learning_rate)
    
    phoneme_iterator=iter(ono_dataloader) #オノマトペの音素のみが詰まったイテレータ
    imageono_iterator=iter(imageono_dataloader) #画像と音素がバインドされたデータセットのイテレータ
    max_iter=max(len(phoneme_iterator),len(imageono_iterator)) #max_iterは長さが大きいデータセットにあわされる今回だとimageono_dataloaderであわされる
    resize_transform=transforms.Resize((64,64))
    for count in tqdm.tqdm(range(len(imageono_iterator))):
        batch_total_loss=0    
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # image_optimizer.zero_grad()
        # prompt_optimizer.zero_grad()
        phonemevae_optimizer.zero_grad()
        for j in range(imageono_dataloader.batch_size): #batchサイズ分行う

    # #--------------------------------------------------------------------------音素復元単体(imageono_dataloaderの音素列のみでは14単語しかないので297単語学習するためのコード)            
    #         try:

    #             _,phoneme=next(phoneme_iterator)
    #             phoneme2_tensor=tensorFromSentence(lang,phoneme[0],EOS_token,device)
    #             encoder_hidden=encoder.initHidden().to(device)

    #             input_length  = phoneme2_tensor.size(0)  
    #             for i in range( input_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
    #                 encoder_output, encoder_hidden = encoder( phoneme2_tensor[ i ], encoder_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！  
    #             # Decoder phese
    #             loss_ono = 0 #seq2seqのloss
    #             decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to(device)
    #             decoder_hidden = encoder_hidden
    #             for i in range( input_length ):
    #                 decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden ) 
                        
    #                 decoder_input = phoneme2_tensor[ i ] #次の音素（インデックス）をセットしておく
    #                 if random.random() < 0.5: 
    #                     topv, topi                     = decoder_output.topk( 1 )
    #                     decoder_input                  = topi.squeeze().detach() # detach from history as input
    #                 loss_ono += criterion( decoder_output, phoneme2_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算
    #                 topv, topi = decoder_output.data.topk(1)

    #                 if topi.item() == EOS_token: break #decoder_inputの中がEOSだったらここで終了
    #             # loss_ono.backward()
    #             # encoder_optimizer.step()
    #             # decoder_optimizer.step()
    #             train_ono_loss += loss_ono.item()
    #         except StopIteration:
    #             phoneme_iterator=iter(ono_dataloader)
    #             loss_ono=0
    #-----------------------------------------------------------      画像復元  
            try:

                IMG,PATH,ONO,PHONEME,IMG_HIDDEN=next(imageono_iterator)
                IMG_HIDDEN=IMG_HIDDEN[0].to(device)
                IMG_tensor=IMG[0].to(device) #画像のテンソル
                IMG_input=IMG_tensor.view(-1,3,size,size)
                loss_img=0 #画像のloss


                
                hidden=image_model(IMG_input)
                my_hidden=prompt_converter(hidden)
                my_hidden=my_hidden.to(dtype=torch.bfloat16).requires_grad_(True)
                # loss_img=F.mse_loss(my_hidden,IMG_HIDDEN,reduction="sum")

            #---------これより下はStableDiffusionに通します、計算重いです
                # image, torchimage = pipe(prompt_embeds=my_hidden)
                # torchimage=torchimage.to(torch.float32)
                # torchimage=resize_transform(torchimage).to(device)
                # s_loss, _ = style_loss_and_diffs(torchimage, IMG_input, device)
                # loss_img=s_loss
                # train_s_loss +=s_loss.item()
                # train_img_loss +=loss_img.item()
                # loss_img=loss_img *image_weight
    #----------------------------------------------------------　音素復元
                PHONEME2_tensor=tensorFromSentence(lang,PHONEME[0],EOS_token,device)
                ENCODER_hidden=encoder.initHidden().to(device)
                INPUT_length  = PHONEME2_tensor.size(0)  
                for i in range( INPUT_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
                    encoder_output, ENCODER_hidden = encoder( PHONEME2_tensor[ i ], ENCODER_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！  
                # Decoder phese
                loss_ONO = 0 #seq2seqのloss
                decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to(device)
                decoder_hidden = ENCODER_hidden
                for i in range( INPUT_length ):
                    decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden ) 
                        
                    decoder_input = PHONEME2_tensor[ i ] #次の音素（インデックス）をセットしておく
                
                    if random.random() < 0.5: 
                        topv, topi                     = decoder_output.topk( 1 )
                        decoder_input                  = topi.squeeze().detach() # detach from history as input
                    loss_ONO += criterion( decoder_output, PHONEME2_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算
                    topv, topi = decoder_output.data.topk(1)

                    if topi.item() == EOS_token: break #decoder_inputの中がEOSだったらここで終了
                train_ONO_loss += loss_ONO.item()
                loss_ONO=loss_ONO*ono_weight
    #----------------------------------------------------------------　２つを近づける
                
                loss_imgono=0 #特徴ベクトルを近づけるloss

                ENCODER_hidden = ENCODER_hidden.squeeze(1)
                mu_p,log_var_p,z,mu,log_var=phonemevae(ENCODER_hidden)
                best_outputs,my_hidden,log_var_p=select_top_k_outputs(my_hidden,mu_p,log_var_p,top_k=80)
                loss_imgono,recon_loss,kl_loss,sq_error,log_var,var,nll_per,mse_loss=criterion_VAE(my_hidden,mu,log_var,best_outputs,log_var_p)
                train_recon_loss +=recon_loss.item() #バッチの値をため込む
                train_sq_error_loss +=sq_error.item()
                train_log_var_loss +=log_var.item()
                train_var_loss += var.item()
                train_nll_per_loss+=nll_per.item()                         
                train_imgono_loss += loss_imgono.item()
                train_kl_loss+=kl_loss.item()
                train_mse_loss+=mse_loss.item()
                loss=loss_imgono
                total_loss=loss+loss_img
                train_total_loss += total_loss.item()
                batch_total_loss+=total_loss #こいつは1バッチ処理するたびに0に初期化される
                total_loss=0
                loss=0
            except StopIteration:
                imageono_iterator=iter(imageono_dataloader)
        batch_total_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        # image_optimizer.step()
        # prompt_optimizer.step()
        phonemevae_optimizer.step()

    train_ono_loss=train_ono_loss/(max_iter*(imageono_dataloader.batch_size)) #ミニバッチ*イテレータの数で割ることで1データ当たりのLossの値を算出
    train_ONO_loss=train_ONO_loss/(max_iter*(imageono_dataloader.batch_size))
    train_img_loss=train_img_loss/(max_iter*(imageono_dataloader.batch_size))
    train_recon_hidden_loss=train_recon_hidden_loss/(max_iter*(imageono_dataloader.batch_size))
    train_imgono_loss=train_imgono_loss/(max_iter*(imageono_dataloader.batch_size))
    train_kl_loss=train_kl_loss/(max_iter*(imageono_dataloader.batch_size))
    train_total_loss=train_total_loss/(max_iter*(imageono_dataloader.batch_size))
    train_recon_loss=train_recon_loss/(max_iter*(imageono_dataloader.batch_size))
    train_sq_error_loss=train_sq_error_loss/(max_iter*(imageono_dataloader.batch_size))
    train_log_var_loss=train_log_var_loss/(max_iter*(imageono_dataloader.batch_size))
    train_var_loss=train_var_loss/(max_iter*(imageono_dataloader.batch_size))
    train_nll_per_loss=train_nll_per_loss/(max_iter*(imageono_dataloader.batch_size))
    train_mse_loss=train_mse_loss/(max_iter*(imageono_dataloader.batch_size))


    return train_ono_loss,train_ONO_loss,train_img_loss,train_imgono_loss,train_recon_hidden_loss,train_kl_loss,train_total_loss,train_recon_loss,train_sq_error_loss,train_log_var_loss,train_var_loss,train_nll_per_loss,train_mse_loss,encoder,decoder,image_model,prompt_converter,phonemevae


def main():

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    nums=113#モデル番号
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

    epochs=2000
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

    prompt_converter.load_state_dict(torch.load(f"model/prompt_converter.pth"))
    image_model.load_state_dict(torch.load(f"model/img_model.pth"))
    encoder.load_state_dict( torch.load( "model/phonemeencoder.pth" ) ) #読み込み
    decoder.load_state_dict( torch.load( "model/phonemedecoder.pth" ) )  
    #---------stablediffusionのパイプラインの用意 
    model_id = "dream-textures/texture-diffusion"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16) 
    pipe = pipe.to(device)

    #--------データセットの用意    
    lang  = Lang( 'dataset/onomatope/dictionary.csv') #オノマトペ音素の辞書
    imageono_train_dataset=ImageLang('dataset/imageono/onomatope/train/train_image_onomatope.csv',"dataset/imageono/image/train","dataset/image_hidden/model29",transform)
    imageono_train_dataloader = DataLoader(imageono_train_dataset, batch_size=batch_size, shuffle=True,drop_last=True) #drop_lastをtrueにすると最後の中途半端に入っているミニバッチを排除してくれる
    imageono_valid_dataset=ImageLang('dataset/imageono/onomatope/train/train_image_onomatope.csv',"dataset/imageono/image/valid","dataset/image_hidden/model29",transform)
    imageono_valid_dataloader=DataLoader(imageono_valid_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

    #--------------オノマトペ音素単体のデータセット    
    ono_train_dataset  = Lang( 'dataset/onomatope/dictionary.csv',augment)
    ono_train_dataloader=DataLoader(ono_train_dataset,batch_size=batch_size, shuffle=True,drop_last=True)
    ono_valid_dataset=Lang('dataset/onomatope/onomatopeunknown.csv')
    ono_valid_dataloader=DataLoader(ono_valid_dataset,batch_size=batch_size,shuffle=False,drop_last=True)    
    writer=SummaryWriter(log_dir=f"log/crossmodalstable_{nums}")

    for epoch in range(epochs):
        
        train_ono,train_ONO,train_img,train_imgono,train_recon_hidden_loss,train_kl,train_total,train_recon_loss,train_sq_error_loss,train_log_var_loss,train_var_loss,train_nll_per_loss,train_mse_loss,encoder,decoder,image_model,prompt_converter,phonemevae=Train(encoder,decoder,image_model,prompt_converter,phonemevae,pipe,lang,imageono_train_dataloader,ono_train_dataloader,device)
        print( "[epoch num %d ] [ train_mse: %f]" % ( epoch+1, train_imgono) )

        #trainのみ
        writer.add_scalars('loss/ono', {'train': train_ono}, epoch+1)
        writer.add_scalars('loss/img', {'train': train_img}, epoch+1)
        writer.add_scalars('loss/recon_hidden', {'train': train_recon_hidden_loss}, epoch+1)
        writer.add_scalars('loss/imgono', {'train': train_imgono}, epoch+1)
        writer.add_scalars('loss/kl', {'train': train_kl}, epoch+1)
        writer.add_scalars('loss/total', {'train': train_total}, epoch+1)
        writer.add_scalars('loss/recon', {'train': train_recon_loss}, epoch+1)
        writer.add_scalars('loss/sq_error', {'train': train_sq_error_loss}, epoch+1)
        writer.add_scalars('loss/log_var', {'train': train_log_var_loss}, epoch+1)
        writer.add_scalars('loss/var', {'train': train_var_loss}, epoch+1)
        writer.add_scalars('loss/nll_per', {'train': train_nll_per_loss}, epoch+1)
        writer.add_scalars('loss/mse', {'train': train_mse_loss}, epoch+1)
        
        if not os.path.exists(f"model/{nums}"):
            os.makedirs(f"model/{nums}")
        torch.save(encoder.state_dict(), f'model/{nums}/phonemeencoder_{nums}.pth')
        torch.save(decoder.state_dict(), f'model/{nums}/phonemedecoder_{nums}.pth')
        torch.save(image_model.state_dict(),f"model/{nums}/image_model_{nums}.pth")
        torch.save(prompt_converter.state_dict(),f"model/{nums}/prompt_converter_{nums}.pth")
        torch.save(phonemevae.state_dict(),f"model/{nums}/phonemevae_{nums}.pth")
    writer.close()
if __name__ == '__main__':
    main()
    