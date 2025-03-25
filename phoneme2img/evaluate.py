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
import torchvision
from torchvision import models, transforms,datasets
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from torch.utils.data.dataset import Subset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.collections as mc
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from pykakasi import kakasi
import keyboard

from pipe import StableDiffusionPipeline
from lossfunc import style_loss_and_diffs
from net import Encoder,Decoder,PromptEncoder,PhonemeVAE,TextureNet
from dataset import ImageLang,Lang,tensorFromSentence

#-------音素列生成の評価
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1) #s2の音素数＋1をpreviousにする
    for i, c1 in enumerate(s1): #s1の各音素がどうなるかs2の各音素でチェックする
        current_row = [i + 1]

        for j, c2 in enumerate(s2): #s1の音素（C1)に対してs2は何をしたらどれくらいコストがかかるのか表示する
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2) #Trueなら1をFalseなら0を返す

            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
def is_close_match(s1, s2, tolerance=2):
    return levenshtein_distance(s1, s2) <= tolerance
def calc_accu(encoder,decoder,dataloader,lang,EOS_token,device):
    score=0
    count=0
    for batch_num,(ono,phoneme) in enumerate(dataloader):  
        for data_num in range(dataloader.batch_size):           
            ono_word,encoder_hidden=ono_to_ono(phoneme[data_num],encoder,decoder,lang,EOS_token,device)

            word=[x.replace("<EOS>","") for x in ono_word]
            word=[x+' 'for x in word] #1音素ずつに半角の空白を追加
            word[-1]=word[-1].strip() #最後の音素の後ろの空白だけ消す
            word=''.join(word) #リストになってたものを１つの単語にする
 
            if is_close_match(phoneme[data_num],word):
                score+=1

            count+=1
    return (score/count)
def calc_img2ono_accu(image_model,decoder,dataloader,lang,EOS_token,device):
    score=0
    count=0
    for batch_num,(img,_,_,phoneme,_) in enumerate(dataloader):  
        for data_num in range(dataloader.batch_size):           
            img_word,_=img_to_ono(img[data_num],image_model,decoder,lang)
            
            word=[x.replace("<EOS>","") for x in img_word]
            word=[x+' 'for x in word] #1音素ずつに半角の空白を追加
            word[-1]=word[-1].strip() #最後の音素の後ろの空白だけ消す
            word=''.join(word) #リストになってたものを１つの単語にする

            if is_close_match(phoneme[data_num],word):
                score+=1
            count+=1
    return (score/count)
#-----------------------------------------------

def img_to_ono(img,image_model,decoder,lang,SOS_token,EOS_token,device):
    img_input=img.view(-1,3,size,size).to(device)
    img_hidden=image_model(img_input)   
    img_hidden=img_hidden.view(1,1,128) #画像の特徴ベクトルをオノマトペの隠れベクトルの大きさに合わせる
    decoder_input      = torch.tensor([[SOS_token]], device=device)  # SOS
    decoder_hidden     = img_hidden
    decoded_words      = []
    max_length=20
    for di in range(max_length):
        decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden )
        topv, topi = decoder_output.data.topk(1) #topiはアウトプットの中から最も確率の高いラベル（音素のインデックス番号）取り出す
        if topi.item() == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[topi.item()]) 

        decoder_input = topi.squeeze().detach()
    return decoded_words,img_hidden
def ono_to_ono(sentence,encoder,decoder,lang,SOS_token,EOS_token,device):
    max_length=20
    input_tensor   = tensorFromSentence(lang, sentence,EOS_token,device)
    input_length   = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden().to(device)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
    decoder_input      = torch.tensor([[SOS_token]], device=device)  # SOS
    decoder_hidden     = encoder_hidden
    decoded_words      = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden )

            
        topv, topi = decoder_output.data.topk(1)
        if topi.item() == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[topi.item()])

        decoder_input = topi.squeeze().detach()
    return decoded_words,encoder_hidden



def generate_stable_images(encoder,pipe,prompt_converter,phonemevae,lang,sentence,step,model_num,EOS_token):
    encoder_hidden_numpy=None
    for i in range(step):
        input_tensor   = tensorFromSentence(lang, sentence,EOS_token,device)
        input_length   = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden().to(device)
        
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_hidden=encoder_hidden.view(-1,128)  
        my_hidden,_,_,_,_=phonemevae(encoder_hidden)
        my_hidden=my_hidden.to(dtype=torch.bfloat16).requires_grad_(False).squeeze(0)
        image,torchimage = pipe(prompt_embeds=my_hidden)
        if not os.path.exists(f"output/{model_num}/"):
            os.makedirs(f"output/{model_num}/")
        image[0].save(f"output/{model_num}/{sentence}{i}.png")
        encoder_hidden=encoder_hidden.view(1,128).to('cpu').numpy()  
    # 連結処理 PhonemeEncoderから得られる128次元の特徴ベクトルを返す
        if encoder_hidden_numpy is None:
            encoder_hidden_numpy = encoder_hidden
        else:
            encoder_hidden_numpy = np.concatenate((encoder_hidden_numpy, encoder_hidden), axis=0)
    return encoder_hidden_numpy
                   
def calc_gram(image_model,device): #2つのグラム行列の差を計算する関数
    gram_total=0
    img1_path='dataset/グラム行列計算用生成画像/train/s a r a s a r a.jpg' #指定画像（音素からの生成画像等）
    img1=Image.open(img1_path).convert("RGB")
    img1=transform(img1).unsqueeze(0).to(device)
    # img2_path='dataset/グラム行列計算用生成画像/train/s a r a r i.jpg' #指定画像（音素からの生成画像等）
    # img2=Image.open(img2_path).convert("RGB")
    # img2=transform(img2).unsqueeze(0).to(device)
    # gram_diff,_=style_loss_and_diffs(img1,img2,image_model)
    # print(gram_diff)
    img2_paths=glob.glob('dataset/imageono/image/train/あみあみ/*')
    gram_num=len(img2_paths)
    for img2_path in img2_paths:
        img2=Image.open(img2_path).convert("RGB") #オリジナル画像
        img2=transform(img2).unsqueeze(0).to(device)

        img3,_=image_model(img2) #画像からの生成画像

        gram_diff,_=style_loss_and_diffs(img1,img2,image_model)
        gram_total +=gram_diff

    gram_average=gram_total/gram_num
    print(gram_average)
    return 




def evaluate(encoder,decoder,image_model,pipe,prompt_converter,phonemevae,nums,device):
    with torch.no_grad():
        SOS_token = 0
        EOS_token = 1
        batch_size =1
        imgono_loss=0
        max_length=20
        size=64
        cosine_dict={}
        wordcount=0
        #データセットの準備
        transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
        lang=Lang('dataset/onomatope/dictionary.csv')
        train_lang  = ImageLang( 'dataset/imageono/onomatope/train/train_image_onomatope.csv',"dataset/imageono/image/train","dataset/image_hidden/model29",transform)
        valid_lang  = ImageLang( 'dataset/imageono/onomatope/train/train_image_onomatope.csv',"dataset/imageono/image/valid","dataset/image_hidden/model29",transform)
        
        train_dataloader = DataLoader(train_lang, batch_size=batch_size, shuffle=False,drop_last=True) #drop_lastをtruenにすると最後の中途半端に入っているミニバッチを排除してくれる
        valid_dataloader=DataLoader(valid_lang,batch_size=batch_size, shuffle=False,drop_last=True)

        ono_train_dataloader=DataLoader(lang,batch_size=batch_size,shuffle=False,drop_last=True)    

        criterion=nn.MSELoss()
        cos=nn.CosineEmbeddingLoss()
        cos2=nn.CosineSimilarity()

        word_list=[]
        path_list=[]
        
        dataloader=train_dataloader
        img_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,128))
        valid_img_batch_numpy=np.zeros((len(valid_dataloader),valid_dataloader.batch_size,128))
        mu_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,128))
        log_var_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,128))
        ono_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,128))
        hidden_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,78848))
        phoneme_hidden_numpy=np.zeros((len(dataloader),dataloader.batch_size,78848))
        pca_phoneme_batch_numpy=np.zeros((len(dataloader),dataloader.batch_size,100))
        count=1
        while True:
            # ユーザーに入力を求める
            sentence = input("input onomatope_phoneme（q=exit）: ")

            # 'Q' が押されたらループを抜ける
            if sentence.lower() == "q":
                break

            # 画像生成関数を実行
            _ = generate_stable_images(encoder, pipe, prompt_converter, phonemevae, lang, 
                                    sentence=sentence, step=10, model_num=nums, EOS_token=EOS_token)
        for batch_num,(img,path,ono,phoneme,IMG_HIDDEN) in tqdm.tqdm(enumerate(dataloader),total=len(dataloader)): #プログレスバーあり
            for data_num in range(dataloader.batch_size):      
                IMG_HIDDEN=IMG_HIDDEN[data_num].to(device)
                img_data=img[data_num].to(device)
                img_data=img_data.view(-1,3,size,size)
                output_batch=image_model(img_data)
                my_hidden=prompt_converter(output_batch)
                my_hidden=my_hidden.to(dtype=torch.bfloat16)          
                
        
                img_word,img_hidden=img_to_ono(img[data_num],image_model,decoder,lang,SOS_token,EOS_token,device) #画像の隠れベクトルからオノマトペを生成             
                ono_word,encoder_hidden=ono_to_ono(phoneme[data_num],encoder,decoder,lang,SOS_token,EOS_token,device)
                encoder_hidden=encoder_hidden.squeeze(0)
                mu_p,log_var_p,z,mu,log_var=phonemevae(encoder_hidden)

                word_list.append(ono[data_num]) #オノマトペの単語をリストにアペンド（主成分分析のラベルとして使う）
                
                #画像のPathのリストを作成
                filename = path[data_num].split('/')[-1]  # ファイル名を取得
                basename = filename.split('.')[0]  # 拡張子を除いた名前を取得
                basenum=basename.split("_")[-1]
                path_list.append(basenum)
                
    
                # print(img_word,ono[data_num],ono_word,phoneme[data_num],path[data_num]) #画像特徴からオノマトペの音素とオノマトペの音素からオノマトペの音素の結果を表示
                #--------------------------------------
                if ono[data_num] not in cosine_dict:
                    cosine_dict[ono[data_num]]=[]
                    wordcount+=1 #wordをカウントしておくデータセットを変更しなければ14単語になるはず
                #-------------------------------------------



            
            img_batch_numpy[batch_num]=img_hidden.view(-1,128).to('cpu').numpy() 
            ono_batch_numpy[batch_num]=encoder_hidden.view(-1,128).to('cpu').numpy()  
            hidden_batch_numpy[batch_num]=my_hidden.view(1,-1).to("cpu").to(torch.float32).numpy()
            phoneme_hidden_numpy[batch_num]=mu_p.view(1,-1).to("cpu").to(torch.float32).numpy()
            mu_batch_numpy[batch_num]=mu.view(-1,128).to('cpu').numpy()   
            log_var_batch_numpy[batch_num]=log_var.view(-1,128).to('cpu').numpy()   
            
            
        
   
    
        # print(calc_accu(encoder,decoder,ono_train_dataloader,lang,EOS_token,device)) #3番目の引数のデータセットから音素to音素のaccuracyを測定
        # print(calc_img2ono_accu(image_model,decoder,dataloader,lang,EOS_token,device)) #3番目の引数のデータセットから画像to音素のaccuracyを測定
        

        myPCA(hidden_batch_numpy,phoneme_hidden_numpy,word_list,path_list,10,wordcount) #78000次元の特徴ベクトルをプロット
        generate_VAE_PCA(mu_batch_numpy,log_var_batch_numpy,word_list) #VAEから得られる潜在変数をプロット、3σ区間も楕円としてプロットする
    return 

def generate_VAE_PCA(mu,log_var,word_list): #μとσを埋め込み、どこを中心としてどれくらいの幅で埋め込まれる可能性があるのか描画
    kakasiobj=kakasi()
    kakasiobj.setMode("H","a")
    converter=kakasiobj.getConverter()
    romaji_list = [converter.do(text) for text in word_list]    
    # ---------------------------------------
    # 1. PCAを使って潜在空間の平均ベクトル(mu)を2次元に写像
    # ---------------------------------------
    pca =PCA(n_components=2) 
    mu=mu.squeeze(1)
    log_var=log_var.squeeze(1)
    mu_2d = pca.fit_transform(mu)
    W = pca.components_
    fig, ax = plt.subplots(figsize=(8, 6))
    color1 =["red","yellow", "gray","silver","rosybrown","firebrick",
            "darksalmon","sienna","sandybrown","tan",
                "gold","olivedrab","chartreuse","palegreen",
                "darkgreen","lightseagreen","paleturquoise",
                "deepskyblue","blue","pink","orange","crimson",
                "mediumvioletred","plum","darkorchid","mediumpurple",
                "chocolate","peru","yellow","y","aqua","lightsteelblue","linen","teal"]
    # 散布図（平均ベクトルを2次元にした点）
    startidx=0
    onelabelcount=100
    for i in range(14):
        ax.scatter(mu_2d[startidx:startidx+onelabelcount, 0], mu_2d[startidx:startidx+onelabelcount, 1],c=color1[i],label=romaji_list[i*100],marker="$x$")#onomatopoeia        
        startidx+=onelabelcount
    # ax.scatter(mu_2d[:, 0], mu_2d[:, 1], c='blue', s=10, alpha=0.6, label=word_list)

    for i in range(1400):
        # 潜在空間での分散 = exp(logvar)
        # まずは対角行列として共分散行列を作る
        var_diag = np.exp(log_var[i])  # shape: (D,)
        Sigma = np.diag(var_diag)                      # shape: (D, D)
        # PCA空間への線形変換: Cov2D = W * Sigma * W^T
        # ここでWは shape: (2, D) なので (2, D) x (D, D) x (D, 2) => (2, 2)
        Cov2D = W @ Sigma @ W.T

        center_2d = mu_2d[i]  # shape: (2,)
        plot_ellipse(ax, center_2d, Cov2D, edge_color='red', face_color='none', alpha=0.3)

    ax.legend()
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('VAE latent space (mu) embedded by PCA with uncertainty ellipse')
    ax.grid(True)
    # plt.scatter(0,0,c="black",marker="o")
    plt.xlim([-7.5,10])
    plt.ylim([-4,6]) 
    fig.tight_layout()
    if not os.path.exists(f"figure/{nums}"):
        os.makedirs(f"figure/{nums}")
    fig.savefig(f"figure/{nums}/VAEtrainimagehiddenPCA.png")
    plt.show()
def plot_ellipse(ax, center, cov2d, edge_color='black', face_color='none', alpha=1.0):# 2. 楕円を描画するための関数定義
    """
    center: [x, y] (楕円の中心)
    cov2d:  2x2 の共分散行列
    """
    # 共分散行列の固有分解
    eigenvals, eigenvecs = np.linalg.eigh(cov2d)
    
    # 固有値の平方根が半径に相当（ここでは「標準偏差分の楕円」を描画）
    # もし「何σ区間」の楕円を描画したい場合は係数をかける
    # 例) 2σ楕円 => 2 * np.sqrt(eigenvals)
    r1, r2 = np.sqrt(eigenvals)
    r1 *=3.0
    r2 *=3.0
    # 楕円の向き(角度)を計算
    # eigenvecs[:,0] が第一主軸方向
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

    ellipse = Ellipse(
        xy=center,                # 楕円の中心
        width=2*r1,               # 楕円の横軸（直径）
        height=2*r2,              # 楕円の縦軸（直径）
        angle=angle,              # 楕円の回転角度(度数法)
        edgecolor=edge_color,
        facecolor=face_color,
        alpha=alpha
    )
    ax.add_patch(ellipse)
def myPCA(img_batch_numpy,ono_batch_numpy,word_list,path_list,onelabelcount,numlabel):
    #主成分分析する
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances
    from pykakasi import kakasi
    kakasi=kakasi()
    kakasi.setMode("H","a")
    converter=kakasi.getConverter()
    romaji_list = [converter.do(text) for text in word_list]
    img_batch_numpy=img_batch_numpy.reshape(-1,78848) #変数2番目は次元数に合わせる
    ono_batch_numpy=ono_batch_numpy.reshape(-1,78848) 
    # ono_batch_numpy=ono_batch_numpy.reshape(-1,100) #PCAで分析して、100次元で再構成を合わせようとしたとき
    combined_numpy = np.concatenate((img_batch_numpy, ono_batch_numpy), axis=0)
    combined_numpy = select_subset_from_combined(combined_numpy, 
                                       n_labels=14, 
                                       n_samples_per_label=100, 
                                       n_subset=10)

    pca=PCA(n_components=100)
    # 分析結果を元にデータセットを主成分に変換する
    transformed=pca.fit_transform(combined_numpy)
    eigenvalues = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    # # 固有値と寄与率の表示
    # print("固有値:", eigenvalues)
    # print("寄与率:", explained_variance_ratio)
    markers1 = ["$x$","$\\alpha$", "$A$","$B$","$C$","$D$","$E$","$F$","$G$","$H$","$J$","$K$", "$L$","$M$","$N$","$O$","$P$","$Q$","$R$","$S$","$T$","$U$","$V$","$W$", ",", "o", "v", "^", "p", "*", "D","d"]
    color1 =["red","yellow", "gray","silver","rosybrown","firebrick",
           "darksalmon","sienna","sandybrown","tan",
            "gold","olivedrab","chartreuse","palegreen",
            "darkgreen","lightseagreen","paleturquoise",
            "deepskyblue","blue","pink","orange","crimson",
            "mediumvioletred","plum","darkorchid","mediumpurple",
            "chocolate","peru","yellow","y","aqua","lightsteelblue","linen","teal"]
    fig = plt.figure()
    num=int(len(combined_numpy)/2)
    imagecount=0
    numlabel=numlabel #ラベルの数
    original_num=100 #削減する前の1ラベルのデータ数
    for i in range(numlabel):
        plt.scatter(transformed[imagecount:imagecount+onelabelcount, 0], transformed[imagecount:imagecount+onelabelcount, 1],c=color1[i],label=romaji_list[original_num*i])#画像        
        imagecount+=onelabelcount
    # plt.scatter(transformed[onelabelcount*numlabel:num, 0], transformed[onelabelcount*numlabel:num, 1],c="teal",label=romaji_list[onelabelcount*numlabel])#データローダのdrop_lastによって削られた余り者たちをまとめてプロット
    
    # for i in range(int(len(transformed)/7)): #画像のパスをプロットする
    #     plt.text(transformed[i, 0], transformed[i, 1],path_list[i],fontsize=8)    
    # onelabelcount=1
    for i in range(numlabel): #ラベルをプロットしていく
        plt.scatter(transformed[i*onelabelcount+int(len(transformed)/2), 0], transformed[i*onelabelcount+int(len(transformed)/2), 1],c="blue",marker="$x$")#オノマトペ
        plt.text(transformed[i*onelabelcount+int(len(transformed)/2), 0], transformed[i*onelabelcount+int(len(transformed)/2), 1],romaji_list[i*original_num],fontsize=8)

    # plt.scatter(transformed[35, 0], transformed[35, 1],c=color1[0],label=romaji_list[1])#画像の単一プロット  
    # plt.scatter(transformed[:, 0], transformed[:, 1],c=color1[0],label=romaji_list[:])#画像の単一プロット

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) #画像の凡例をラベルに基づいてグラフ外に表示
    plt.title('principal component')
    EVR_pc1=round(explained_variance_ratio[0],2)
    EVR_pc2=round(explained_variance_ratio[1],2)
    plt.xlabel('pc1 contribution ratio'+str(EVR_pc1))
    plt.ylabel('pc2 contribution ratio'+str(EVR_pc2))
    # plt.xlim([-1,1])
    # plt.ylim([-1.1,0.5])
    plt.grid(True)
 
    plt.tight_layout()
    if not os.path.exists(f"figure/{nums}"):
        os.makedirs(f"figure/{nums}")
    fig.savefig(f"figure/{nums}/trainimagehiddenPCA.png")
    plt.show()
    return 

def select_subset_from_combined(combined_numpy, n_labels=14, n_samples_per_label=100, n_subset=10): #2400サンプルの78848次元の共分散行列の計算は重たいので、数サンプルピックする関数
    """
    combined_numpy (2400, 78848) の配列から、
    画像特徴(先頭1400) と 音素特徴(後ろ1400) を、
    ラベルごとに n_subset 個ずつ抜き出した新たな配列を作成する。

    ・画像特徴は先頭1400行 (14ラベル x 100個)
    ・音素特徴は後ろ1400行 (14ラベル x 100個)
    ・各ラベルごとに先頭 n_subset 行を抜き出す（本例では 10）

    Args:
        combined_numpy: shape (2400, 78848) の配列
        n_labels: ラベルの総数 (デフォルト14)
        n_samples_per_label: 各ラベルのデータ数 (デフォルト100)
        n_subset: 1ラベルあたり何個抜き出すか (デフォルト10)

    Returns:
        sub_combined: shape ((n_labels*n_subset*2), 78848) の配列
                      (画像特徴 + 音素特徴でラベルごと各 n_subset 個)
    """
    # 画像特徴: 先頭 1400 行 (14ラベル x 100個 = 1400)
    # 音素特徴: 後ろ 1400 行 (14ラベル x 100個 = 1400)
    # 合計 2400 行（最初の 1400 が画像、次の 1400 が音素）
    
    # 出力用リスト
    selected_image_features = []
    selected_phoneme_features = []

    # 画像特徴をラベル毎に n_subset 個抜き出し
    for label_idx in range(n_labels):
        start_idx = label_idx * n_samples_per_label
        end_idx = start_idx + n_subset
        selected_image_features.append(combined_numpy[start_idx:end_idx])

    # 音素特徴をラベル毎に n_subset 個抜き出し
    # 画像特徴1400行をオフセットとして足す
    offset = n_labels * n_samples_per_label  # 14*100 = 1400
    for label_idx in range(n_labels):
        start_idx = offset + label_idx * n_samples_per_label
        end_idx = start_idx + n_subset
        selected_phoneme_features.append(combined_numpy[start_idx:end_idx])

    # listを結合して縦方向に積む
    sub_combined = np.vstack(selected_image_features + selected_phoneme_features)
    return sub_combined


if __name__ == '__main__':
    with torch.no_grad():

        device = "cuda:1" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        phoneme_num=40 #入出力として使える音素の数=データセット内の.n_wordsに等しい
        embedding_size = 128
        hidden_size   = 128
        
        size=64
        nums=112


        lang=Lang('dataset/onomatope/dictionary.csv')
 
        #モデルの準備
        encoder           = Encoder( phoneme_num, embedding_size, hidden_size ).to(device)
        decoder           = Decoder( hidden_size, embedding_size, phoneme_num ).to(device)
        image_model=TextureNet().to(device)
        prompt_converter=PromptEncoder().to(device)
        phonemevae=PhonemeVAE().to(device)

        
        enfile=f"model/{nums}/phonemeencoder_{nums}.pth"
        defile=f"model/{nums}/phonemedecoder_{nums}.pth"
        imgfile=f"model/{nums}/image_model_{nums}.pth"
        model_save_path=f"model/{nums}/prompt_converter_{nums}.pth"
        phonemevaefile=f"model/{nums}/phonemevae_{nums}.pth"
        
        encoder.load_state_dict( torch.load( enfile ) ) #読み込み
        decoder.load_state_dict( torch.load( defile ) )
        image_model.load_state_dict(torch.load(imgfile,map_location=device)) #map_locationをすることで読み込みデバイスの指定ができる（なぜだかわからんが、モデルを別のcudaに乗せようとして実行できない時があった）
        prompt_converter.load_state_dict(torch.load(model_save_path))    
        phonemevae.load_state_dict(torch.load(phonemevaefile))
        model_id = "dream-textures/texture-diffusion"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16) #from_pretrainedは/pipelines/pipelines_utils.py内で定義されているクラス
        pipe = pipe.to(device)        
        
        encoder.eval()
        decoder.eval()
        image_model.eval()
        prompt_converter.eval()
        phonemevae.eval()
               
        criterion=nn.MSELoss()
        cos=nn.CosineEmbeddingLoss()
        cos2=nn.CosineSimilarity()
        transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
        
        # calc_gram(image_model,device) #任意の2つのグラム行列の差を計算
        evaluate(encoder,decoder,image_model,pipe,prompt_converter,phonemevae,nums,device)
    

       


