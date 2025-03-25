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
import japanize_matplotlib  # 日本語をサポートするためのライブラリ
from sklearn.decomposition import PCA
import math
import matplotlib.collections as mc


from seq2seq import Lang,Encoder,Decoder,tensorFromSentence,SkipGramEncoder


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

def calc_accu(encoder,decoder,dataloader,lang):
    score=0
    for batch_num,(ono,phoneme) in enumerate(dataloader):  
        for data_num in range(dataloader.batch_size):           
            ono_word,encoder_hidden=ono_to_ono(phoneme[data_num],encoder,decoder,lang)
            print(ono_word,phoneme[data_num])
            word=[x.replace("<EOS>","") for x in ono_word]
            word=[x+' 'for x in word] #1音素ずつに半角の空白を追加
            word[-1]=word[-1].strip() #最後の音素の後ろの空白だけ消す
            word=''.join(word) #リストになってたものを１つの単語にする
            print(word)
            if is_close_match(phoneme[data_num],word):
                score+=1
            print(score,len(dataloader.dataset))
    return (score/len(dataloader.dataset))

def ono_to_ono(sentence,encoder,decoder,lang):
    SOS_token = 0
    EOS_token = 1    
    input_tensor   = tensorFromSentence(lang, sentence)
    input_length   = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden()
    
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

def get_phoneme_hidden(encoder,decoder,lang):
    SOS_token=0
    EOS_token=1
    phoneme_numpy=np.zeros((0,128))
    word_list=[]
    for i in range(2,len(lang.index2word)):
        input_tensor   = tensorFromSentence(lang, lang.index2word[i])
        input_length   = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        word_list.append(lang.index2word[i])

        for ei in range(input_length):
        
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            
        ono_hidden=encoder_hidden.view(1,128).to('cpu').numpy() 
        phoneme_numpy=np.concatenate((phoneme_numpy,ono_hidden),axis=0)
        i+=1
    return phoneme_numpy,word_list

def myPCA(ono_batch_numpy,word_list):
    #主成分分析する
    from sklearn.preprocessing import StandardScaler

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances

    scaler=StandardScaler()

    
    ono_batch_numpy=ono_batch_numpy.reshape(-1,128)

    pca=PCA()

    ono_pca=pca.fit(ono_batch_numpy)
    # 分析結果を元にデータセットを主成分に変換する




    transformed_phoneme=ono_pca.fit_transform(ono_batch_numpy)
    eigenvalues = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_

    # # 固有値と寄与率の表示
    # print("固有値:", eigenvalues)
    # print("寄与率:", explained_variance_ratio)

    markers1 = ["$x$","$\\alpha$", "$A$","$B$","$C$","$D$","$E$","$F$","$G$","$H$","$J$","$K$", "$L$","$M$","$N$","$O$","$P$","$Q$","$R$","$S$","$T$","$U$","$V$","$W$", ",", "o", "v", "^", "p", "*", "D","d"]
    color1 =["gray","silver","rosybrown","firebrick",
            "red","darksalmon","sienna","sandybrown","tan",
            "gold","olivedrab","chartreuse","palegreen",
            "darkgreen","lightseagreen","paleturquoise",
            "deepskyblue","blue","pink","orange","crimson",
            "mediumvioletred","plum","darkorchid","mediumpurple",
            "navy","chocolate","peru","yellow","y","aqua","lightsteelblue","linen","teal"]
    
    fig = plt.figure()
 
   

    plt.scatter(transformed_phoneme[:, 0], transformed_phoneme[:, 1],c="blue",marker="$x$")#オノマトペ
    print(len(transformed_phoneme))
    print(len(word_list))
    for i in range(int(len(transformed_phoneme))): #ラベルをプロットしていく
        plt.text(transformed_phoneme[i, 0], transformed_phoneme[i, 1],word_list[i],fontsize=16)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) #画像の凡例をラベルに基づいてグラフ外に表示
    plt.title('principal component')
    EVR_pc1=round(explained_variance_ratio[0],2)
    EVR_pc2=round(explained_variance_ratio[1],2)
    plt.xlabel('pc1 寄与率は'+str(EVR_pc1))
    plt.ylabel('pc2 寄与率は'+str(EVR_pc2))
    # plt.xlim([-0.5,1.0])
    # plt.ylim([-0.6,0.7])
    

 
    plt.tight_layout()
    fig.savefig("figure/phoneme_PCA2.png")
    plt.show()
    return 

if __name__ == '__main__':
    with torch.no_grad():
        SOS_token = 0
        EOS_token = 1
        device = "cuda" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size =8
        num=40 #入出力として使える音素の数=データセット内の.n_wordsに等しい
        embedding_size = 128
        hidden_size   = 128
        max_length=20
        size=32
        
        #データセットの準備
        transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
        lang  = Lang( 'dataset/onomatope/dictionary.csv')
        # train_lang  = Lang( 'dataset/onomatope/onomatope.csv')
        valid_lang  = Lang( 'dataset/onomatope/onomatopeunknown.csv')

        train_dataloader = DataLoader(lang, batch_size=batch_size, shuffle=False,drop_last=True) #drop_lastをtruenにすると最後の中途半端に入っているミニバッチを排除してくれる
        valid_dataloader=DataLoader(valid_lang,batch_size=batch_size, shuffle=False,drop_last=True)

        #モデルの準備
        encoder           = SkipGramEncoder( num, embedding_size, hidden_size ).to( device )
        decoder           = Decoder( hidden_size, embedding_size, num ).to( device )
        enfile = "model/firstencodercheck" #学習済みのエンコーダモデル
        defile = "model/firstdecodercheck" #学習済みのデコーダモデル

        encoder.load_state_dict( torch.load( enfile ) ) #読み込み
        decoder.load_state_dict( torch.load( defile ) )
        encoder.eval()
        decoder.eval()
        dataloader=valid_dataloader
        sentence="h o q k o r i"

        phoneme_numpy,word_list=get_phoneme_hidden(encoder,decoder,lang)
        myPCA(phoneme_numpy,word_list)
        ono_word,_=ono_to_ono(sentence,encoder,decoder,lang)
        print(ono_word)
        # print(calc_accu(encoder,decoder,dataloader,lang))