#データセットの関数、無駄にインポートしているものもあるかも
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

from utils import tensorFromSentence

class Lang: #オノマトペ単語のデータセット
    def __init__( self, filename ,augmentation=False): #呼び出されたとき、最初に行うこと
        max_length    = 20
        self.filename = filename
        self.ono=[]    
        self.word2index = {}
        self.word2count = {}
        self.sentences = []
        self.index2word = { 0: "SOS", 1: "EOS" }
        self.n_words = 2  # Count SOS and EOS
        self.augmentation=augmentation
        df = pd.read_csv(filename)
        num=df.shape[0] #csvファイルの行数を取得（データセットの数）
        
        for i in range(num): #264種類のラベルを作る
            word=df.iloc[i,2] #単語(3列目)
            phoneme = df.iloc[i, 1]  # 音素(2列目)
            i=i+1
            self.ono.append(word)
            self.sentences.append(phoneme)
        self.allow_list = [ True ] * len( self.sentences )
        allow_list  = self.get_allow_list( max_length )#maxlength以下の長さの音素数をallowlistに入れる（長すぎる音素は省かれる)
        self.load_file(allow_list)#max以下の長さである単語を引数に渡す（今回の場合は264こ）
            
    def get_sentences( self ):
        return self.sentences[ :: ] 

    def get_sentence( self, index ):
        return self.sentences[ index ]

    def choice( self ): #ランダムに配列をピックする
        while True:
            index = random.randint( 0, len( self.allow_list ) - 1 )
            if self.allow_list[ index ]:
                break
        return self.sentences[ index ], index

    def get_allow_list( self, max_length ):
        allow_list = []
        for sentence in self.sentences:
            if len( sentence.split() ) < max_length:
                allow_list.append( True )
            else:
                allow_list.append( False )
        return allow_list
                    
    def load_file( self, allow_list = [] ):
        if allow_list: #allow_listが空でなければ実行
            self.allow_list = [x and y for (x,y) in zip( self.allow_list, allow_list ) ] #自分のTrueの情報を与えられたデータセットに合わせる(max_lengthより長い音素は全てFalseに変わる)
        self.target_sentences = []
        for i, sentence in enumerate( self.sentences ):
            if self.allow_list[i]: #i番目がtrueであったら行う、Falseならスルー
                self.addSentence( sentence ) #単語に含まれている音素が初見であればリストに加えていく
                self.target_sentences.append( sentence ) #音素をtarget_sentencesに加えていく
                    
    def addSentence( self, sentence ): #一単語に含まれている音素を全てリストに入れていく
        for word in sentence.split():
            self.addWord(word)
            

    def addWord( self, word ): #wordを数値化する
      
        
        if word not in self.word2index:
            self.word2index[ word ] = self.n_words #word2indexに音素を入力すると数字が出てくる
            
            self.word2count[ word ] = 1 #ある音素が出現した数をカウントしてる
            self.index2word[ self.n_words ] = word #index2wordに数字を入力すると音素が出てくる
            self.n_words += 1 #リストに入っている音素の総数(かぶっているものは除外される)語彙数みたいなもん
        else:
            self.word2count[word] += 1
    #ここから自分



    def __len__(self):
        """data の行数を返す。
        """
        return len(self.ono)
    def __getitem__(self, index): #dataloaderを作ったときにできるやつ
        """サンプルを返す。
        """
        
        ono=self.ono[index] #取ってきたラベル番号のオノマトペを取ってくる（オノマトペのラベル番号とリストの番号は一致している）
        phoneme = self.sentences[index] #同上


        if self.augmentation: #データセットの引数においてTrueが渡されていればaugmentationを実行する
        #25％の確率でランダムに生成されたオノマトペに代わる
            if random.random()<0.25:
                rand_num=torch.randint(low=2,high=self.n_words,size=(2,))
                rand_ono=rand_num.repeat(2)
                if random.random()<0.5: #50％の確率で３文字が連なったオノマトペを生成
                    rand_num=torch.randint(low=2,high=self.n_words,size=(3,))
                    rand_ono=rand_num.repeat(2)
                word=[]
                for di in range(len(rand_ono)): #ラベルを対応する音素に変換
                    word.append(self.index2word[rand_ono[di].item()])
                word=[x+' 'for x in word] #1音素ずつに半角の空白を追加
                word[-1]=word[-1].strip() #最後の音素の後ろの空白だけ消す
                word=''.join(word) #リストになってたものを１つの単語にする
                
                ono=word
                phoneme=word

        return ono, phoneme


class ImageDataset: #画像単体のデータセット
    def __init__( self,dir,transform ): #呼び出されたとき、最初に行うこと
        self.transform = transform
        self.data = [] #画像が入ってる
        target_dir = os.path.join(dir, "*/*")
        for path in glob.glob(target_dir):
            self.data.append(path)
        self.data.sort()
    def __len__(self):
        """data の行数を返す。
        """
        return len(self.data)
    def __getitem__(self, index): #dataloaderを作ったときにできるやつ
        """サンプルを返す。
        """
        img_path = self.data[index] #適当な画像を取ってくる
        img = Image.open(img_path).convert("RGB") #img_pathの画像を開く
        img = self.transform(img) #transformする
        return img, img_path

class ImageLang:   #画像とオノマトペの単語を返すデータセット
    def __init__( self, filename,dir,image_hiddendir,transform ): #呼び出されたとき、最初に行うこと
        self.transform = transform
        max_length    = 20
        self.filename = filename
        self.data = [] #画像のパスが入ってる
        self.img=[] #画像が入ってる
        self.img_hidden=[]
        self.labels = [] #画像のラベルを入れてる
        self.ono=[]    
        self.word2index = {}
        self.word2count = {}
        self.sentences = []
        self.index2word = { 0: "SOS", 1: "EOS" }
        self.n_words = 2  # Count SOS and EOS
    

        name_to_label={}
        df = pd.read_csv(filename)
        num=df.shape[0] #csvファイルの行数を取得（データセットの数）
        for i in range(num): #264種類のラベルを作る
            word=df.iloc[i,2] #単語(3列目)
            phoneme = df.iloc[i, 1]  # 音素(2列目)
            ono_labels = df.iloc[i, 0].astype(str)  # ラベル (1列目)
            dict={word:i}
            i=i+1
            self.ono.append(word)
            self.sentences.append(phoneme)
            name_to_label.update(dict)
        self.allow_list = [ True ] * len( self.sentences )


        target_dir = os.path.join(dir, "*") #target_dirにtrainやvalidの下を総なめするように指定されるパスが入る
        #フォルダではなくファイル名でラベルを振る場合----------------------------------------------------------------
        # for path in glob.glob(target_dir):

        #     name = os.path.splitext(os.path.basename(path))[0]
        #     name=unicodedata.normalize("NFKC",name)
        #     label = name_to_label[name]
        #     self.data.append(path)
        #     self.labels.append(label) 


        #----------------------------------------------------------------
        #フォルダにラベル名を振る場合のコード----------------------------------------------------------------
        for path in glob.glob(target_dir): #pathの中にラベルフォルダまでが指定される

            name = os.path.splitext(os.path.basename(path))[0] #nameで音素とマッチングさせるためのラベル名が取得される
            name=unicodedata.normalize("NFKC",name) #文字化け対策でノーマライズ
            label = name_to_label[name] #ラベルリストからnameのラベル番号を取得
            
            
            for data in glob.glob(os.path.join(path,"*")): #os.path.join~~することであみあみの下を総なめするようにディレクトリを修正、それをglobによって全部取り出す
                self.labels.append(label) 
                self.data.append(data)
                img = Image.open(data).convert("RGB") #img_pathの画像を開く
                img = self.transform(img) #transformする
                self.img.append(img)
        for path in glob.glob(f"{image_hiddendir}/*"):
            for data in glob.glob(f"{path}/*"):
                self.img_hidden.append(data)
        #----------------------------------------------------------------
        allow_list  = self.get_allow_list( max_length )#maxlength以下の長さの音素数をallowlistに入れる（長すぎる音素は省かれる)
        self.load_file(allow_list)#max以下の長さである単語を引数に渡す（今回の場合は264こ）
        
    def get_sentences( self ):
        return self.sentences[ :: ] 

    def get_sentence( self, index ):
        return self.sentences[ index ]

    def choice( self ): #ランダムに配列をピックする
        while True:
            index = random.randint( 0, len( self.allow_list ) - 1 )
            if self.allow_list[ index ]:
                break
        return self.sentences[ index ], index

    def get_allow_list( self, max_length ):
        allow_list = []
        for sentence in self.sentences:
            if len( sentence.split() ) < max_length:
                allow_list.append( True )
            else:
                allow_list.append( False )
        return allow_list
                    
    def load_file( self, allow_list = [] ):
        if allow_list: #allow_listが空でなければ実行
            self.allow_list = [x and y for (x,y) in zip( self.allow_list, allow_list ) ] #自分のTrueの情報を与えられたデータセットに合わせる(max_lengthより長い音素は全てFalseに変わる)
        self.target_sentences = []
        for i, sentence in enumerate( self.sentences ):
            if self.allow_list[i]: #i番目がtrueであったら行う、Falseならスルー
                self.addSentence( sentence ) #単語に含まれている音素が初見であればリストに加えていく
                self.target_sentences.append( sentence ) #音素をtarget_sentencesに加えていく
                    
    def addSentence( self, sentence ): #一単語に含まれている音素を全てリストに入れていく
        for word in sentence.split():
            self.addWord(word)
            

    def addWord( self, word ): #wordを数値化する
    
        
        if word not in self.word2index:
            self.word2index[ word ] = self.n_words #word2indexに音素を入力すると数字が出てくる
            
            self.word2count[ word ] = 1 #ある音素が出現した数をカウントしてる
            self.index2word[ self.n_words ] = word #index2wordに数字を入力すると音素が出てくる
            self.n_words += 1 #リストに入っている音素の総数(かぶっているものは除外される)語彙数みたいなもん
        else:
            self.word2count[word] += 1
    #ここから自分



    def __len__(self):
        """data の行数を返す。
        """
        return len(self.data)
    def __getitem__(self, index):

        img_path = self.data[index] #適当な画像を取ってくる
        img_label = self.labels[index] #その画像のラベル番号は何番か取ってくる
        ono=self.ono[img_label] #取ってきたラベル番号のオノマトペを取ってくる（オノマトペのラベル番号とリストの番号は一致している）
        
        phoneme = self.sentences[img_label] #同上

        img=self.img[index]
        img_hidden=torch.load(self.img_hidden[index]) #こいつはtensor配列なのでrequires_grad=Trueとなる
        img_hidden.requires_grad=False
        return img, img_path, ono, phoneme,img_hidden    