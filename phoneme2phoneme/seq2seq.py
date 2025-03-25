#自分なりにseq2seqを書いてみた
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


SOS_token = 0
EOS_token = 1

device = "cuda:1" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lang:
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


# Start core part
class Encoder( nn.Module ):
    def __init__( self, input_size, embedding_size, hidden_size ):
        super().__init__()
        self.hidden_size = hidden_size
        # 単語をベクトル化する。1単語はembedding_sie次元のベクトルとなる
        self.embedding   = nn.Embedding( input_size, embedding_size )
        # GRUに依る実装. 
        self.gru         = nn.GRU( embedding_size, hidden_size )
        # self.layer_norm = nn.LayerNorm(normalized_shape=128)
        self.sigmoid = nn.Sigmoid()  # Sigmoid
        self.tanh=nn.Tanh()
    

    def initHidden( self ):
        return torch.zeros( 1, 1, self.hidden_size ).to( device )

    def forward( self, _input, hidden ):
        # 単語のベクトル化
        embedded        = self.embedding( _input ).view( 1, 1, -1 )
        out, new_hidden = self.gru( embedded, hidden )
        new_hidden=new_hidden/(torch.norm(new_hidden))
        return out, new_hidden

class SkipGramEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)  # 追加された全結合層

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(device)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        out, new_hidden = self.gru(embedded, hidden)
        new_hidden = new_hidden / (torch.norm(new_hidden))
        logits = self.fc(new_hidden.squeeze(0))  # 文脈単語の確率を計算
        probabilities = F.softmax(logits, dim=1)
        return probabilities, new_hidden
    
class Decoder( nn.Module ):
    def __init__( self, hidden_size, embedding_size, output_size ):
        super().__init__()
        self.hidden_size = hidden_size
        # 単語をベクトル化する。1単語はembedding_sie次元のベクトルとなる
        self.embedding   = nn.Embedding( output_size, embedding_size )
        # GRUによる実装（RNN素子の一種）
        self.gru         = nn.GRU( embedding_size, hidden_size )
        # 全結合して１層のネットワークにする
        self.linear         = nn.Linear( hidden_size, output_size )
        # softmaxのLogバージョン。dim=1で行方向を確率変換する(dim=0で列方向となる)
        self.softmax     = nn.LogSoftmax( dim = 1 )
        # self.layer_norm = nn.LayerNorm(normalized_shape=128)
        self.sigmoid = nn.Sigmoid()  # Sigmoid
        
    def forward( self, _input, hidden ):
        # 単語のベクトル化。GRUの入力に合わせ三次元テンソルにして渡す。
        embedded           = self.embedding( _input ).view( 1, 1, -1 )
        # relu活性化関数に突っ込む( 3次元のテンソル）
        relu_embedded      = F.relu( embedded )
        # GRU関数( 入力は３次元のテンソル )
        gru_output, hidden = self.gru( relu_embedded, hidden )
        #hiddenは次に渡す新しい特徴ベクトル、こいつも正規化しないとデコーダにおいて2回目以降は0～1の範囲じゃないやつを渡してしまう
        # hidden=self.layer_norm(hidden)
        # hidden = self.sigmoid(hidden)  # apply Sigmoid
        # softmax関数の適用。outputは３次元のテンソルなので２次元のテンソルを渡す
        result             = self.linear( gru_output[ 0 ] ) 


        return result, hidden
    
    def initHidden( self ):
        return torch.zeros( 1, 1, self.hidden_size ).to( device )



def tensorFromSentence( lang, sentence ): #sentenceをインデックス番号に変換したtensor配列にする
    indexes = [ lang.word2index[ word ] for word in sentence.split(' ') ] #sentenceの音素をインデックス番号に変換してリストにする
    indexes.append( EOS_token )
    

    return torch.tensor( indexes, dtype=torch.long ).to( device ).view(-1, 1)
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
def ono_to_ono(sentence,encoder,decoder,lang,max_length):
    
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

def Train(encoder,decoder,lang,dataloader):
    train_ono_loss=0
    train_phoneme_loss=0
    train_total_loss=0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    learning_rate = 1e-5 #10e-6
    batch_size =8
    criterion      = nn.CrossEntropyLoss() #これをバッチサイズ分繰り返してそれをエポック回数分まわす？
    score=0

    encoder_optimizer = optim.Adam( encoder.parameters(), lr=learning_rate )
    decoder_optimizer = optim.Adam( decoder.parameters(), lr=learning_rate )
    for batch_num,(ono,phoneme) in tqdm.tqdm(enumerate(dataloader),total=len(dataloader)):
        batch_ono_loss=0 #1バッチ分のLossをここに入れていく
        batch_phoneme_loss=0
        batch_total_loss=0
        for data_num in range(dataloader.batch_size):
            ono2_tensor=tensorFromSentence(lang,phoneme[data_num]).to(device) #一つ一つの音素をインデックス番号に変換する
            SOS_tensor=torch.tensor([[0]]).to(device)

            encoder_hidden              = encoder.initHidden()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            input_length  = ono2_tensor.size(0)
#------------------------------------------------------------------------------phoneme2vec
            ONO2_tensor=torch.cat((SOS_tensor,ono2_tensor),dim=0).to(device)
            pre_loss=0
            next_loss=0
            for i in range( 1,input_length-1 ): #skip-gram用、RNNの特性を無視した毎回encoder_hiddenをイニシャライズして1音素ごとの分散表現を獲得する

                phoneme_hidden, encoder_hidden = encoder( ONO2_tensor[ i ], encoder_hidden ) 
                #lossの計算してバックワード
                pre_loss+=criterion(phoneme_hidden,ONO2_tensor[i-1])
                next_loss+=criterion(phoneme_hidden,ONO2_tensor[i+1])
                
                encoder_hidden=encoder.initHidden()    
            phoneme_loss=pre_loss+next_loss
            phoneme_loss=phoneme_loss/2
#------------------------------------------------------------------------------phoneme2vec

            for i in range( input_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
                phoneme_hidden, encoder_hidden = encoder( ono2_tensor[ i ], encoder_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！  
            # Decoder phese
            loss_ono = 0 #seq2seqのloss
            decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to( device )
            decoder_hidden = encoder_hidden
            for i in range( input_length ):
                decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden ) 
                    
                decoder_input = ono2_tensor[ i ] #次の音素（インデックス）をセットしておく
                if random.random() < 0.5: 
                    topv, topi                     = decoder_output.topk( 1 )
                    decoder_input                  = topi.squeeze().detach() # detach from history as input
                loss_ono += criterion( decoder_output, ono2_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算
                topv, topi = decoder_output.data.topk(1)


                if topi.item() == EOS_token: break #decoder_inputの中がEOSだったらここで終了

            #ここは精度評価---------------------------------------------------------          
            ono_word,encoder_hidden=ono_to_ono(phoneme[data_num],encoder,decoder,lang,20) 
            word=[x.replace("<EOS>","") for x in ono_word]
            word=[x+' 'for x in word] #1音素ずつに半角の空白を追加
            word[-1]=word[-1].strip() #最後の音素の後ろの空白だけ消す
            word=''.join(word) #リストになってたものを１つの単語にする
            if is_close_match(phoneme[data_num],word):
                score+=1
            #ここは精度評価---------------------------------------------------------  

            
            loss=loss_ono+phoneme_loss
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            #1バッチ分(８個のデータ)のLossになるように加算していく    
            batch_ono_loss +=loss_ono
            batch_phoneme_loss+=phoneme_loss    
            batch_total_loss +=loss
        # print(batch_loss/dataloader.batch_size) #１バッチにおける1個のデータあたりのLossを計算する
        train_ono_loss +=batch_ono_loss/dataloader.batch_size
        train_phoneme_loss+=batch_phoneme_loss/dataloader.batch_size
        train_total_loss +=batch_total_loss/dataloader.batch_size

    #(total_loss)をバッチの個数分で割ることで１データローダにおけるLossがわかる   
    train_ono_loss=train_ono_loss/len(dataloader) 
    train_phoneme_loss=train_phoneme_loss/len(dataloader) 
    train_total_loss=train_total_loss/len(dataloader) 
    accu=score/len(dataloader.dataset)
    return train_ono_loss,train_phoneme_loss,train_total_loss,encoder,decoder,accu

def Validation(encoder,decoder,lang,dataloader):
    with torch.no_grad():
        valid_ono_loss=0
        valid_phoneme_loss=0
        valid_total_loss=0

        criterion=nn.CrossEntropyLoss()
        mse=nn.MSELoss()
        score=0
        for batch_num,(ono,phoneme) in tqdm.tqdm(enumerate(dataloader),total=len(dataloader)):
            batch_ono_loss=0 #1バッチ分のLossをここに入れていく
            batch_phoneme_loss=0
            batch_total_loss=0
            for data_num in range(dataloader.batch_size): 
                SOS_tensor=torch.tensor([[0]]).to(device)
                ono2_tensor=tensorFromSentence(lang,phoneme[data_num]).to(device) #一つ一つの音素をインデックス番号に変換する
                encoder_hidden              = encoder.initHidden()
                input_length  = ono2_tensor.size(0)

#------------------------------------------------------------------------------phoneme2vec
                ONO2_tensor=torch.cat((SOS_tensor,ono2_tensor),dim=0).to(device)
                pre_loss=0
                next_loss=0
                for i in range( 1,input_length-1 ): #skip-gram用、RNNの特性を無視した毎回encoder_hiddenをイニシャライズして1音素ごとの分散表現を獲得する

                    phoneme_hidden, encoder_hidden = encoder( ONO2_tensor[ i ], encoder_hidden ) 
                    #lossの計算してバックワード
                    pre_loss+=criterion(phoneme_hidden,ONO2_tensor[i-1])
                    next_loss+=criterion(phoneme_hidden,ONO2_tensor[i+1])
                    
                    encoder_hidden=encoder.initHidden()    
                phoneme_loss=pre_loss+next_loss
                phoneme_loss=phoneme_loss/2
#------------------------------------------------------------------------------phoneme2vec
                for i in range( input_length ): #input_length（単語の長さ）の回数分繰り返す、つまりencoder_hiddenが一音素ごとに更新されていく。これが終わったencode_hiddenは一単語を網羅して考慮された特徴ベクトルとなる
                    encoder_output, encoder_hidden = encoder( ono2_tensor[ i ], encoder_hidden ) #i番目のデータをエンコーダに投げる、このデータのラベルさえわかれば・・・！！           

                loss_ono = 0 #seq2seqのloss
                decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to( device )
                decoder_hidden = encoder_hidden
                for i in range( input_length ):
                    decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden ) #
                        
                    decoder_input = ono2_tensor[ i ] #次の音素（インデックス）をセットしておく
                    if random.random() < 0.5: 
                        topv, topi                     = decoder_output.topk( 1 )
                        # print(topi)
                        # sys.exit()
                        decoder_input                  = topi.squeeze().detach() # detach from history as input
                    loss_ono += criterion( decoder_output, ono2_tensor[ i ] ) #入力となる音素とデコーダのアウトプットから得られる音素の確率密度を計算

                    topv, topi = decoder_output.data.topk(1)
                    if topi.item() == EOS_token: break #decoder_inputの中がEOSだったらここで終了
                #ここは精度評価---------------------------------------------------------          
                ono_word,encoder_hidden=ono_to_ono(phoneme[data_num],encoder,decoder,lang,20) 
                word=[x.replace("<EOS>","") for x in ono_word]
                word=[x+' 'for x in word] #1音素ずつに半角の空白を追加
                word[-1]=word[-1].strip() #最後の音素の後ろの空白だけ消す
                word=''.join(word) #リストになってたものを１つの単語にする
                if is_close_match(phoneme[data_num],word):
                    score+=1
                #ここは精度評価---------------------------------------------------------  
                loss=loss_ono+phoneme_loss

                #1バッチ分(８個のデータ)のLossになるように加算していく    
                batch_ono_loss +=loss_ono
                batch_phoneme_loss+=phoneme_loss    
                batch_total_loss +=loss
            # print(batch_loss/dataloader.batch_size) #１バッチにおける1個のデータあたりのLossを計算する
            valid_ono_loss +=batch_ono_loss/dataloader.batch_size
            valid_phoneme_loss+=batch_phoneme_loss/dataloader.batch_size
            valid_total_loss +=batch_total_loss/dataloader.batch_size

        #(total_loss)をバッチの個数分で割ることで１データローダにおけるLossがわかる   
        valid_ono_loss=valid_ono_loss/len(dataloader) 
        valid_phoneme_loss=valid_phoneme_loss/len(dataloader) 
        valid_total_loss=valid_total_loss/len(dataloader) 
        accu=score/len(dataloader.dataset)
        return valid_ono_loss,valid_phoneme_loss,valid_total_loss,accu
def main():
    embedding_size = 128
    hidden_size   = 128
    num=40 #入出力として使える音素のラベル数=データセット内の.n_wordsに等しい
    


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # encoder           = Encoder( num, embedding_size, hidden_size ).to( device )
    encoder           = SkipGramEncoder( num, embedding_size, hidden_size ,num).to( device )
    decoder           = Decoder( hidden_size, embedding_size, num ).to( device )
    epochs=100000
    save_loss=40
    batch_size=4
    augment=False
    lang  = Lang( 'dataset/dictionary.csv',augment) #word2indexを作るための辞書
    valid_dataset  = Lang( 'dataset/onomatopeunknown.csv',augment)
    train_dataloader=DataLoader(lang,batch_size=batch_size, shuffle=True,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,drop_last=True) #drop_lastをtrueにすると最後の中途半端に入っているミニバッチを排除してくれる

    # enfile = "model/testenc" #学習済みのエンコーダモデル
    # defile = "model/testdec" #学習済みのデコーダモデル
    # encoder.load_state_dict( torch.load( enfile ) ) #読み込み
    # decoder.load_state_dict( torch.load( defile ) )
    for epoch in range(epochs):
        writer=SummaryWriter(log_dir="/workspace/mycode/Documents/log/phoneme2vec")
        train_ono,train_phoneme,train_total,encoder,decoder,trainaccu=Train(encoder,decoder,lang,train_dataloader)
        valid_ono,valid_phoneme,valid_total,validaccu=Validation(encoder,decoder,lang,valid_dataloader)
        print( "[epoch num %d ] [ train: %f]" % ( epoch+1, train_total ) )
        print( "[epoch num %d ] [ valid: %f]" % ( epoch+1, valid_total ) )
        writer.add_scalars('loss/ono',{'train':train_ono,'valid':valid_ono} ,epoch+1)
        writer.add_scalars('loss/phoneme',{'train':train_phoneme,'valid':valid_phoneme} ,epoch+1)
        writer.add_scalars('loss/total',{'train':train_total,'valid':valid_total} ,epoch+1)
        writer.add_scalars('accuracy/total',{'train':trainaccu,'valid':validaccu} ,epoch+1)
        writer.close()
        if (save_loss >= valid_total):
            torch.save(encoder.state_dict(), 'model/phonemeencoder')
            torch.save(decoder.state_dict(), 'model/phonemedecoder')
            save_loss=valid_total
            print("-------model 更新---------")
        torch.save(encoder.state_dict(), 'model/firstencodercheck')
        torch.save(decoder.state_dict(), 'model/firstdecodercheck')
if __name__ == '__main__':
    main()
    #追加で書き込み
    