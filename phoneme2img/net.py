import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from utils import gram_matrix
#phoneme network--------------------------------------------------------------
# Start core part
class Encoder( nn.Module ):
    def __init__( self, input_size, embedding_size, hidden_size):
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
        return torch.zeros( 1, 1, self.hidden_size )

    def forward( self, _input, hidden ):
        # 単語のベクトル化
        embedded        = self.embedding( _input ).view( 1, 1, -1 )
        out, new_hidden = self.gru( embedded, hidden )
        # new_hidden=new_hidden/(torch.norm(new_hidden)) #F.normalizeと同じ処理ではある
        new_hidden=F.normalize(new_hidden,p=2,dim=2)
        return out, new_hidden
    
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
        return torch.zeros( 1, 1, self.hidden_size )
    
#image network--------------------------------------------------------------
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self,x):
        return x.view(self.shape)

class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        self.vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.layers ={
                        '0': 'conv1_1',
                       '5': 'conv2_1',
                       '10': 'conv3_1',
                    #    '19': 'conv4_1',
                    #    '21': 'conv4_2',
                    #    '28': 'conv5_1'
                    }
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x.clone()
        return features

class TextureNet(nn.Module):
    def __init__(self, fc_out_dim=128, img_size=64, img_channels=3):
        super(TextureNet, self).__init__()
        self.img_size = img_size
        self.extractor =VGGFeatures()#VGG16で特徴を抽出する部分
        # 86016次元から128次元に圧縮するための段階的な全結合層
        self.fc1 = nn.Linear(86016, 4096)   # 最初に大きく次元を削減
        self.ln1 = nn.LayerNorm(4096)     # バッチ正規化で学習を安定化
        self.fc2 = nn.Linear(4096, 1024)    # 次に1024次元に減らす
        self.ln2 = nn.LayerNorm(1024)     # バッチ正規化
        self.fc3 = nn.Linear(1024, 256)     # さらに256次元に減らす
        self.ln3 = nn.LayerNorm(256)      # バッチ正規化
        self.fc4 = nn.Linear(256, 128)      # 最終的に128次元に圧縮する

    def forward(self, x):
        features = self.extractor(x)
        gram_features = [gram_matrix(f).view(f.size(0), -1) for f in features.values()]
        concatenated_features = torch.cat(gram_features, dim=1)
        compressed = F.relu(self.ln1(self.fc1(concatenated_features)))   # ReLU + Layer Normalization
        compressed = F.relu(self.ln2(self.fc2(compressed)))   # ReLU + Layer Normalization
        compressed = F.relu(self.ln3(self.fc3(compressed)))   # ReLU + Layer Normalization
        compressed = F.tanh(self.fc4(compressed))
        compressed=F.normalize(compressed, p=2)
        return compressed 
    

#prompt converter-----------------------------------------------------------
class PromptEncoder(nn.Module): #PromptEncoderの構造を改良し、データセットも1680枚の画像から3700枚の画像に増やして学習させたモデル構造
    def __init__(self):
        super(PromptEncoder, self).__init__()
        self.fc1 = nn.Linear(128, 256)# 128 次元 -> 256 次元
        self.fc2 = nn.Linear(256, 77 * 1024)# 256 次元 -> 77*1024 次元
        
        self.ln1 = nn.LayerNorm(256)# 中間ベクトル(256次元)に対する LayerNorm
        self.ln2 = nn.LayerNorm(77*1024)# 埋め込み次元(77*1024)に対する LayerNorm

        # 出力範囲を [-1, 1] に
        self.final_activation = nn.Tanh()

    def forward(self, x):
        """
        x: shape = (batch_size, 128)
        """
        # fc1 -> ReLU -> LayerNorm
        x = self.fc1(x)
        x = F.relu(x)
        x = self.ln1(x)

        # fc2
        x = self.fc2(x)
        x = self.ln2(x)
        # (N, 77*1024) -> (N, 77, 1024) に reshape
        x = x.view(-1, 77, 1024)



        # [-1, 1] に正規化(Tanh)
        x = self.final_activation(x)

        return x



#phoneme2img converter--------------------------------------------------
class PhonemeEncoder(nn.Module):
    def __init__(self, input_dim=128, z_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_ave = nn.Linear(128, z_dim)   # 平均（μ）を出力
        self.fc_dev = nn.Linear(128, z_dim)   # log(σ^2) を出力
        self.relu = nn.ReLU()

    def forward(self, x, num_samples=1): #num_samplesで出力するアウトプットの数を調整、何も指定しなかったら普通のVAE
        """
        x: [batch_size, input_dim]
        num_samples: 生成する潜在変数の数
        """
        # 2段の全結合層を通す
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # 平均と分散の対数を計算
        ave = self.fc_ave(x)     # μ
        log_dev = self.fc_dev(x) # log(σ^2)

        # 再パラメータ化トリック（複数サンプル）
        eps = torch.randn(num_samples, *ave.shape, device=ave.device)
        z = ave.unsqueeze(0) + torch.exp(log_dev.unsqueeze(0) / 2) * eps  # [num_samples, batch_size, z_dim]
        
        return z, ave, log_dev


class PhonemeDecoder(nn.Module):
    def __init__(self, z_dim=128, output_dim=77 * 1024):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.mu = nn.Linear(512, output_dim)
        self.log_var = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        """
        z: [num_samples, batch_size, z_dim]
        """
        num_samples, batch_size, z_dim = z.shape
        z = z.view(-1, z_dim)  # [num_samples * batch_size, z_dim]

        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        mu = self.mu(x)
        log_var = self.log_var(x)

        # 形状を戻す
        mu = mu.view(num_samples, batch_size, 77, 1024)
        log_var = log_var.view(num_samples, batch_size, 77, 1024)
        
        return mu, log_var


class PhonemeVAE(nn.Module):
    """
    Encoder + Decoder (複数サンプル対応)
    """
    def __init__(self, input_dim=128, z_dim=128, output_dim=77 * 1024,num_samples=1):
        super().__init__()
        self.num_samples=num_samples
        self.encoder = PhonemeEncoder(input_dim, z_dim)
        self.decoder = PhonemeDecoder(z_dim, output_dim)

    def forward(self, x,):
        """
        x: [batch_size, input_dim]
        num_samples: 生成する潜在変数の数
        """
        z, ave, log_dev = self.encoder(x, self.num_samples)  # [num_samples, batch_size, z_dim]
        mu, log_var = self.decoder(z)  # [num_samples, batch_size, 77, 1024]
        
        return mu, log_var, z, ave, log_dev
    