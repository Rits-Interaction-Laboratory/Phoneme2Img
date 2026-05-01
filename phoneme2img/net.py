import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from utils import gram_matrix,draw_pca_plot_vae_samples
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
        # new_hidden=F.normalize(new_hidden,p=2,dim=2)
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
        # self.softmax     = nn.LogSoftmax( dim = 1 )
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
        # compressed=F.normalize(compressed, p=2)
        'tanhしないと、prompt_converterに入れれないっぽい。F.normalizeは多分意味ない'
        # compressed = torch.nn.functional.layer_norm(compressed, compressed.shape[-1:])
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
        # self.final_activation = nn.Tanh()

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
        # x = self.final_activation(x)
        #------
        # x=F.normalize(x, p=2) # Tanh関数は-1~1の間にするやつ、F.normalizeやと、長さ(合計)が1になる
        # x = torch.nn.functional.layer_norm(x, x.shape[-1:])
        #------
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

    def forward(self, x, num_samples=100): #num_samplesで出力するアウトプットの数を調整、何も指定しなかったら普通のVAE
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
        return z, ave, log_dev # z:１つだけサンプリングしたもの,ave:平均,log_dev:標準偏差
        # zはSDに通すやつ→正規化必要かも？、log_devが大きくないと数字が散らばらない


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

        #------
        # mu=F.normalize(mu, p=2)
        #------
        return mu, log_var


class PhonemeVAE(nn.Module):
    """
    Encoder + Decoder (複数サンプル対応)
    log_devを1（対数分散0）に固定してサンプリングする
    """
    def __init__(self, input_dim=128, z_dim=128, output_dim=77 * 1024, num_samples=1):
        super().__init__()
        self.num_samples = num_samples
        self.encoder = PhonemeEncoder(input_dim, z_dim)
        self.decoder = PhonemeDecoder(z_dim, output_dim)
        self.final_activation = nn.Tanh()

    def reparameterize_fixed(self, ave, num_samples):
        """
        平均 ave [batch_size, z_dim] を受け取り、
        分散を 1 固定で [num_samples, batch_size, z_dim] を生成する
        """
        # ave を [1, batch_size, z_dim] に拡張
        ave_expanded = ave.unsqueeze(0)
        
        # 期待する出力形状 [num_samples, batch_size, z_dim] と同じ形の標準正規分布ノイズを作成
        # これにより分散が1(標準偏差1)に固定される
        eps = torch.randn(num_samples, *ave.shape, device=ave.device)
        # randnにより、分散が1の正規分布が計算される。
        # ここはlog_devの数字は関係ない、zを正規分布からサンプリングしなおしているだけ。
        eps2 = torch.randn(500, *ave.shape, device=ave.device)
        # z = μ + ε * 1.0 (std=1.0)
        return ave_expanded + eps * 0.2 ,ave_expanded + eps2 * 0.2  #←最終サンプリングされる100個のベクトル

    def forward(self, x, epoch, nums, idx, ONO):
        """
        x: [batch_size, input_dim]
        """
        # 1. エンコーダから平均(ave)と予測された対数分散(log_dev)を取得
        # ※内部でサンプリングされている場合は、aveのみを利用する形に上書きします
        z_original, ave, log_dev = self.encoder(x, self.num_samples)

        # 2. 分散を1に固定してサンプリングし直す (zを上書き)
        # z: [num_samples, batch_size, z_dim]
        z,z2 = self.reparameterize_fixed(ave, self.num_samples)

        if idx % 10 == 0:
            with torch.no_grad():
                amiami_features = []
                amiami_img_features = []
                ave_expanded = ave.unsqueeze(0)
                amiami_img_features.append(ave_expanded.to(torch.float32).reshape(-1).detach().cpu().numpy())

            #     # print(epoch+1, "エポック目の", idx,"バッチ目")
            #zは100個、z2は上の関数内で指定した個数
                for i in range(z.size(0)):
                    single_sample = z[i:i+1]
                    amiami_features.append(single_sample.to(torch.float32).reshape(-1).detach().cpu().numpy())
                common_limits = ([-1.0, 1.0], [-1.0, 1.0])
                draw_pca_plot_vae_samples(epoch,nums,idx,amiami_features,amiami_img_features,pca_model=None, limits=common_limits, dir="VAEsampling_KL",mode="KL1e2_std0.2_z",ono=f"{ONO}")


        # 3. デコーダに渡す
        mu, log_var = self.decoder(z)

        # 元の入出力形式を維持
        return mu, log_var, z, ave, log_dev


class PhonemeVAE1(nn.Module):
    """
    Encoder + Decoder (複数サンプル対応)
    """
    def __init__(self, input_dim=128, z_dim=128, output_dim=77 * 1024,num_samples=1):
        super().__init__()
        self.num_samples=num_samples
        self.encoder = PhonemeEncoder(input_dim, z_dim)
        self.decoder = PhonemeDecoder(z_dim, output_dim)

        self.final_activation = nn.Tanh()

    def forward(self, x,):
        """
        x: [batch_size, input_dim]
        num_samples: 生成する潜在変数の数
        """
        z, ave, log_dev = self.encoder(x, self.num_samples)  # [num_samples, batch_size, z_dim]
        mu, log_var = self.decoder(z)  # [num_samples, batch_size, 77, 1024]
        #------
        # mu=F.normalize(mu, p=2)
        # mu = self.final_activation(mu)
        #------
        return mu, log_var, z, ave, log_dev # mu:楕円の中心,log_var:楕円の角度,z:１つだけサンプリングしたもの,ave:平均,log_dev:標準偏差
        # zはSDに通すやつ→正規化必要かも？、log_devが大きくないと数字が散らばらない
        'mu→VAEによって再構成された埋め込みベクトル 77*1024'
        'log_var→再構成分布の分散 77*1024'
        'z→100個サンプリングしたときの潜在変数 128'
        'ave→潜在空間上の平均 z_dim=128'
        'log_dev→潜在空間上の分散。潜在変数の分散の対数。KLダイバージェンスの計算に使う 128'


class ImageEncoder(nn.Module):
    def __init__(self, input_dim=77 * 1024, z_dim=128):
        super().__init__()
        # 入力が巨大なため、段階的に次元を落とす
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_ave = nn.Linear(256, z_dim)  # 平均 μ
        self.fc_dev = nn.Linear(256, z_dim)  # 分散 log(σ^2)
        self.relu = nn.ReLU()

    def forward(self, x, num_samples=100):
        # x: [batch_size, 77, 1024] -> [batch_size, 78848] へ平坦化
        batch_size = x.size(0)
        x = x.view(batch_size, -1) 

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        ave = self.fc_ave(x)
        log_dev = self.fc_dev(x)

        # 再パラメータ化
        eps = torch.randn(num_samples, batch_size, ave.size(-1), device=ave.device)
        z = ave.unsqueeze(0) + torch.exp(log_dev.unsqueeze(0) / 2) * eps
        return z, ave, log_dev

class ImageDecoder(nn.Module):
    def __init__(self, z_dim=128, output_dim=77 * 1024):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.mu = nn.Linear(1024, output_dim)
        self.log_var = nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        # z: [num_samples, batch_size, z_dim]
        num_samples, batch_size, z_dim = z.shape
        z = z.view(-1, z_dim) # 全サンプルをバッチとして処理

        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        mu = self.mu(x) # [num_samples * batch_size, 78848]
        log_var = self.log_var(x)

        # 形を [num_samples, batch_size, 77, 1024] に戻す
        mu = mu.view(num_samples, batch_size, 77, 1024)
        log_var = log_var.view(num_samples, batch_size, 77, 1024)
        return mu,log_var

class ImageVAE(nn.Module):
    def __init__(self, z_dim=128, num_samples=100):
        super().__init__()
        self.num_samples = num_samples
        self.encoder = ImageEncoder(input_dim=77 * 1024, z_dim=z_dim)
        self.decoder = ImageDecoder(z_dim=z_dim, output_dim=77 * 1024)

    def forward(self, x):
        # x: [batch_size, 77, 1024]
        z, ave, log_dev = self.encoder(x, self.num_samples)
        mu, log_var = self.decoder(z) # [num_samples, batch_size, 77, 1024]
        
        return mu, log_var, z, ave, log_dev