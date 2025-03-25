#カテゴライズが難しい関数はここに
import torch
import torch.nn as nn
# グラム行列計算
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (h * w)
    return gram


def tensorFromSentence( lang, sentence,EOS_token,device): #sentenceをインデックス番号に変換したtensor配列にする 
    indexes = [ lang.word2index[ word ] for word in sentence.split(' ') ] #sentenceの音素をインデックス番号に変換してリストにする
    indexes.append( EOS_token )
    

    return torch.tensor( indexes, dtype=torch.long ).to( device ).view(-1, 1)


def select_top_k_outputs(target, mu_p,log_var_p,top_k=10):
    # 各 mu_p[i] と target 間の MSE を各バッチごとに計算
    # nn.MSELoss(reduction='none') は各要素ごとの損失を返すので、mean(dim=(1,2))でバッチ内の全要素の平均を取る
    losses = torch.stack([
        nn.MSELoss(reduction='none')(out, target).mean(dim=(1, 2))
        for out in mu_p
    ], dim=0)  # shape: (num_samples, batch_size)

    # 各バッチごとに MSE が最小の上位 top_k のインデックスを取得
    best_indices = torch.argsort(losses, dim=0)[:top_k]  # shape: (top_k, batch_size)

    # 各バッチで、上位 top_k の mu_p の出力を選択
    batch_indices = torch.arange(mu_p.shape[1]).unsqueeze(0).expand(top_k, -1)  # shape: (top_k, batch_size)
    best_outputs = mu_p[best_indices, batch_indices].to(dtype=torch.bfloat16).requires_grad_(True)  # shape: (top_k, batch_size, 77, 1024)
    best_log_var_p = log_var_p[best_indices, batch_indices].to(dtype=torch.bfloat16).requires_grad_(True)     # shape: (top_k, batch_size, 77, 1024)
    # target を top_k 個に複製（先頭に次元を追加して expand）
    expanded_target = target.unsqueeze(0).expand(top_k, -1, -1, -1)  # shape: (top_k, batch_size, 77, 1024)
    return best_outputs, expanded_target,best_log_var_p
