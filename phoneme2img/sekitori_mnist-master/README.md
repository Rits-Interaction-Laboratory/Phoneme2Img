# Sekitori MNIST

## 学習スクリプト

### train_fixed.py（ランダム教師版）

各入力に対して digit-1・same・digit+1 から教師画像をランダムに選んで学習します。same digit の教師も入力画像そのものではなく、同じ数字の別画像を優先して選びます。

```bash
python train_fixed.py
```

10エポックごとにcheckpointが保存されます。

```text
checkpoints/random_teacher/YYYY-MM-DD/epoch_XXXX.pt
```

**監視用の代表画像（0〜9 各1枚）**

![fixed teachers](fig/fix_teacher.png)

**数字 5 の生成結果（Epoch 1 → 50 → 200）**

同じ入力から潜在空間をサンプリングして10パターン生成。学習が進むにつれて5らしい形が安定して出力されています。

![output fixed 5](fig/output_fixed_5.png)

**数字 8 の生成結果（Epoch 1 → 50 → 244）**

同様に8を生成。エポックが進むと8の形がはっきり現れます。

![output fixed 8](fig/output_fixed_8.png)

### train_random.py（ランダム教師版）❌ 未解決

バッチごとに教師画像をランダムサンプリングする実装です。教師がバッチごとに変わるため席取り割り当てが安定せず、現状ではうまく機能しません。

```bash
python train_random.py
```

**数字 5 の生成結果（Epoch 1 → 100 → 200）**

学習が進んでも出力がノイズ状のまま改善されず、機能していないことがわかります。

![output random 5](fig/output_random_5.png)

## 学習済みcheckpointの解析・可視化

学習後のモデルについて、潜在空間上の分布や生成画像の分布を確認するためのスクリプトとして `analyze_checkpoint.py` を用意しています。

標準では、2026-07-01に学習したランダム教師版の最終checkpointを読み込みます。

```bash
python analyze_checkpoint.py
```

標準入力:

```text
checkpoints/random_teacher/2026-07-01/epoch_0200.pt
```

標準出力先:

```text
analysis/random_teacher/epoch_0200/
```

別のcheckpointや出力先を使う場合は、以下のように指定します。

```bash
python analyze_checkpoint.py \
  --checkpoint checkpoints/random_teacher/2026-07-01/epoch_0100.pt \
  --out-dir analysis/random_teacher/epoch_0100
```

主なオプション:

| オプション | 既定値 | 内容 |
|---|---:|---|
| `--checkpoint` | `checkpoints/random_teacher/2026-07-01/epoch_0200.pt` | 読み込むcheckpoint |
| `--out-dir` | `analysis/random_teacher/epoch_0200` | 解析画像とsummaryの出力先 |
| `--per-digit` | `200` | 各数字についてencoder分布を調べる入力画像数 |
| `--z-samples` | `20` | 各入力画像からサンプルする潜在変数 `z` の数 |
| `--grid-samples` | `12` | 生成画像グリッドに表示する各数字あたりの生成枚数 |

生成されるファイル:

| ファイル | 内容 |
|---|---|
| `epoch_XXXX_latent_mu_pca.png` | encoderが出力した `mu` のPCA散布図 |
| `epoch_XXXX_latent_z_samples_pca.png` | `mu/logvar` からサンプルした `z` のPCA散布図 |
| `epoch_XXXX_generated_grid.png` | 各数字の代表入力から生成した画像グリッド |
| `epoch_XXXX_generated_pixel_pca.png` | 実画像と生成画像を画素空間PCAで比較した散布図 |
| `epoch_XXXX_summary.txt` | checkpoint・epoch・設定・metricsの概要 |

潜在空間の可視化では、まずencoderの `mu` だけでPCA軸をfitします。`z` サンプルの散布図も同じPCA軸へ射影します。また、`z` 図の描画範囲は `mu` 図と同じ `xlim/ylim` に固定しています。遠くに出た `z` は表示上切れますが、これは `mu` 分布に対してサンプル分布がどれだけ広がっているかを同じ座標系で見るためです。

`mu` 図と `z` 図には、各数字クラスタの中心付近に `0`〜`9` のラベルを重ねて表示します。`z` 図では表示範囲内に残っている点の中央値を使ってラベル位置を決めます。

現在の解析結果の詳細は `EXPERIMENT_REPORT.md` にまとめています。

## 生成される画像

学習を実行すると `img_random_teacher/` または `img_random/` 以下に日付フォルダが作られ、以下の画像が保存されます。

### バッチごとに生成

| フォルダ | 内容 |
|---|---|
| `batch_teachers/` | バッチ内の先頭8サンプルについて、入力画像と教師3枚（digit-1・same・digit+1）を横並びで表示 |

### エポック終了時に生成

| フォルダ | 内容 |
|---|---|
| `latent_sekitori_target/` | 追跡数字（`TRACK_DIGIT`）の固定サンプルから生成した N 個の z を、席取り割り当て結果で色分けして PCA 可視化 |
| `latent_sekitori_mse/` | **【重要】** 同じ z を最近傍教師（最小 MSE）で色分けして PCA 可視化 |
| `latent_all_digits/` | 全 10 数字の μ と z サンプルを PCA で2次元に落として可視化 |
| `sekitori_teachers/` | 追跡数字の入力画像と使用された教師3枚を横並びで表示 |
| `digit_variations_5/` | 数字 5 の代表画像を encode し、同じ μ/logvar から 10 パターン生成して比較 |
| `digit_variations_8/` | 数字 8 の代表画像を encode し、同じ μ/logvar から 10 パターン生成して比較 |
| `teachers/` | 監視用の代表画像（0〜9 各1枚）の一覧（`train_fixed.py` のみ） |
