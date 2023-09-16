# ライブラリのインポート
import streamlit as st
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import cv2


# タイトル
st.title('黄金律構図判定')
st.subheader('写真の構図が黄金律か判定します！')

# ダミー画像2枚
original_img = np.zeros((380, 600, 3), dtype=np.uint8)
original_img.fill(255)
#processed_img = np.zeros((380, 600, 3), dtype=np.uint8)
#processed_img.fill(255)

# ファイルアップローダー
upload_file = st.file_uploader("横長の画像を読み込んでください", type=['jpg', 'png'])

# 画像のレイアウト
image_container = st.empty()
col1, col2 = st.columns(2)
# 画像を配置
with col1:
     st.write('実行前')
     if upload_file is None:
        st.image(original_img,use_column_width=True)
     else:
          st.image(upload_file,use_column_width=True)
with col2:
     st.write('実行後')
     


if upload_file is not None:

    img = Image.open(upload_file).convert("RGB") 
    # img_array = np.array(img)
    # st.image(img_array,caption='読み込んだ画像', use_column_width= True)

    if st.button("処理を開始"):
        # 入力画像の変形
        # 新しい高さを指定
        new_height = 400

        # アスペクト比を計算して新しい幅を決定
        width_percent = (new_height / float(img.size[1]))
        new_width = int((float(img.size[0]) * float(width_percent)))

        # 画像をリサイズ
        resized_image = img.resize((new_width, new_height), Image.ANTIALIAS)

        # センタークロップの範囲を計算
        left = (new_width - 600) // 2
        top = (new_height - 380) // 2
        right = left + 600
        bottom = top + 380

        # 画像をセンタークロップ
        cropped_image = resized_image.crop((left, top, right, bottom))

        # ここからプログレスバーを開始
        progress_bar = st.progress(0)

        # 深度画像の作成
        # ZoeD_N
        from zoedepth.models.builder import build_model
        from zoedepth.utils.config import get_config

        conf = get_config("zoedepth", "infer")
        model_zoe_n = build_model(conf)  

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        zoe = model_zoe_n.to(DEVICE)
        # depth画像をPILでアウトプット
        depth_numpy = zoe.infer_pil(cropped_image)

        from zoedepth.utils.misc import colorize
        colored_depth = colorize(depth_numpy) # カラーのnumpy配列を作成

        # depth_numpyを評価モデルに渡す
        # RGBA形式のNumPy画像データをPIL形式に変換
        rgba_pil_image = Image.fromarray(colored_depth)

        # PIL形式の画像をグレースケールに変換
        gray_pil_image = rgba_pil_image.convert('L')

        # PIL形式のグレースケール画像をPyTorchのTensorに変換し、形状を変更
        gray_tensor = torch.from_numpy(np.array(gray_pil_image)).unsqueeze(0)


        # プログレスバーを完了
        progress_bar.progress(100)


        # ネットワークの定義
        class Net(pl.LightningModule):

            def __init__(self):
                super().__init__()

                self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
                self.bn = nn.BatchNorm2d(3)
                self.fc = nn.Linear(171000,3)

            def forward(self,x):
                h = self.conv(x)
                h = F.relu(h)
                h = self.bn(h)
                h = F.max_pool2d(h, kernel_size=2, stride=2)
                h = h.view(-1, 171000)
                h = self.fc(h)
                return h

            def training_step(self, batch, batch_idx):
                    x, t = batch
                    y = self(x)
                    loss = F.cross_entropy(y, t)
                    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
                    self.log('train_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=3, top_k=1), on_step=True, on_epoch=True, prog_bar=True)
                    return loss

            def validation_step(self, batch, batch_idx):
                    x, t = batch
                    y = self(x)
                    loss = F.cross_entropy(y, t)
                    self.log('val_loss', loss, on_step=False, on_epoch=True)
                    self.log('val_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=3, top_k=1), on_step=False, on_epoch=True)
                    return loss

            def test_step(self, batch, batch_idx):
                    x, t = batch
                    y = self(x)
                    loss = F.cross_entropy(y, t)
                    self.log('test_loss', loss, on_step=False, on_epoch=True)
                    self.log('test_acc', accuracy(y.softmax(dim=-1), t, task='multiclass', num_classes=3, top_k=1), on_step=False, on_epoch=True)
                    return loss


            def configure_optimizers(self):
                    optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
                    return optimizer
        
        # ネットワークの準備
        net = Net().cpu().eval()

        # 重みの読み込み
        net.load_state_dict(torch.load('golden.pt', map_location=torch.device('cpu')))

        x = gray_tensor.to(torch.float32) 

        # 予測値の算出
        y = net(x.unsqueeze(0))
        # 確率に変換
        y = F.softmax(y)

        # 予測ラベル
        y = torch.argmax(y)
        print(y)

        # 予測結果を表示
        if y == 0:
            st.subheader(f"黄金律構図です！")
            gold_img = Image.open('1-3.png')
            base = cropped_image
            base_np = np.array(base)
            gold_img_np = np.array(gold_img)
            with col2:
                 # 透過画像を背景画像に貼り付ける
                for c in range(0, 3):
                    base_np[:, :, c] = \
                        base_np[:, :, c] * (1 - gold_img_np[:, :, 3] / 255.0) + \
                        gold_img_np[:, :, c] * (gold_img_np[:, :, 3] / 255.0)
                processed_img = Image.fromarray(base_np)
                st.image(processed_img,use_column_width=True)
        elif y == 1:
            st.subheader(f"黄金律構図です！")
            gold_img = Image.open('2-4.png')
            base = cropped_image
            base_np = np.array(base)
            gold_img_np = np.array(gold_img)
            with col2:
                 # 透過画像を背景画像に貼り付ける
                for c in range(0, 3):
                    base_np[:, :, c] = \
                        base_np[:, :, c] * (1 - gold_img_np[:, :, 3] / 255.0) + \
                        gold_img_np[:, :, c] * (gold_img_np[:, :, 3] / 255.0)
                processed_img = Image.fromarray(base_np)
                st.image(processed_img,use_column_width=True)
        else:
            st.subheader('黄金律構図ではないようです')
            with col2:
                 st.image(upload_file,use_column_width=True)

