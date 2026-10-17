# -*- coding: utf-8 -*-
"""TransMorph × mixed_v2（我們的 ASD／FreeSurfer 資料）共用的設定與函式。

官方程式（IXI/TransMorph/）一個字都不改，這裡只 import。跟官方的差別（2026-10-17 使用者確認）：
  - 影像大小 192×224×192（官方 160×192×224）→ Swin 視窗 (6, 7, 6)（官方 (5, 6, 7)）。
    四層特徵圖是 48×56×48 → 24×28×24 → 12×14×12 → 6×7×6，每層都能被 (6, 7, 6) 整除，不必補零。
    （官方程式會把除不盡的特徵圖補零到視窗的倍數，所以用 (5, 6, 7) 也能跑，但會多算一堆沒意義的零。）
  - 方向：受試者 → atlas（scan-to-atlas，跟我們的 VoxelMorph 一樣）。官方 IXI 是 atlas → 受試者。
  - 版本：基本版 TransMorph（直接輸出位移場、不積分），對照 mix_exp3（位移場、λ 1）。
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
TM_DIR = os.path.join(os.path.dirname(HERE), 'IXI', 'TransMorph')
if TM_DIR not in sys.path:
    sys.path.insert(0, TM_DIR)

import models.configs_TransMorph as configs          # noqa: E402  官方設定檔
from models.TransMorph import TransMorph, SpatialTransformer   # noqa: E402,F401  官方模型

IMG_SIZE = (192, 224, 192)
WINDOW = (6, 7, 6)
# Dice 評估用的 30 個 FreeSurfer 結構（同 VoxelMorph 專案 voxelmorph-code/data/labels.npz、論文 Table I）
LABELS = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24,
          28, 31, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60, 63]


def get_config(img_size=IMG_SIZE, window=WINDOW):
    """官方的基本版設定（get_3DTransMorph_config），只換影像大小與視窗。"""
    c = configs.get_3DTransMorph_config()
    c.img_size = tuple(img_size)
    c.window_size = tuple(window)
    return c


def build_model(img_size=IMG_SIZE, window=WINDOW):
    model = TransMorph(get_config(img_size, window))
    model.mixv2_shape = (tuple(img_size), tuple(window))      # 存檔時要記下來，讀回來才能蓋出同樣的網路
    return model


def save_model(model, path, **extra):
    img_size, window = model.mixv2_shape
    torch.save({'state_dict': model.state_dict(), 'img_size': img_size, 'window_size': window, **extra}, path)


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    model = build_model(ck['img_size'], ck['window_size'])
    model.load_state_dict(ck['state_dict'])
    return model.to(device)
