# -*- coding: utf-8 -*-
"""TransMorph（mixv2）對單一受試者的形變後影像與 Jacobian determinant，存成 npz 給 VoxelMorph 專案的簡報圖用。2026-10-23。

claude_cheng 的 ASD/slides_src/2026-10-28_redpen/make_ants_vis.py 把 ANTs、VoxelMorph、TransMorph 並排；
那支在 vxm_env 跑、沒有 TransMorph 需要的 timm，所以 TransMorph 的推論在這裡（venv）先做好、存檔交給它讀。
  warped：model 輸出之形變後影像（同 make_ants_vis.py 對 VoxelMorph 取 moved）
  det   ：det(I + ∇u)，中央差分（同 make_ants_vis.py 的 jac_det、test_dice.py 的 jacobian_det）
  flow  ：位移場 u（3, D, H, W，float16，voxel），給 make_paper_vis.py 畫變形網格

用法
  python mixv2\\export_vis.py --model runs\\tm_mixv2\\0205.pt --npz <data>\\test\\T054.npz --atlas <atlas.npz> --out <xxx.npz>
"""
import os
import sys
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--npz', required=True, help='受試者的 npz（vol）')
ap.add_argument('--atlas', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--gpu', default='0')
args = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model                   # noqa: E402

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
atlas = np.load(args.atlas)['vol'].astype(np.float32)
vol = np.load(args.npz)['vol'].astype(np.float32)
model = load_model(args.model, dev).eval()
with torch.no_grad():
    x = torch.cat([torch.from_numpy(vol)[None, None], torch.from_numpy(atlas)[None, None]], dim=1).to(dev)
    moved, flow = model(x)                      # 受試者 → atlas（同訓練、同 eval_mixv2.py）
f = flow[0].cpu().numpy()
g = [[np.gradient(f[c], axis=a) for a in range(3)] for c in range(3)]
det = ((1 + g[0][0]) * ((1 + g[1][1]) * (1 + g[2][2]) - g[1][2] * g[2][1])
       - g[0][1] * (g[1][0] * (1 + g[2][2]) - g[1][2] * g[2][0])
       + g[0][2] * (g[1][0] * g[2][1] - (1 + g[1][1]) * g[2][0]))
# flow：10-23 加，給簡報的變形網格用（float16 省空間；單位 voxel，同 VoxelMorph：moved(x) = moving(x + u(x))）
np.savez_compressed(args.out, warped=moved[0, 0].cpu().numpy(), det=det.astype(np.float32), flow=f.astype(np.float16))
print('->', args.out, '；腦內 |J| <= 0：%d 個' % int(((det <= 0) & (atlas > 0)).sum()))
