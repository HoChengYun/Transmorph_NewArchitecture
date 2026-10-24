# -*- coding: utf-8 -*-
"""TransMorph-diff 包裝（common.DiffAsDisp）的驗證，2026-10-24。

  1. 全尺寸蓋得起來：參數量、推論顯存
  2. 位移換算：把 flow_mean 設成常數速度 b（權重歸零），官方積分 7 次（沒有先除 2^7）→ 平移 128·b voxel；
     包裝輸出的 flow 在腦內要等於 128·b，用我們的 SpatialTransformer（eval_mixv2.py 用的）搬影像要跟官方自己搬的一樣
  3. 方向：moved(x) = moving(x + u(x))（同 VoxelMorph）
  4. 存檔讀回：輸出完全一樣；舊存檔（沒有 arch）讀成基本版

用法：python mixv2\\verify_diff.py --npz <一位受試者.npz> --atlas <atlas.npz> [--old runs\\tm_mixv2\\0205.pt]
"""
import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_model, save_model, load_model, SpatialTransformer, IMG_SIZE   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True)
ap.add_argument('--atlas', required=True)
ap.add_argument('--old', help='舊的基本版存檔，確認讀得回來')
ap.add_argument('--tmp', default=os.path.join(os.environ.get('TEMP', '.'), 'verify_diff.pt'))
args = ap.parse_args()
dev = 'cuda'
ok = True


def check(name, cond, msg=''):
    global ok
    ok &= bool(cond)
    print('[%s] %s %s' % ('OK' if cond else 'X', name, msg), flush=True)


vol = np.load(args.npz)['vol'].astype(np.float32)
atl = np.load(args.atlas)['vol'].astype(np.float32)
x = torch.cat([torch.from_numpy(vol)[None, None], torch.from_numpy(atl)[None, None]], dim=1).to(dev)

# 1. 全尺寸
model = build_model(arch='tm_diff').to(dev).eval()
n = sum(p.numel() for p in model.parameters())
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    w0, f0 = model(x)
check('全尺寸推論', tuple(w0.shape[2:]) == IMG_SIZE and tuple(f0.shape) == (1, 3) + IMG_SIZE,
      '參數 %d；剛初始化位移最大 %.4f voxel；推論顯存峰值 %.2f GB' % (n, f0.abs().max().item(), torch.cuda.max_memory_allocated() / 2 ** 30))

# 2. 常數速度 → 平移
b = torch.tensor([0.02, -0.015, 0.01])
with torch.no_grad():
    model.net.flow_mean.weight.zero_()
    model.net.flow_mean.bias.copy_(b)
    warped, flow = model(x)
inner = flow[0, :, 40:-40, 40:-40, 40:-40]
expect = 128 * b
for d in range(3):
    v = inner[d]
    check('平移 軸 %d' % d, torch.allclose(v, torch.full_like(v, expect[d].item()), atol=1e-3),
          '腦內 %.4f～%.4f，預期 %.4f voxel' % (v.min().item(), v.max().item(), expect[d].item()))
st = SpatialTransformer(IMG_SIZE, mode='bilinear').to(dev)
with torch.no_grad():
    ours = st(x[:, :1], flow)
diff = (ours - warped).abs()
check('我們的 SpatialTransformer 搬 = 官方搬', diff.max().item() < 1e-4, '最大差 %.2e' % diff.max().item())

# 3. 方向：整數平移時，moved[i] 應該等於 moving[i + t]
with torch.no_grad():
    model.net.flow_mean.bias.copy_(torch.tensor([3., -2., 1.]) / 128)
    warped, flow = model(x)
t = [3, -2, 1]
mv = vol[40 + t[0]:-40 + t[0], 40 + t[1]:-40 + t[1], 40 + t[2]:-40 + t[2]]
wv = warped[0, 0, 40:-40, 40:-40, 40:-40].cpu().numpy()
check('方向 moved(x) = moving(x + u)', np.abs(wv - mv).max() < 1e-3, '最大差 %.2e' % np.abs(wv - mv).max())

# 4. 存檔讀回
save_model(model, args.tmp, epoch=0)
m2 = load_model(args.tmp, dev).eval()
with torch.no_grad():
    w2, f2 = m2(x)
check('存檔讀回', m2.mixv2_arch == 'tm_diff' and torch.equal(w2, warped) and torch.equal(f2, flow))
os.remove(args.tmp)
if args.old:
    mo = load_model(args.old, dev)
    check('舊存檔讀成基本版', mo.mixv2_arch == 'tm' and type(mo).__name__ == 'TransMorph')
print('全部通過' if ok else '[X] 有沒通過的')
sys.exit(0 if ok else 1)
