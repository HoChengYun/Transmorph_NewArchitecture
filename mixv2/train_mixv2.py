# -*- coding: utf-8 -*-
"""TransMorph（基本版）在 mixed_v2 上訓練，設定跟 VoxelMorph 專案的 mix_exp3 對齊，只差網路架構。

🟡 tm_mixv2（test 0.8139）就是這支（commit 38d3cd5）練的。2026-10-24 起之後的 TransMorph 改成盡量照官方訓練方式（見 README），這支算舊程式。

跟 mix_exp3（ASD/run_train.py → voxelmorph-code/scripts/torch/train.py）一樣的地方：
  - 受試者 → atlas（scan-to-atlas）；atlas = IXI/atlas_mni152_09c_v3.npz（MNI152 2009c）
  - 每一步隨機挑一位受試者（可重複），batch 1；每輪 100 步、250 輪 = 25,000 步
  - 損失 = −NCC（窗格 9）＋ λ × 位移場梯度的 L2，λ = 1（官方 NCC_vxm、Grad3d，就是 VoxelMorph 的同一套公式）
  - Adam，學習率固定 1e-4；不做資料增強
  - 之後用 val 每 5 輪挑一次 epoch（eval_mixv2.py --step 5）
存檔：每 --save-every（預設 5）輪的開頭存一顆 NNNN.pt，最後再存一顆 NNNN（= --epochs）.pt。
  一顆約 187 MB（4,679 萬個參數；VoxelMorph 一顆只有 1.6 MB），每輪都存的話 250 輪約 47 GB → 只存挑 epoch 會用到的那些（51 顆、約 9.5 GB）
跟官方 train_TransMorph.py 不同的地方（2026-10-17 我建議「跟我們對齊」、使用者同意）：
  - 官方：每輪跑完全部訓練資料、500 輪（約 20 萬步）、學習率 poly 遞減、Adam amsgrad、三個方向各 50% 隨機翻轉（atlas 和受試者一起翻）、每輪用 val 存檔只留 8 個
log 每一步一行，格式同 VoxelMorph 專案的 train_arch.py（epoch: NNNN  step: k/100  time: …  loss: 總 (影像項, 平滑項)），
ASD/plot_loss_curve.py 可以直接讀。計時前先 torch.cuda.synchronize()，每步時間才準。

用法（筆電試跑，只跑 2 步、中間裁 128×160×128）：
  python mixv2\\train_mixv2.py <train_dir> --atlas <atlas.npz> --model-dir runs\\test --crop 128 160 128 --max-steps 2
"""
import os
import sys
import glob
import time
import argparse

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('train_dir', help='訓練資料夾（npz：vol、seg；seg 訓練時不用）')
ap.add_argument('--atlas', required=True, help='atlas 影像 npz（key: vol）')
ap.add_argument('--model-dir', required=True)
ap.add_argument('--epochs', type=int, default=250)
ap.add_argument('--steps-per-epoch', type=int, default=100)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--lambda', dest='lam', type=float, default=1.0, help='平滑項權重（同 VoxelMorph 的 --lambda）')
ap.add_argument('--gpu', default='0')
ap.add_argument('--save-every', type=int, default=5, help='每幾輪存一顆（要跟 eval_mixv2.py 的 --step 一樣）')
ap.add_argument('--initial-epoch', type=int, default=0)
ap.add_argument('--load-model', help='從這顆 .pt 接著練（搭配 --initial-epoch）')
ap.add_argument('--seed', type=int, default=None)
ap.add_argument('--crop', type=int, nargs=3, help='測試用：只取中間這麼大一塊（每邊要能被 32 整除），模型照這個大小蓋')
ap.add_argument('--max-steps', type=int, default=0, help='測試用：總共只跑幾步就停，並印顯存峰值')
args = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_model, save_model, load_model, IMG_SIZE, WINDOW   # noqa: E402
import losses                                   # noqa: E402  官方 IXI/TransMorph/losses.py（common.py 已加進 sys.path）

if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

files = sorted(glob.glob(os.path.join(args.train_dir, '*.npz')))
if not files:
    sys.exit('[X] %s 裡沒有 npz' % args.train_dir)
atlas = np.load(args.atlas)['vol'].astype(np.float32)
if atlas.shape != IMG_SIZE:
    sys.exit('[X] atlas 大小 %s，預期 %s' % (atlas.shape, IMG_SIZE))


def crop(v):
    if not args.crop:
        return v
    s = [(n - c) // 2 for n, c in zip(v.shape, args.crop)]
    return v[s[0]:s[0] + args.crop[0], s[1]:s[1] + args.crop[1], s[2]:s[2] + args.crop[2]]


if args.crop:
    img_size = tuple(args.crop)
    window = tuple(n // 32 for n in img_size)        # 最底層 = 影像 / 32，視窗剛好蓋住整張（同正式設定的關係）
else:
    img_size, window = IMG_SIZE, WINDOW

device = 'cuda'
os.makedirs(args.model_dir, exist_ok=True)
if args.load_model:
    model = load_model(args.load_model, device)
    assert model.mixv2_shape == (img_size, window), '--load-model 的大小 %s 跟這次的 %s 不一樣' % (model.mixv2_shape, (img_size, window))
else:
    model = build_model(img_size, window).to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
ncc = losses.NCC_vxm()                       # 窗格 9，回傳 −mean(cc)
grad = losses.Grad3d(penalty='l2')
atlas_t = torch.from_numpy(crop(atlas))[None, None].to(device)

log = open(os.path.join(args.model_dir, 'train_log.txt'), 'a', encoding='utf-8')


def say(msg):
    print(msg, flush=True)
    log.write(msg + '\n')
    log.flush()


n_par = sum(p.numel() for p in model.parameters())
say('指令：' + ' '.join(sys.argv))
say('架構：TransMorph（基本版、位移場），共 %d 個參數；影像 %s、Swin 視窗 %s；影像項 ncc（窗格 9），λ = %g；'
    '%d 位訓練資料；每輪 %d 步、到第 %d 輪；lr %g（固定）' % (n_par, 'x'.join(map(str, img_size)), window, args.lam,
                                                   len(files), args.steps_per_epoch, args.epochs, args.lr))
done = 0
for epoch in range(args.initial_epoch, args.epochs):
    if epoch % args.save_every == 0:
        save_model(model, os.path.join(args.model_dir, '%04d.pt' % epoch), epoch=epoch)
    for step in range(args.steps_per_epoch):
        t0 = time.time()
        m = np.load(files[np.random.randint(len(files))])['vol'].astype(np.float32)
        x = torch.from_numpy(crop(m))[None, None].to(device)
        warped, flow = model(torch.cat([x, atlas_t], dim=1))     # 官方：第 0 個通道是 moving、第 1 個是 fixed
        l_sim = ncc(atlas_t, warped)
        l_reg = args.lam * grad(flow, atlas_t)
        loss = l_sim + l_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        say('epoch: %04d  step: %d/%d  time: %.2f sec  loss: %.6f  (%.6f, %.6f)'
            % (epoch + 1, step + 1, args.steps_per_epoch, time.time() - t0, loss.item(), l_sim.item(), l_reg.item()))
        done += 1
        if args.max_steps and done >= args.max_steps:
            say('--max-steps %d 到了，停止（測試用）。顯存峰值：實際用到 %.2f GB、PyTorch 預留 %.2f GB'
                % (args.max_steps, torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30))
            save_model(model, os.path.join(args.model_dir, '%04d.pt' % (epoch + 1)), epoch=epoch + 1)
            sys.exit(0)
save_model(model, os.path.join(args.model_dir, '%04d.pt' % args.epochs), epoch=args.epochs)
say('完成。')
