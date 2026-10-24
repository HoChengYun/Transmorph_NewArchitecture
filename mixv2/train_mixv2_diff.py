# -*- coding: utf-8 -*-
"""TransMorph-diff（官方 IXI/TransMorph/train_TransMorph_diff.py）在 mixed_v2 上訓練。

2026-10-24 使用者選「照官方」（不對齊 mix_exp6），只有訓練量改成跟其他模型一樣 2.5 萬步、翻轉照官方開。
跟官方一樣的地方：
  - 網路：TransMorphDiff，設定 'TransMorphDiff-Org-Res'（全解析度 decoder、7 步積分；機率式：速度場平均＋log 變異數，訓練時抽樣）
  - 損失：官方 model.get_sim_loss()（MSE ÷ image_sigma²，σ = 0.01）＋ model.scale_reg_loss()（KL，prior_lambda 10、prior_lambda_mean 10）
  - Adamax，lr 1e-4，每輪 poly 遞減：lr = 1e-4 × (1 − epoch / 總輪數)^0.9（同官方 adjust_learning_rate，四捨五入到小數 8 位）
  - 資料增強：官方 trans.RandomFlip —— 三個方向各 50% 翻轉，受試者和 atlas 一起翻
    （官方 Base.__call__ 用了 Python 3.10 已拿掉的 collections.Sequence，所以直接叫它的 sample()、tf()，效果一樣）
跟官方不一樣的地方（同 train_mixv2.py）：
  - 影像 192×224×192、Swin 視窗 (6, 7, 6)；受試者 → atlas（官方 atlas → 受試者）
  - 訓練量：每步隨機挑一位（可重複），每輪 100 步、250 輪 = 25,000 步（官方 500 輪 × 每輪全部，約 20 萬步）
  - 每 --save-every 輪存一顆，練完用 eval_mixv2.py 以 val Dice 挑（官方訓練中看 val、只留 8 顆）
log 格式同 train_mixv2.py：loss: 總 (影像項, 正則項)。

用法（筆電試跑）：
  python mixv2\\train_mixv2_diff.py <train_dir> --atlas <atlas.npz> --model-dir runs\\test_diff --crop 128 160 128 --max-steps 2
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
ap.add_argument('--no-flip', action='store_true', help='關掉翻轉（預設照官方開）')
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
from data import trans                          # noqa: E402  官方 IXI/TransMorph/data/trans.py（common.py 已加進 sys.path）

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
    assert model.mixv2_arch == 'tm_diff', '--load-model 不是 TransMorph-diff'
    assert model.mixv2_shape == (img_size, window), '--load-model 的大小 %s 跟這次的 %s 不一樣' % (model.mixv2_shape, (img_size, window))
else:
    model = build_model(img_size, window, arch='tm_diff').to(device)
net = model.net                                  # 官方 TransMorphDiff
net.train()
optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=0)     # 同官方
flip = trans.RandomFlip(0)                       # 同官方；axis 參數官方本來就沒用，固定 (1, 2, 3)
atlas_c = crop(atlas)[None]                      # (1, X, Y, Z)，同官方 dataset 的形狀

log = open(os.path.join(args.model_dir, 'train_log.txt'), 'a', encoding='utf-8')


def say(msg):
    print(msg, flush=True)
    log.write(msg + '\n')
    log.flush()


def poly_lr(epoch):
    """同官方 train_TransMorph_diff.py 的 adjust_learning_rate（每一步都設、同一輪內不變）。"""
    return round(args.lr * np.power(1 - epoch / args.epochs, 0.9), 8)


n_par = sum(p.numel() for p in model.parameters())
say('指令：' + ' '.join(sys.argv))
say('架構：TransMorph-diff（官方 TransMorphDiff-Org-Res），共 %d 個參數；影像 %s、Swin 視窗 %s；損失 MSE/σ²（σ = %g）＋ KL'
    '（prior_lambda %g、prior_lambda_mean %g）；%d 位訓練資料；每輪 %d 步、到第 %d 輪；Adamax lr %g（poly 0.9 遞減）；翻轉 %s'
    % (n_par, 'x'.join(map(str, img_size)), window, net.image_sigma, net.prior_lambda, net.prior_lambda_mean,
       len(files), args.steps_per_epoch, args.epochs, args.lr, '關' if args.no_flip else '開（三個方向各 50%）'))
done = 0
for epoch in range(args.initial_epoch, args.epochs):
    if epoch % args.save_every == 0:
        save_model(model, os.path.join(args.model_dir, '%04d.pt' % epoch), epoch=epoch)
    for g in optimizer.param_groups:
        g['lr'] = poly_lr(epoch)
    for step in range(args.steps_per_epoch):
        t0 = time.time()
        m = crop(np.load(files[np.random.randint(len(files))])['vol'].astype(np.float32))[None]
        a = atlas_c
        if not args.no_flip:
            flip.sample()                        # 同官方 Base.__call__：抽一次，兩張用同一組
            m, a = flip.tf(m), flip.tf(a)
        x = torch.from_numpy(np.ascontiguousarray(m))[None].to(device)     # (1, 1, X, Y, Z)
        y = torch.from_numpy(np.ascontiguousarray(a))[None].to(device)
        net((x, y))                              # 受試者 → atlas
        l_sim = net.get_sim_loss()
        l_reg = net.scale_reg_loss()
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
