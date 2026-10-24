# -*- coding: utf-8 -*-
"""TransMorph（mixv2）的 Dice 與 folding 評估，算法跟 VoxelMorph 專案的 ASD/test_dice.py 一模一樣，數字可以直接比。

  - 受試者的 seg 用模型的位移場、**最近鄰**搬到 atlas 空間，跟 atlas 的 FreeSurfer seg 逐結構算 Dice
    （官方 infer_TransMorph.py 是把標籤拆成 one-hot 再用雙線性搬、取 argmax，分數會比較高；為了跟我們的模型比，不用那個）
  - 30 個結構（common.LABELS），某結構兩邊都沒有時不計入平均（nan）
  - folding：Jacobian determinant 用中央差分（np.gradient），|J| ≤ 0 的比例
      jneg_pct     分母是整個影像（舊定義）
      jneg_fg_pct  分母是 atlas 非背景 voxel（atlas_vol > 0）← 報告用這個；跟論文比一律用 voxel 數
  - CSV 欄位：file, dice_mean, jneg_pct, jneg_fg_pct, label_<編號>…（同 test_dice.py）

用法
  # 用 val 挑 epoch（每 5 輪一顆）→ <model-dir>\\dice_curve_val.csv
  python mixv2\\eval_mixv2.py --model-dir runs\\tm_mixv2 --step 5 --test-dir <data>\\val --atlas <atlas.npz> --atlas-seg <atlas_seg.npz>
  # 單一模型對 test（只跑一次）→ <model 所在資料夾>\\dice_NNNN.csv
  python mixv2\\eval_mixv2.py --model runs\\tm_mixv2\\NNNN.pt --test-dir <data>\\test --atlas <atlas.npz> --atlas-seg <atlas_seg.npz>

2026-10-23 加（只給 --model）：
  --surface  另算 HD95、SDlogJ → surface_NNNN.csv（欄位、算法同 claude_cheng 的 ASD/test_dice.py --surface）
  --heldout  另算未參與訓練之結構（Accumbens、Inf-Lat-Vent、胼胝體）之 Dice、形變後影像與 atlas 之局部 NCC、非背景 MSE
             → semisup_check_NNNN.csv（欄位、算法同 ASD/check_semisup.py；影像用雙線性搬）
"""
import os
import sys
import csv
import glob
import argparse

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument('--model', help='單一 .pt')
g.add_argument('--model-dir', help='資料夾，掃過裡面的 .pt')
ap.add_argument('--step', type=int, default=5, help='--model-dir 時每幾輪評估一次（另外一定包含最後一顆）')
ap.add_argument('--test-dir', required=True)
ap.add_argument('--atlas', required=True)
ap.add_argument('--atlas-seg', required=True)
ap.add_argument('--out-csv', default=None)
ap.add_argument('--max-subjects', type=int, default=0, help='測試用：只評估前幾位')
ap.add_argument('--gpu', default='0')
ap.add_argument('--surface', action='store_true', help='另算 HD95、SDlogJ（只給 --model）')
ap.add_argument('--heldout', action='store_true', help='另算未參與訓練之結構與影像 NCC（只給 --model）')
args = ap.parse_args()
if (args.surface or args.heldout) and not args.model:
    sys.exit('[X] --surface／--heldout 只給 --model')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, SpatialTransformer, LABELS   # noqa: E402

device = 'cuda' if torch.cuda.is_available() else 'cpu'
atlas_vol = np.load(args.atlas)['vol'].astype(np.float32)
atlas_seg = np.load(args.atlas_seg)['seg'].astype(np.int32)
assert atlas_vol.shape == atlas_seg.shape
FG = atlas_vol > 0
atlas_t = torch.from_numpy(atlas_vol)[None, None].to(device)
warp_nn = SpatialTransformer(atlas_vol.shape, mode='nearest').to(device)
warp_lin = SpatialTransformer(atlas_vol.shape, mode='bilinear').to(device)
test_files = sorted(glob.glob(os.path.join(args.test_dir, '*.npz')))
if args.max_subjects:
    test_files = test_files[:args.max_subjects]
if not test_files:
    sys.exit('[X] %s 裡沒有 npz' % args.test_dir)
split = os.path.basename(os.path.normpath(args.test_dir))


def jacobian_det(flow):
    """det(I + ∇u)，中央差分（同 test_dice.py）。flow: (3, X, Y, Z)。"""
    d = [[np.gradient(flow[c], axis=a) for a in range(3)] for c in range(3)]
    j11, j12, j13 = 1 + d[0][0], d[0][1], d[0][2]
    j21, j22, j23 = d[1][0], 1 + d[1][1], d[1][2]
    j31, j32, j33 = d[2][0], d[2][1], 1 + d[2][2]
    return (j11 * (j22 * j33 - j23 * j32)
            - j12 * (j21 * j33 - j23 * j31)
            + j13 * (j21 * j32 - j22 * j31))


def dice(a, b, lab):
    x, y = (a == lab), (b == lab)
    s = x.sum() + y.sum()
    return np.nan if s == 0 else 2.0 * (x & y).sum() / s


# ── 以下照抄 claude_cheng：ASD/test_dice.py（sdlogj、hd95）、ASD/check_semisup.py（HELD、dice_set、row 的 ncc、mse_fg）──
from scipy import ndimage                       # noqa: E402
import torch.nn.functional as F                 # noqa: E402
HELD = {'accumbens_L': [26], 'accumbens_R': [58], 'inflatvent_L': [5], 'inflatvent_R': [44],
        'corpus_callosum': [251, 252, 253, 254, 255]}


def sdlogj(det):
    """log|J| 的標準差（atlas 非背景內）。|J| ≤ 0 先截到 1e-9（同 Learn2Reg）。"""
    return float(np.log(np.clip(det[FG], 1e-9, 1e9)).std())


def hd95(x, y):
    """兩個二值標籤的 95% Hausdorff distance（voxel = 1 mm）；兩個方向各取第 95 百分位，回傳較大者。"""
    if not x.any() or not y.any():
        return np.nan
    idx = np.argwhere(x | y)
    lo, hi = np.maximum(idx.min(0) - 2, 0), idx.max(0) + 3
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    x, y = x[sl], y[sl]
    bx = x & ~ndimage.binary_erosion(x)
    by = y & ~ndimage.binary_erosion(y)
    to_x = ndimage.distance_transform_edt(~bx)
    to_y = ndimage.distance_transform_edt(~by)
    return float(max(np.percentile(to_y[bx], 95), np.percentile(to_x[by], 95)))


def dice_set(a, b, labs):
    x, y = np.isin(a, labs), np.isin(b, labs)
    s = x.sum() + y.sum()
    return np.nan if s == 0 else 2.0 * (x & y).sum() / s


def ncc_loss(I, J, win=9):
    """局部 NCC（窗格 9），回傳 −mean(cc)；同 voxelmorph/torch/losses.py 的 NCC（裝置跟著輸入）。"""
    sum_filt = torch.ones([1, 1, win, win, win], device=I.device)
    pad = win // 2
    I2, J2, IJ = I * I, J * J, I * J
    I_sum, J_sum, I2_sum, J2_sum, IJ_sum = (F.conv3d(t, sum_filt, stride=1, padding=pad) for t in (I, J, I2, J2, IJ))
    n = float(win ** 3)
    u_I, u_J = I_sum / n, J_sum / n
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * n
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * n
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * n
    return float(-torch.mean(cross * cross / (I_var * J_var + 1e-5)))


SROWS, HROWS = [], []                           # --surface、--heldout 的每位結果


def evaluate(path):
    model = load_model(path, device)
    model.eval()
    rows = []
    with torch.no_grad():
        for f in test_files:
            d = np.load(f)
            vol, seg = d['vol'].astype(np.float32), d['seg'].astype(np.int32)
            v = torch.from_numpy(vol)[None, None].to(device)
            _, flow = model(torch.cat([v, atlas_t], dim=1))          # 受試者 → atlas（同訓練）
            s = torch.from_numpy(seg.astype(np.float32))[None, None].to(device)
            seg_w = warp_nn(s, flow)[0, 0].cpu().numpy()
            if np.abs(seg_w - np.round(seg_w)).max() > 1e-4:
                sys.exit('[X] 搬完的標籤出現非整數值 —— 內插法錯了')
            seg_w = np.round(seg_w).astype(np.int32)
            per = {lab: dice(seg_w, atlas_seg, lab) for lab in LABELS}
            det = jacobian_det(flow[0].cpu().numpy())
            r = {'file': os.path.basename(f), 'dice_mean': float(np.nanmean([per[l] for l in LABELS])),
                 'jneg_pct': 100 * float((det <= 0).mean()), 'jneg_fg_pct': 100 * float((det[FG] <= 0).mean())}
            r.update({'label_%d' % l: per[l] for l in LABELS})
            rows.append(r)
            if args.surface:
                hd = {l: hd95(seg_w == l, atlas_seg == l) for l in LABELS}
                SROWS.append({'file': r['file'], 'dice_mean': r['dice_mean'], 'jneg_pct': r['jneg_pct'],
                              'jneg_fg_pct': r['jneg_fg_pct'], 'hd95_mean': float(np.nanmean([hd[l] for l in LABELS])),
                              'sdlogj': sdlogj(det), 'hd': hd})
            if args.heldout:
                moved = warp_lin(v, flow)
                h = {'file': r['file'], 'dice30': float(np.nanmean([dice_set(seg_w, atlas_seg, [l]) for l in LABELS]))}
                for k, labs in HELD.items():
                    h[k] = dice_set(seg_w, atlas_seg, labs)
                h['heldout_mean'] = float(np.nanmean([h[k] for k in HELD]))
                h['ncc'] = -ncc_loss(atlas_t, moved)
                h['mse_fg'] = float(((moved[0, 0].cpu().numpy() - atlas_vol)[FG] ** 2).mean())
                HROWS.append(h)
            print('  %-24s Dice %.4f  folding（非背景）%.4f%%（%d 個）'
                  % (r['file'], r['dice_mean'], r['jneg_fg_pct'], int((det[FG] <= 0).sum())), flush=True)
    return rows


def write(rows, out):
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ('%.6f' % v if isinstance(v, float) else v) for k, v in r.items()})
    print('  CSV ->', out)


if args.model:
    rows = evaluate(args.model)
    ep = os.path.splitext(os.path.basename(args.model))[0]
    out = args.out_csv or os.path.join(os.path.dirname(os.path.abspath(args.model)),
                                       'dice_%s%s.csv' % (ep, '' if split == 'test' else '_' + split))
    write(rows, out)
    print('  %s：Dice %.4f ± %.4f（%d 位）；folding（非背景）平均 %.4f%%'
          % (ep, np.mean([r['dice_mean'] for r in rows]), np.std([r['dice_mean'] for r in rows]), len(rows),
             np.mean([r['jneg_fg_pct'] for r in rows])))
    tail = '' if split == 'test' else '_' + split
    md = os.path.dirname(os.path.abspath(args.model))
    if SROWS:
        o = os.path.join(md, 'surface_%s%s.csv' % (ep, tail))
        with open(o, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['file', 'dice_mean', 'jneg_pct', 'jneg_fg_pct', 'hd95_mean', 'sdlogj'] + ['hd95_label_%d' % l for l in LABELS])
            for r in SROWS:
                w.writerow([r['file'], '%.6f' % r['dice_mean'], '%.6f' % r['jneg_pct'], '%.6f' % r['jneg_fg_pct'],
                            '%.4f' % r['hd95_mean'], '%.6f' % r['sdlogj']] + ['%.4f' % r['hd'][l] for l in LABELS])
        print('  CSV ->', o)
        print('  HD95 %.3f mm；SDlogJ %.4f' % (np.mean([r['hd95_mean'] for r in SROWS]), np.mean([r['sdlogj'] for r in SROWS])))
    if HROWS:
        o = os.path.join(md, 'semisup_check_%s%s.csv' % (ep, tail))
        with open(o, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(HROWS[0].keys()))
            w.writeheader()
            w.writerows(HROWS)
        print('  CSV ->', o)
        print('  未參與訓練之結構 %.4f；NCC %.4f' % (np.nanmean([r['heldout_mean'] for r in HROWS]), np.mean([r['ncc'] for r in HROWS])))
else:
    pts = sorted(glob.glob(os.path.join(args.model_dir, '[0-9][0-9][0-9][0-9].pt')))
    eps = [int(os.path.basename(p)[:4]) for p in pts]
    keep = [p for p, e in zip(pts, eps) if e % args.step == 0 or e == max(eps)]
    curve = []
    for p in keep:
        print('== %s' % os.path.basename(p), flush=True)
        rows = evaluate(p)
        curve.append({'epoch': int(os.path.basename(p)[:4]),
                      'dice_mean': float(np.mean([r['dice_mean'] for r in rows])),
                      'jneg_pct': float(np.mean([r['jneg_pct'] for r in rows])),
                      'jneg_fg_pct': float(np.mean([r['jneg_fg_pct'] for r in rows]))})
        print('   -> Dice %.4f' % curve[-1]['dice_mean'], flush=True)
    out = args.out_csv or os.path.join(args.model_dir, 'dice_curve_%s.csv' % split)
    write(curve, out)
    best = max(curve, key=lambda c: c['dice_mean'])
    print('  最佳：epoch %d，Dice %.4f' % (best['epoch'], best['dice_mean']))
