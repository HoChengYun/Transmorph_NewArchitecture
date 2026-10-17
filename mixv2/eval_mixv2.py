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
args = ap.parse_args()

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
