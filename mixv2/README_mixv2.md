# mixv2：TransMorph 跑在我們的 mixed_v2 資料上

> 2026-10-17 建立。目的：拿 TransMorph（MedIA 2022）跟 VoxelMorph 專案（`claude_cheng`）的模型比，
> 資料、訓練量、評估方式都對齊，**只差網路架構**。
> 官方程式（`IXI/TransMorph/` 等）一個字都沒改；我們加的東西全部在這個資料夾。

資料夾名稱 `mixv2` 指的是資料 `mixed_preprocessed_v2`（四包 FreeSurfer 資料：train 418／val 51／test 51），
VoxelMorph 專案的 mix_exp*、mix_cascade、mix_pyramid 用的都是這一份。

## 檔案

| 檔案 | 內容 |
|---|---|
| `common.py` | 官方基本版設定（`get_3DTransMorph_config`），只換影像大小與 Swin 視窗；存檔、讀檔 |
| `train_mixv2.py` | 訓練（設定跟 mix_exp3 對齊，見下表）|
| `eval_mixv2.py` | Dice、folding 評估，算法跟 VoxelMorph 專案的 `ASD/test_dice.py` 一模一樣 |
| `指令_AI_訓練.md` | AI 那台的 cmd 操作單 |

## 跟官方、跟 VoxelMorph（mix_exp3）的差別

| 項目 | 官方 IXI | **這裡** | mix_exp3 |
|---|---|---|---|
| 資料 | 自己的 IXI `.pkl`，160×192×224 | **直接讀 mixed_v2 的 npz**（不轉檔），192×224×192 | 同左 |
| 對位方向 | atlas → 受試者 | **受試者 → atlas**（輸入第 0 個通道放受試者）| 受試者 → atlas |
| atlas | CycleMorph 提供的 | **MNI152 2009c**（`IXI/atlas_mni152_09c_v3.npz`）| 同左 |
| Swin 視窗 | (5, 6, 7) | **(6, 7, 6)**（四層特徵圖 48×56×48 → … → 6×7×6 都能整除）| — |
| 形變 | 位移場 | 位移場（基本版，不積分）| 位移場 |
| 損失 | −NCC（窗格 9）＋ 1 × 梯度 L2 | 同左（官方 `NCC_vxm`、`Grad3d`）| 同左 |
| 抽樣 | 每輪全部受試者打亂各看一次 | **每一步隨機挑一位（可重複）** | 同左 |
| 訓練量 | 500 輪 × 403 位 ≈ 20 萬步 | **250 輪 × 100 步 = 25,000 步** | 同左 |
| 學習率 | 1e-4，poly 遞減；Adam amsgrad | **1e-4 固定；Adam** | 同左 |
| 資料增強 | 左右翻轉 | **無** | 無 |
| 存檔 | 每輪依 val Dice 存、只留 8 顆 | **每 5 輪存一顆**（一顆約 187 MB）| 每輪（一顆 1.6 MB）|
| 挑 epoch | 訓練中看 val | **練完用 val 每 5 輪評估一次，取最高** | 同左 |
| 標籤怎麼搬（評估）| one-hot 雙線性，取 argmax | **最近鄰** | 最近鄰 |

參數量：4,679 萬（官方大小＋原視窗是 4,677 萬，跟論文的 46.77 M 一致；視窗換成 (6, 7, 6) 多 18,304 個，來自相對位置表）。
**是 mix_exp3（30 萬）的 155 倍**，比較時要一併說明。

## 筆電上的驗證（2026-10-17，RTX 4060 Laptop 8 GB）

1. 全尺寸推論：輸出 192×224×192、剛初始化時位移最大 0.0015 格、推論顯存 6.26 GB、存檔讀回輸出完全一樣
2. 裁成 128×160×128 訓練 3 步（視窗 (4, 5, 4)）：loss 正常、平滑項從 0 開始變大；每步約 0.9 秒、顯存 4.69 GB
3. 沒訓練過的模型（幾乎不變形）對 val 2 位的 Dice：A001 0.6994、A003 0.6581，
   跟 VoxelMorph 專案的 affine 起點（`models/mix_exp2/dice_baseline_val.csv`：0.699374、0.658112）到小數第四位一樣
   → 評估算法跟 `test_dice.py` 一致

全尺寸訓練的估計（影像是試跑的 3.2 倍）：每步約 2.5～3 秒、25,000 步約 17～21 小時；顯存實際約 13 GB。

## 環境

`venv/`（不進 git）：Python 3.10.1、torch 2.10.0+cu130（筆電；**AI 是 2.11.0+cu126**，因為 AI 的驅動最高只支援 CUDA 12.2，跟 AI 的 vxm_env 同版）、numpy 2.2.6、scipy 1.15.3、matplotlib 3.10.8、scikit-image 0.25.2、
timm 1.0.30、ml_collections 1.1.0、natsort 8.4.0、tensorboard 2.21.0、pystrum 0.4（版本都對齊 VoxelMorph 專案的 vxm_env）。

## 用法（筆電）

```powershell
cd D:\MyHome\MRI\Transmorph_NewArchitecture
.\venv\Scripts\activate
# 試跑（裁小、3 步）
python mixv2\train_mixv2.py C:\Users\h4524\claude_cheng\data\mixed_preprocessed_v2\train --atlas C:\Users\h4524\claude_cheng\IXI\atlas_mni152_09c_v3.npz --model-dir runs\test --crop 128 160 128 --max-steps 3
```

正式訓練在 AI 上跑，見 `指令_AI_訓練.md`。
