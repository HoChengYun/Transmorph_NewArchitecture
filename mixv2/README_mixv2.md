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
| `train_mixv2_diff.py` | TransMorph-diff 的訓練（10-24，照官方訓練方式，只改訓練量；見下面「TransMorph-diff」）|
| `verify_diff.py` | TransMorph-diff 包裝的驗證（位移換算、方向、存檔讀回）|
| `指令_AI_訓練_diff.md` | TransMorph-diff 的 AI 操作單 |

## 跟官方、跟 VoxelMorph（mix_exp3）的差別

> 🟡 **tm_mixv2（`runs	m_mixv2\`，test 0.8139）是舊程式練的**：commit `38d3cd5` 的 `train_mixv2.py`，訓練方式全部對齊 mix_exp3（下表「這裡」那欄）。
> 2026-10-24 使用者：「不一定要特別和 mix_exp3 一樣」→ 之後的 TransMorph 改成盡量照官方的訓練方式（TransMorph-diff 已經是；基本版要不要重練另議）。

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
| 資料增強 | 三個方向各 50% 隨機翻轉（atlas 和受試者一起翻；`trans.RandomFlip`）| **無** | 無 |
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

## 結果（2026-10-23，AI 帶回 `runs\tm_mixv2\`）

- 訓練 17.6 小時（TITAN RTX，每步約 2.2～2.5 秒）；參數 46,789,555
- val 每 5 輪評估：約第 85 輪後持平，最高 **第 205 輪 0.8108**
- **test 51 位：Dice 0.8139 ± 0.0124**；folding（atlas 腦內）每位 15,704 個 voxel（0.841%）
- 筆電補算（`--surface --heldout`）：HD95 2.38 mm、SDlogJ 1.975、沒參與訓練的結構 0.681、形變後影像 NCC 0.249；筆電重算的 Dice 跟 AI 的逐人最多差 0.0001

| 對照（VoxelMorph 專案，test 51 位，配對 Wilcoxon）| Dice | TransMorph − 它 | TransMorph 較高 |
|---|---|---|---|
| mix_exp3（設定對齊的那顆，0.30 M）| 0.8062 | +0.0077 | 48/51 |
| mix_wide（加寬 2 倍，位移場）| 0.8114 | +0.0024 | 40/51 |
| mix_pyramid（coarse-to-fine，0.41 M）| 0.8149 | −0.0011 | 17/51 |
| mix_cascade（0.60 M）| 0.8156 | −0.0017 | 17/51 |
| mix_cascade_pyramid（0.82 M）| 0.8192 | −0.0053 | 3/51 |

解讀與後續（tigerbx、簡報）寫在 VoxelMorph 專案 `ASD/ASD相關手冊.md` §30.3。

**10-23 新增**
- `eval_mixv2.py --surface`：HD95、SDlogJ → `surface_NNNN.csv`；`--heldout`：沒參與訓練的結構（Accumbens、Inf-Lat-Vent、胼胝體）、
  形變後影像跟 atlas 的局部 NCC、非背景 MSE → `semisup_check_NNNN.csv`。算法、欄位照抄 VoxelMorph 專案的 `ASD/test_dice.py`、`ASD/check_semisup.py`
- `export_vis.py`：單一受試者的形變後影像、Jacobian determinant、位移場（flow，float16） 存成 npz，給 VoxelMorph 專案的簡報並排圖（那邊的 vxm_env 沒有 timm）

## TransMorph-diff（2026-10-24 加，`runs	m_diff_mixv2\`）

上一顆是基本版（位移場），訓練方式對齊 mix_exp3。老師比較喜歡 SVF，所以另外跑官方的微分同胚版。
使用者選「照官方」（不對齊 mix_exp6），只有訓練量改成跟其他模型一樣 2.5 萬步、翻轉照官方開。
程式全部是新檔；`common.py` 只多了 `DiffAsDisp`、`get_diff_config`、`build_model(arch=...)`，存檔多記 `arch`，舊存檔照舊讀成基本版。

| 項目 | 官方 IXI（`train_TransMorph_diff.py`）| **這裡**（`train_mixv2_diff.py`）|
|---|---|---|
| 網路 | `TransMorphDiff`，設定 `TransMorphDiff-Org-Res`：全解析度 decoder（多 up4）、全解析度 7 步積分（積分前沒有除以 2⁷）| 同，只換影像大小與視窗（同上一顆）；參數 4,679 萬（設定檔註解寫的 1,520 萬是舊的）|
| 輸出 | 機率式：速度場平均＋log 變異數，訓練時抽樣、推論用平均 | 同 |
| 損失 | MSE ÷ σ²（σ = 0.01）＋ KL（prior_lambda 10、prior_lambda_mean 10）| 同（直接叫官方的 `get_sim_loss`、`scale_reg_loss`）|
| 優化器 | Adamax，lr 1e-4，每輪 poly 0.9 遞減 | 同，遞減照 250 輪算 |
| 資料增強 | `trans.RandomFlip`：三個方向各 50% 翻轉，兩張一起翻 | 同（直接叫官方的 `sample()`、`tf()`；它的 `__call__` 用了 Python 3.10 已拿掉的 `collections.Sequence`）|
| 訓練量 | 500 輪 × 每輪全部 | **250 輪 × 100 步 = 2.5 萬步**（同其他模型）|
| 方向、存檔、挑 epoch、評估 | atlas → 受試者；只留 8 顆；one-hot 雙線性 | 同上一顆：受試者 → atlas；每 5 輪存；val 挑；最近鄰 |

⚠️ `TransMorph/models/TransMorph_diff.py`（根目錄那份）預設 `downsample`：decoder 只到 1/2、半解析度積分再放大，放大時位移量沒有乘 2。
IXI 實驗用的是 `IXI/TransMorph/` 這份的 Org-Res，沒有這個問題，我們用這份。

包裝（`common.DiffAsDisp`）：官方 `forward((source, target))` 回傳的 `disp_field` 是 grid_sample 的 [-1, 1] 座標（align_corners=True），
換成 voxel：× (n − 1) / 2。換完 `eval_mixv2.py`、`export_vis.py` 不用改。

**筆電驗證**（`verify_diff.py`，T054）：
1. 全尺寸推論：參數 46,790,854；推論顯存 6.55 GB。剛初始化時位移最大 3.4 voxel（官方 `flow_mean` 的 bias 沒歸零、積分前沒除 2⁷，照原樣）
2. 常數速度 b → 腦內平移剛好 128·b voxel（三軸）；用我們的 SpatialTransformer 搬 = 官方自己搬（最大差 2e-5）
3. 方向 moved(x) = moving(x + u(x))，同 VoxelMorph（最大差 2e-5）
4. 存檔讀回輸出完全一樣；舊的 `runs	m_mixv25.pt` 讀成基本版，test 前 3 位 Dice 跟 10-23 筆電算的 `surface_0205.csv` 逐位相同
   （跟 AI 算的 `dice_0205.csv` 差在小數第五位，是兩台 GPU 的差別，10-23 就是這樣）

試跑（翻轉開）：128×160×128 每步 0.70 秒、顯存實際 4.78 GB；160×192×160 實際 8.67 GB（筆電放不下、溢位變慢）。
外插到全尺寸：實際約 14 GB（上一顆約 13 GB），每步約 2～2.5 秒、2.5 萬步約 15～20 小時。
loss 一開始約 700（影像項 MSE ÷ 0.01² 約 690、KL 約 16），4 步後約 400。
