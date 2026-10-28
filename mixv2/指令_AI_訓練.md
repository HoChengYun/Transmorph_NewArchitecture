# TransMorph × mixed_v2 操作單（AI，cmd 版）

> **這顆在問什麼**：同樣的資料、同樣的訓練量、同樣的評估，只把網路從 VoxelMorph 換成 TransMorph（Swin Transformer）。
> 對照組是 VoxelMorph 專案的 **mix_exp3**（位移場、λ 1，test Dice 0.8062）。設定差異見 `mixv2\README_mixv2.md`。
>
> 跑的機器：**AI**。🔴 **一次只跑一顆**：這顆約用 13 GB，跟 VoxelMorph 那邊的訓練（cascade_pyramid 19 GB、pyramid_500 約 10 GB）不能同時跑。
> 指令全部是 **cmd（命令提示字元）** 語法。
> 最後更新：2026-10-17（建環境改以 virtualenv 為主，venv 當備案；AI 的 torch 改成 2.11.0+cu126，跟 AI 的 vxm_env 一樣）
>
> 🟡 **2026-10-24 起 `train_mixv2.py` 預設改成官方訓練方式**。下面的訓練指令是 tm_mixv2（舊設定）當初用的；要重現 tm_mixv2 請在訓練指令最後加 `--legacy`。
>
> 🟡 **2026-10-28 起 `train_mixv2.py`、`train_mixv2_diff.py` 的預設改成官方的「每輪全部訓練資料洗牌、500 輪」**（以前每輪隨機抽 100 位、250 輪）。下面的訓練指令都有寫 `--epochs 250 --steps-per-epoch 100`，照抄仍是舊做法。照原作重現的操作單在 VoxelMorph 專案的 `ASD/指令_reproduce.md`。

---

## cmd 要注意的三件事

1. **每個指令都是一整行**：整行複製、貼上、按 Enter。看起來換行的地方是畫面自動折行，不是真的斷開
2. **換磁碟要用 `cd /d`**：只打 `cd D:\...` 的話，如果目前在 C 槽，會停在原地不動
3. 虛擬環境啟動成功的話，提示字元前面會多一個 `(venv)` 或 `(vxm_env)`

---

## 0. 第一次：下載程式、建環境（只要做一次）

```bat
cd /d D:\chengyun
```

```bat
git clone https://github.com/HoChengYun/Transmorph_NewArchitecture.git
```

建虛擬環境：用跟 vxm_env 同一個 Python 3.10（torch 2.10.0+cu130 要用這一版）。建好就跟 vxm_env 無關了。

先查 vxm_env 用的是哪一個 Python：

```bat
type D:\chengyun\TRY_voxelmorph\vxm_env\pyvenv.cfg
```

看 `home = ...` 那一行（例如 `home = C:\Users\xxx\.pyenv\pyenv-win\versions\3.10.1`），下面的 `<home 路徑>` 換成它。

```bat
cd /d D:\chengyun\Transmorph_NewArchitecture
```

**用 virtualenv 建**（前後的雙引號要保留）：

```bat
virtualenv venv -p "<home 路徑>\python.exe"
```

| 出現 | 怎麼辦 |
|---|---|
| `'virtualenv' 不是內部或外部命令` | 改打 `"<home 路徑>\python.exe" -m virtualenv venv` |
| 還是不行（`No module named virtualenv`）| 先 `"<home 路徑>\python.exe" -m pip install virtualenv`，再打上一行 |

<details><summary>不用 virtualenv、改用 Python 內建的 venv 也可以（建出來一樣）</summary>

venv 用的是「執行它的那個 Python」，所以先啟動 vxm_env 借它的 Python 3.10，建好再關掉：

```bat
D:\chengyun\TRY_voxelmorph\vxm_env\Scripts\activate.bat
```

```bat
python -m venv venv
```

```bat
deactivate
```

</details>

啟動新環境：

```bat
venv\Scripts\activate.bat
```

裝套件（torch 要等一陣子；**版本跟 AI 上的 vxm_env 一樣：torch 2.11.0+cu126**）。
⚠️ 不要裝筆電用的 `+cu130`：AI 的驅動（537.42）最高只支援 CUDA 12.2，cu130 的 torch 會抓不到 GPU（`cuda.is_available()` 是 False）；cu126 在 AI 上可以用（vxm_env 就是這版）。

```bat
python -m pip install --upgrade pip
```

```bat
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
```

```bat
python -m pip install numpy==2.2.6 scipy==1.15.3 matplotlib==3.10.8 scikit-image==0.25.2 timm==1.0.30 ml_collections==1.1.0 natsort==8.4.0 tensorboard==2.21.0 pystrum==0.4
```

確認：

```bat
python -c "import torch, timm; print(torch.__version__, torch.cuda.is_available(), timm.__version__)"
```

要看到 `2.11.0+cu126 True 1.0.30`。

如果之前已經裝成 cu130（看到 `2.10.0+cu130 False`），先移除再裝上面那行：`python -m pip uninstall -y torch torchvision`

---

## 1. 每次開始前

```bat
cd /d D:\chengyun\Transmorph_NewArchitecture
```

```bat
venv\Scripts\activate.bat
```

```bat
git pull
```

```bat
nvidia-smi
```

確認沒有其他訓練在用 GPU。資料用 VoxelMorph 那邊已經有的，**不用搬**：
`D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\`、`D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz`、`atlas_mni152_09c_v3_seg.npz`。

---

## 2. 先試跑 3 步（約 1 分鐘）

```bat
python mixv2\train_mixv2.py D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\train --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --model-dir runs\test --max-steps 3
```

最後一行會印 `顯存峰值：實際用到 X GB、PyTorch 預留 Y GB`。**把這行跟我說**（預估實際約 13 GB），沒問題再開始正式訓練。
試跑完可以刪掉：

```bat
rmdir /s /q runs\test
```

---

## 3. 正式訓練（約 17～21 小時）

先打這一行（只管記憶體、不改計算）：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.85
```

然後在**同一個視窗**：

```bat
python mixv2\train_mixv2.py D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\train --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --model-dir runs\tm_mixv2 --epochs 250 --steps-per-epoch 100 --lambda 1.0 --gpu 0
```

- log 會自動寫到 `runs\tm_mixv2\train_log.txt`（畫面上也會印），**不用自己加 `>`**
- 每 5 輪存一顆 `NNNN.pt`（一顆約 187 MB，共 51 顆、約 9.5 GB），確認 D 槽空間夠

| 看到 | 怎麼辦 |
|---|---|
| 噴 `CUDA out of memory` | 把 `set` 那行的 `0.85` 改成 `0.9` 重打，再從第 3 步的訓練指令重來（或照下面的「中斷了」接著跑）|
| 每步**超過 6 秒** | 不對勁：`Ctrl + C` 停掉，先跟我說，**不要自己改參數** |

**中斷了怎麼辦**：照第 1 步進資料夾、啟動環境，再打一次 `set` 那行；看 `runs\tm_mixv2\` 裡**編號最大**的 `.pt`（例如 `0120.pt`），然後（兩個地方都換成那個數字）：

```bat
python mixv2\train_mixv2.py D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\train --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --model-dir runs\tm_mixv2 --epochs 250 --steps-per-epoch 100 --lambda 1.0 --gpu 0 --initial-epoch 120 --load-model runs\tm_mixv2\0120.pt
```

⚠️ 接著練時 Adam 的狀態會重新累積（沒存在 .pt 裡），前幾步 loss 可能跳一下，屬正常。

---

## 4. 練完：用 val 挑 epoch

🔴 **不要拿 test 挑 epoch。**

```bat
python mixv2\eval_mixv2.py --model-dir runs\tm_mixv2 --step 5 --test-dir D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\val --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --atlas-seg D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3_seg.npz --gpu 0
```

輸出 `runs\tm_mixv2\dice_curve_val.csv`，最後一行印出「最佳：epoch N」。

---

## 5. test 評估（只跑一次）

把 `NNNN` 換成第 4 步挑出來的 epoch（四位數，例如 `0235`）：

```bat
python mixv2\eval_mixv2.py --model runs\tm_mixv2\NNNN.pt --test-dir D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\test --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --atlas-seg D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3_seg.npz --gpu 0
```

輸出 `runs\tm_mixv2\dice_NNNN.csv`。

---

## 6. 收尾、帶回來

帶回主機的東西：

```
runs\tm_mixv2\NNNN.pt               最佳 epoch 的權重（約 187 MB）
runs\tm_mixv2\dice_curve_val.csv    挑 epoch 的依據
runs\tm_mixv2\dice_NNNN.csv         test 結果
runs\tm_mixv2\train_log.txt         訓練 log（含當初的指令）
```

帶回來之後跟 mix_exp3、mix_exp6、mix_cascade、mix_pyramid 逐人配對比較，我來算。
其他 `.pt` 確認不需要之後再刪（每顆 187 MB）。
