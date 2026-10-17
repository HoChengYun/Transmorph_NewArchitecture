# TransMorph × mixed_v2 操作單（AI，cmd 版）

> **這顆在問什麼**：同樣的資料、同樣的訓練量、同樣的評估，只把網路從 VoxelMorph 換成 TransMorph（Swin Transformer）。
> 對照組是 VoxelMorph 專案的 **mix_exp3**（位移場、λ 1，test Dice 0.8062）。設定差異見 `mixv2\README_mixv2.md`。
>
> 跑的機器：**AI**。🔴 **一次只跑一顆**：這顆約用 13 GB，跟 VoxelMorph 那邊的訓練（cascade_pyramid 19 GB、pyramid_500 約 10 GB）不能同時跑。
> 指令全部是 **cmd（命令提示字元）** 語法。
> 最後更新：2026-10-17

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

建虛擬環境（借 VoxelMorph 那邊的 Python 3.10 來建；建好就跟 vxm_env 無關了）：

```bat
D:\chengyun\TRY_voxelmorph\vxm_env\Scripts\activate.bat
```

```bat
cd /d D:\chengyun\Transmorph_NewArchitecture
```

```bat
python -m venv venv
```

```bat
deactivate
```

```bat
venv\Scripts\activate.bat
```

裝套件（torch 約 1.9 GB，要等一陣子；版本跟 vxm_env 一樣）：

```bat
python -m pip install --upgrade pip
```

```bat
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
```

```bat
python -m pip install numpy==2.2.6 scipy==1.15.3 matplotlib==3.10.8 scikit-image==0.25.2 timm==1.0.30 ml_collections==1.1.0 natsort==8.4.0 tensorboard==2.21.0 pystrum==0.4
```

確認：

```bat
python -c "import torch, timm; print(torch.__version__, torch.cuda.is_available(), timm.__version__)"
```

要看到 `2.10.0+cu130 True 1.0.30`。

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
