# TransMorph-diff × mixed_v2 操作單（AI，cmd 版）

> **這顆在問什麼**：官方發表的 **TransMorph-diff**（微分同胚版，SVF 積分）在我們資料上多少分。
> 上一顆 tm_mixv2 是基本版（位移場），訓練方式對齊 mix_exp3；**這顆照官方的訓練方式**
> （機率式輸出、MSE＋KL 損失、Adamax、lr 遞減、三個方向隨機翻轉），只有訓練量跟其他模型一樣 2.5 萬步（2026-10-24 使用者決定）。
> 跟官方的差別、驗證結果見 `mixv2\README_mixv2.md` 的「TransMorph-diff」一節。
>
> 跑的機器：**AI**。🔴 **一次只跑一顆**：這顆預估約用 14 GB（筆電外插），不能跟 VoxelMorph 那邊的訓練同時跑。
> 指令全部是 **cmd（命令提示字元）** 語法。環境沿用上一顆建好的 `venv`，只要多裝一個套件。
> 最後更新：2026-10-24

---

## cmd 要注意的三件事

1. **每個指令都是一整行**：整行複製、貼上、按 Enter。看起來換行的地方是畫面自動折行，不是真的斷開
2. **換磁碟要用 `cd /d`**：只打 `cd D:\...` 的話，如果目前在 C 槽，會停在原地不動
3. 虛擬環境啟動成功的話，提示字元前面會多一個 `(venv)`

---

## 0. 第一次：多裝一個套件（只要做一次）

```bat
cd /d D:\chengyun\Transmorph_NewArchitecture
```

```bat
venv\Scripts\activate.bat
```

```bat
git pull
```

官方 TransMorph-diff 的程式要用 `future` 這個套件（筆電裝的是 1.0.0）：

```bat
python -m pip install future==1.0.0
```

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

確認沒有其他訓練在用 GPU。資料用 VoxelMorph 那邊已經有的，**不用搬**（同上一顆）。

---

## 2. 先驗證、試跑 3 步（約 2 分鐘）

先跑驗證（位移換算、方向、存檔讀回，筆電上全部通過；最後一行要是「全部通過」）：

```bat
python mixv2\verify_diff.py --npz D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\test\T054.npz --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --old runs\tm_mixv2\0205.pt
```

再試跑 3 步：

```bat
python mixv2\train_mixv2_diff.py D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\train --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --model-dir runs\test_diff --max-steps 3
```

最後一行會印 `顯存峰值：實際用到 X GB、PyTorch 預留 Y GB`。**把這行跟我說**（預估實際約 14 GB），沒問題再開始正式訓練。
loss 一開始是幾百（影像項是 MSE ÷ 0.01²，官方就是這個量級），屬正常。試跑完可以刪掉：

```bat
rmdir /s /q runs\test_diff
```

---

## 3. 正式訓練（約 15～20 小時）

先打這一行（只管記憶體、不改計算）：

```bat
set PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.85
```

然後在**同一個視窗**：

```bat
python mixv2\train_mixv2_diff.py D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\train --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --model-dir runs\tm_diff_mixv2 --epochs 250 --steps-per-epoch 100 --gpu 0
```

- log 會自動寫到 `runs\tm_diff_mixv2\train_log.txt`（畫面上也會印），**不用自己加 `>`**
- 每 5 輪存一顆 `NNNN.pt`（一顆約 187 MB，共 51 顆、約 9.5 GB），確認 D 槽空間夠
- 學習率會隨輪數慢慢降（官方做法），第一行 log 會寫「Adamax lr 0.0001（poly 0.9 遞減）；翻轉 開」

| 看到 | 怎麼辦 |
|---|---|
| 噴 `CUDA out of memory` | 把 `set` 那行的 `0.85` 改成 `0.9` 重打，再從第 3 步的訓練指令重來（或照下面的「中斷了」接著跑）|
| 每步**超過 6 秒** | 不對勁：`Ctrl + C` 停掉，先跟我說，**不要自己改參數** |

**中斷了怎麼辦**：照第 1 步進資料夾、啟動環境，再打一次 `set` 那行；看 `runs\tm_diff_mixv2\` 裡**編號最大**的 `.pt`（例如 `0120.pt`），然後（兩個地方都換成那個數字）：

```bat
python mixv2\train_mixv2_diff.py D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\train --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --model-dir runs\tm_diff_mixv2 --epochs 250 --steps-per-epoch 100 --gpu 0 --initial-epoch 120 --load-model runs\tm_diff_mixv2\0120.pt
```

⚠️ 接著練時 Adamax 的狀態會重新累積（沒存在 .pt 裡），前幾步 loss 可能跳一下，屬正常；學習率會照輪數接回去。

---

## 4. 練完：用 val 挑 epoch

🔴 **不要拿 test 挑 epoch。** 評估程式跟上一顆同一支（會自己認出是 TransMorph-diff）：

```bat
python mixv2\eval_mixv2.py --model-dir runs\tm_diff_mixv2 --step 5 --test-dir D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\val --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --atlas-seg D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3_seg.npz --gpu 0
```

輸出 `runs\tm_diff_mixv2\dice_curve_val.csv`，最後一行印出「最佳：epoch N」。

---

## 5. test 評估（只跑一次）

把 `NNNN` 換成第 4 步挑出來的 epoch（四位數，例如 `0235`）：

```bat
python mixv2\eval_mixv2.py --model runs\tm_diff_mixv2\NNNN.pt --test-dir D:\chengyun\TRY_voxelmorph\data\mixed_preprocessed_v2\test --atlas D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3.npz --atlas-seg D:\chengyun\TRY_voxelmorph\IXI\atlas_mni152_09c_v3_seg.npz --gpu 0
```

輸出 `runs\tm_diff_mixv2\dice_NNNN.csv`。

---

## 6. 收尾、帶回來

```
runs\tm_diff_mixv2\NNNN.pt               最佳 epoch 的權重（約 187 MB）
runs\tm_diff_mixv2\dice_curve_val.csv    挑 epoch 的依據
runs\tm_diff_mixv2\dice_NNNN.csv         test 結果
runs\tm_diff_mixv2\train_log.txt         訓練 log（含當初的指令）
```

帶回來之後 HD95、SDlogJ、沒參與訓練的結構在筆電補算（`eval_mixv2.py --surface --heldout`），再跟 tm_mixv2、mix_exp3、mix_exp6、架構修改那幾顆逐人比，我來算。
