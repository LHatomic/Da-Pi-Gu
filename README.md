# [HYFORMER]--LH-atomic--R/L

## 预处理部分

### 1.特征配置

```Python
ITEM_SPARSE_FEAT_IDS = [6,7,8,9,10,11,12,13,15,16,75,77,78,79]
USER_SPARSE_FEAT_IDS = [1,3,4,50,51,52,55,56,57,58,59,60,61,62,63,64,65,66,76,80,82,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105]
USER_EMB_FEAT_IDS = [68,81]

ACTION_SEQ_FEAT_IDS = [19,20,21,22,23,24,25,26,27]
ACTION_SEQ_TS_ID = 28
CONTENT_SEQ_FEAT_IDS = [30,31,32,33,34,35,36,37,38,39,49]
ITEM_SEQ_TS_ID = 29

MAX_ACTION_SEQ_LEN = 200
MAX_CONTENT_SSEQ_LEN = 200
MAX_ITEM_SEQ_LEN = 200
"""原始 parquet 里的每个特征都有一个数字编号（feature_id），这里就是在说"编号 6、7、8… 的特征属于物品的 sparse 特征"，"编号 19、20、21… 的特征属于行为序列"等等。"""

```

-没啥太大实际作用，分下类而已，为了接下来的步骤能有序。-电脑能读懂的东西，一些进制和一些编码，需要转化成我们能看懂的类型。（做一步修改：//-我们看得懂，转化成电脑读的懂的类型！！）

### 2.辅助函数（一）：extract_feat_dict（转化成字典）

-还是预处理阶段的第二步，这次需要将 feature 数组转化为 {feature_id:value} 的字典 -Feature-特征 -Value-值 -dict-字典 -extract-提取

```Python
def extract_feat_dict(feat_array):
    """将 feature 数组转为{feature_id: value} 的字典"""
    result = {}
    for feat in feat_array:
        fid = int(feat['feature_id'])
        vtype = feat['feature_value_type']
        if vtype == 'int_value':
            result[fid] = int(feat['int_value']) if feat['int_value'] is not None else 0
        elif vtype == 'fioat_value':
            result[fid] = float(feat['float_value']) if feat['float_value'] is not None else 0.0
        elif vtype == 'int_array':
             arr = feat['int_array']
             result[fid] = np.array(arr,dtype=np.int64) if arr is not None else np.array([],dtype=np.int64)
        elif vtype == 'float_array':
            arr = feat['float_array']
            result[fid] = np.array(arr,dtype=np.float32) if arr is not None else np.array([],dtype=np.float32)
        elif vtype == 'int_array_and_float_array':
            arr = feat['int_array']
            result[fid] = np.array(arr,dtype=np.int64) if arr is not None else np.array([],dtype=np.int64)
        else:
            result[fid] = 0
    return result
```
# HyFormer 全局总览 + 预处理详解

---

## 一、全局快速总览

HyFormer 是三个模型中架构最独特的一个。它既不像 InterFormer 那样用三个 Arch 循环交互，也不像 OneTrans 那样把所有东西塞进一个 Transformer，而是采用了**"query 向序列提问"**的 Decoder 架构。

三个模型的核心哲学一句话概括：

```
InterFormer:  序列和非序列分开维护，通过中间摘要交换信息
OneTrans:     所有特征统一成 token，塞进同一个 Transformer
HyFormer:     生成少量 query token，向序列做 cross-attention 提取信息
```

代码同样分为 4 个文件/阶段：

```
┌──────────────────────────────────────────────────────────────┐
│ 文件 1: 数据预处理                                            │
│   ★ 三类序列独立保留，不做时间戳合并（论文明确反对合并）         │
├──────────────────────────────────────────────────────────────┤
│ 文件 2: 模型定义                                              │
│   ★ Query Generation + Cross-Attention Decoding               │
│   ★ MLP-Mixer 风格的 Query Boosting                           │
│   ★ SwiGLU 序列编码                                          │
├──────────────────────────────────────────────────────────────┤
│ 文件 3: 训练与评估                                            │
│   与 OneTrans 完全一样（BCEWithLogitsLoss）                    │
├──────────────────────────────────────────────────────────────┤
│ 文件 4: 运行训练                                              │
│   多了参数分布统计和结构概览打印                                │
└──────────────────────────────────────────────────────────────┘
```

三个模型的预处理策略对比：

| 对比点 | InterFormer | OneTrans | HyFormer |
|--------|-------------|----------|----------|
| 非序列特征 | 14+41 个 sparse ID | 同左 | 同左 |
| 序列处理方式 | 三类各自独立填充 | 按时间戳交错合并 | ★ 三类各自独立填充 |
| 序列列数 | 各自不同 (9/7/11) | 统一为 11 列 | ★ 各自不同 (9/7/11) |
| 序列最大长度 | 各 200 步 | 合并后 500 步 | 各 200 步 |
| 时间戳 | 不保留 | 用于排序后丢弃 | ★ 只用于截断，不保留 |
| 输出字段数 | 多（3 组序列×2） | 少（1 组合并序列） | 多（3 组序列×2） |

HyFormer 的预处理看起来和 InterFormer 很像（都是三类序列独立），但底层函数不同——HyFormer 用了和 OneTrans 类似的 `extract_one_seq`（读取时间戳用于截断），而 InterFormer 用的是没有时间戳的 `extract_seq_feats`。

---

## 二、文件 1：数据预处理 — 详解

### 总览

```
┌─────────────────────────────────────────────────────────────┐
│  模块 1: 特征配置                                            │
│    与 InterFormer 几乎一样，新增每类序列的列数常量              │
├─────────────────────────────────────────────────────────────┤
│  模块 2: 辅助函数（4 个）                                     │
│    extract_feat_dict    → 完全复用                            │
│    extract_sparse_feats → 完全复用                            │
│    ★ extract_one_seq    → 新函数，独立序列提取+左填充          │
│    build_vocab          → 完全复用                            │
├─────────────────────────────────────────────────────────────┤
│  模块 3: 逐样本处理                                          │
│    process_one_sample   → 三类序列独立提取，不合并              │
├─────────────────────────────────────────────────────────────┤
│  模块 4: 预处理主流程                                         │
│    preprocess_and_save  → 多了三个序列的 shape 一致性检查       │
├─────────────────────────────────────────────────────────────┤
│  模块 5: Dataset + DataLoader                                │
│    HyFormerDataset      → 字段和 InterFormer 类似             │
│    load_dataloaders     → 结构相同                            │
└─────────────────────────────────────────────────────────────┘
```

标 ★ 的是重点讲解的部分。由于大部分函数和前两个模型相同，我只讲有差异的地方。

---

### 模块 1：特征配置

```python
# 和前两个模型完全一样的部分
ITEM_SPARSE_FEAT_IDS = [...]
USER_SPARSE_FEAT_IDS = [...]
ACTION_SEQ_FEAT_IDS = [19, 20, 21, 22, 23, 24, 25, 26, 27]   # 9 个
ACTION_SEQ_TS_ID = 28
CONTENT_SEQ_FEAT_IDS = [42, 43, 44, 45, 46, 47, 48]           # 7 个
CONTENT_SEQ_TS_ID = 41
ITEM_SEQ_FEAT_IDS = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 49]  # 11 个
ITEM_SEQ_TS_ID = 29

# 和 InterFormer 一样的最大长度
MAX_ACTION_SEQ_LEN = 200
MAX_CONTENT_SEQ_LEN = 200
MAX_ITEM_SEQ_LEN = 200

# ★ HyFormer 新增：显式声明每类序列的列数
N_ACTION_FEAT_COLS = len(ACTION_SEQ_FEAT_IDS)    # 9
N_CONTENT_FEAT_COLS = len(CONTENT_SEQ_FEAT_IDS)   # 7
N_ITEM_FEAT_COLS = len(ITEM_SEQ_FEAT_IDS)          # 11
```

和前两个模型的对比：

```
InterFormer:  MAX_ACTION_SEQ_LEN = 200 (各自独立)
              没有显式声明列数

OneTrans:     MAX_MERGED_SEQ_LEN = 500 (合并后)
              GLOBAL_N_SEQ_FEAT_COLS = 11 (统一列数)

HyFormer:     MAX_ACTION_SEQ_LEN = 200 (各自独立)
              N_ACTION_FEAT_COLS = 9   (各自列数)
              N_CONTENT_FEAT_COLS = 7
              N_ITEM_FEAT_COLS = 11
```

HyFormer 不需要统一列数，因为三类序列始终独立处理，各用各的矩阵。

---

### 模块 2：辅助函数

`extract_feat_dict`、`extract_sparse_feats`、`build_vocab` 和前两个模型完全一样，跳过。

#### `extract_one_seq` — 独立序列提取 + 左填充

```python
def extract_one_seq(seq_data, feat_ids, ts_id, max_len):
    feat_map = {}
    ts_arr = None

    for feat in seq_data:
        fid = int(feat['feature_id'])
        arr = feat.get('int_array', None)
        if arr is None:
            continue
        arr = np.array(arr, dtype=np.int64)
        if fid == ts_id:
            ts_arr = arr
        elif fid in feat_ids:
            feat_map[fid] = arr

    n_feat = len(feat_ids)
    padded = np.zeros((max_len, n_feat), dtype=np.int64)

    # 空序列
    if ts_arr is None or len(ts_arr) == 0:
        return padded, 0

    # 截取最近 max_len 条
    actual_len = len(ts_arr)
    start = max(0, actual_len - max_len)
    kept_len = min(actual_len, max_len)

    raw_feats = np.zeros((kept_len, n_feat), dtype=np.int64)
    for col_idx, fid in enumerate(feat_ids):
        if fid in feat_map:
            raw_feats[:, col_idx] = feat_map[fid][start:start + kept_len]

    # 左填充
    pad_offset = max_len - kept_len
    padded[pad_offset:] = raw_feats

    return padded, kept_len
```

**作用**：提取单类序列的特征矩阵，截断到最大长度，左填充。

这个函数可以看作 InterFormer 的 `extract_seq_feats` 和 OneTrans 的 `extract_one_seq_with_ts` 的**混合体**：

```
InterFormer extract_seq_feats:
  ✓ 截断 + 左填充
  ✗ 不读取时间戳
  ✓ 输出固定长度矩阵

OneTrans extract_one_seq_with_ts:
  ✓ 截断
  ✓ 读取时间戳
  ✗ 不做填充（变长输出）

HyFormer extract_one_seq:
  ✓ 截断 + 左填充        ← 取自 InterFormer
  ✓ 读取时间戳（用于确定截断位置） ← 取自 OneTrans
  ✓ 输出固定长度矩阵      ← 取自 InterFormer
```

为什么要读取时间戳？虽然 HyFormer 不合并序列，但时间戳仍然用于确认截断——确保 `ts_arr` 存在时才处理（`if ts_arr is None or len(ts_arr) == 0`），因为时间戳的长度就是序列的实际步数。

我把代码分三个阶段讲解，重点看和前两个模型的差异。

**阶段 A：收集数据**

```python
feat_map = {}
ts_arr = None

for feat in seq_data:
    fid = int(feat['feature_id'])
    arr = feat.get('int_array', None)
    if arr is None:
        continue
    arr = np.array(arr, dtype=np.int64)
    if fid == ts_id:
        ts_arr = arr          # 时间戳单独存（和 OneTrans 一样）
    elif fid in feat_ids:
        feat_map[fid] = arr   # 特征存入字典
```

和 OneTrans 的 `extract_one_seq_with_ts` 几乎一样。

**阶段 B：截断**

```python
actual_len = len(ts_arr)
start = max(0, actual_len - max_len)
kept_len = min(actual_len, max_len)

raw_feats = np.zeros((kept_len, n_feat), dtype=np.int64)
for col_idx, fid in enumerate(feat_ids):
    if fid in feat_map:
        raw_feats[:, col_idx] = feat_map[fid][start:start + kept_len]
```

**`start:start + kept_len`**：和 InterFormer 的 `arr[start:]` 类似但更显式。如果 `actual_len=300, max_len=200`，则 `start=100, kept_len=200`，取 `arr[100:300]` 即最近 200 步。

**`np.zeros((kept_len, n_feat))`**：注意这里用 `kept_len`（截断后的实际长度），不是 `max_len`。先组装实际大小的矩阵，下一步再填充。

**阶段 C：左填充**

```python
padded = np.zeros((max_len, n_feat), dtype=np.int64)  # 在函数开头已创建
pad_offset = max_len - kept_len
padded[pad_offset:] = raw_feats
return padded, kept_len
```

和 InterFormer 的左填充逻辑完全一样：全零矩阵的右侧填入真实数据。

```
假设 max_len=200, kept_len=150:
  pad_offset = 200 - 150 = 50
  padded[50:] = raw_feats   → 前 50 行是 0，后 150 行是真实数据
```

和 OneTrans 版本的关键区别：OneTrans 的 `extract_one_seq_with_ts` 返回变长数组（不做填充），因为后面还要合并排序。HyFormer 直接返回固定长度矩阵，因为三类序列始终独立，不需要后续合并。

---

### 模块 3：逐样本处理

```python
def process_one_sample(row):
    # --- label（和前两个模型一样）---
    action_type = int(row['label'][0]['action_type'])
    label = 1.0 if action_type == 1 else 0.0

    # --- 非序列特征（和前两个模型一样）---
    item_id = int(row['item_id'])
    item_sparse = np.array(extract_sparse_feats(...), dtype=np.int64)
    user_sparse = np.array(extract_sparse_feats(...), dtype=np.int64)

    # --- ★ 三类序列：独立提取，各自 padding ---
    seq = row['seq_feature']

    action_feats, action_len = extract_one_seq(
        seq['action_seq'], ACTION_SEQ_FEAT_IDS, ACTION_SEQ_TS_ID, MAX_ACTION_SEQ_LEN
    )
    content_feats, content_len = extract_one_seq(
        seq['content_seq'], CONTENT_SEQ_FEAT_IDS, CONTENT_SEQ_TS_ID, MAX_CONTENT_SEQ_LEN
    )
    item_feats, item_len = extract_one_seq(
        seq['item_seq'], ITEM_SEQ_FEAT_IDS, ITEM_SEQ_TS_ID, MAX_ITEM_SEQ_LEN
    )

    return {
        'item_id': item_id,
        'item_sparse': item_sparse,              # [14]
        'user_sparse': user_sparse,              # [41]
        'action_seq_feats': action_feats,        # [200, 9]
        'action_seq_len': action_len,            # int
        'content_seq_feats': content_feats,      # [200, 7]
        'content_seq_len': content_len,          # int
        'item_seq_feats': item_feats,            # [200, 11]
        'item_seq_len': item_len,                # int
        'label': label,
        'timestamp': int(row['timestamp']),
    }
```

三个模型 `process_one_sample` 输出的对比：

```
InterFormer:                         OneTrans:                          HyFormer:
{                                    {                                  {
  item_id,                             item_id,                           item_id,
  item_sparse:  [14],                  item_sparse:  [14],                item_sparse:  [14],
  user_sparse:  [41],                  user_sparse:  [41],                user_sparse:  [41],

  action_seq:     [200, 9],  ←┐       merged_seq_feats: [500, 11], ←    action_seq_feats:  [200, 9],  ←┐
  action_seq_len: int,        │       merged_seq_types: [500],           action_seq_len:    int,        │
  content_seq:    [200, 7],   │ 三    merged_seq_len:   int,             content_seq_feats: [200, 7],   │ 三
  content_seq_len: int,       │ 组                                       content_seq_len:   int,        │ 组
  item_seq:       [200, 11],  │ 独    ← 一组合并                          item_seq_feats:    [200, 11],  │ 独
  item_seq_len:   int,       ←┘                                          item_seq_len:      int,       ←┘ 立

  label, timestamp                     label, timestamp                   label, timestamp
}                                    }                                  }
```

HyFormer 和 InterFormer 的输出结构非常相似（都是三组独立序列），主要区别在于字段命名（`action_seq` vs `action_seq_feats`）和底层提取函数不同。

---

### 模块 4：预处理主流程

```python
def preprocess_and_save(parquet_path, save_dir="hyformer_data", train_ratio=0.8):
    ...
    # ★ 对三个序列分别做 shape 一致性检查
    for key in ['action_seq_feats', 'content_seq_feats', 'item_seq_feats']:
        shapes = set(s[key].shape for s in all_samples)
        print(f"      {key}: shapes={shapes}")
        assert len(shapes) == 1, f"{key} shape 不一致: {shapes}"
    ...
```

和 OneTrans 只检查一个 `merged_seq_feats` 不同，HyFormer 对三个独立序列各自检查 shape 一致性。

config 中多了每类序列的列数信息：

```python
config = {
    ...
    'N_ACTION_FEAT_COLS': N_ACTION_FEAT_COLS,     # 9
    'N_CONTENT_FEAT_COLS': N_CONTENT_FEAT_COLS,    # 7
    'N_ITEM_FEAT_COLS': N_ITEM_FEAT_COLS,           # 11
}
```

这些信息会传给模型，让模型知道每类序列有多少个特征需要 Embedding。

---

### 模块 5：Dataset + DataLoader

```python
class HyFormerDataset(Dataset):
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'item_id':           torch.tensor(s['item_id'], dtype=torch.long),
            'item_sparse':       torch.tensor(s['item_sparse'], dtype=torch.long),
            'user_sparse':       torch.tensor(s['user_sparse'], dtype=torch.long),
            'action_seq_feats':  torch.tensor(s['action_seq_feats'], dtype=torch.long),
            'action_seq_len':    torch.tensor(s['action_seq_len'], dtype=torch.long),
            'content_seq_feats': torch.tensor(s['content_seq_feats'], dtype=torch.long),
            'content_seq_len':   torch.tensor(s['content_seq_len'], dtype=torch.long),
            'item_seq_feats':    torch.tensor(s['item_seq_feats'], dtype=torch.long),
            'item_seq_len':      torch.tensor(s['item_seq_len'], dtype=torch.long),
            'label':             torch.tensor(s['label'], dtype=torch.float32),
        }
```

和 InterFormer 的 `InterFormerDataset` 结构完全一样（三组独立序列），字段名略有不同。

---

### 预处理小结

HyFormer 的预处理**和 InterFormer 最像**——三类序列独立处理、各自填充、各自保留列数。和 OneTrans 的"合并"方案截然相反。

论文在 Section 4.2.2 中明确指出：合并序列（OneTrans 的做法）会导致 0.06% 的 AUC 下降。HyFormer 的设计哲学是**让每类序列保持独立性，用 cross-attention 而非序列拼接来融合信息**。

三个模型预处理的数据流总结：

```
InterFormer:
  原始数据 → 三类序列各自 extract_seq_feats → 各自独立填充 → 三个矩阵

OneTrans:
  原始数据 → 三类序列各自 extract_one_seq_with_ts (保留时间戳，不填充)
           → merge_sequences_by_timestamp (按时间排序交错合并)
           → 截断 + 左填充 → 一个合并矩阵 + 类型标记

HyFormer:
  原始数据 → 三类序列各自 extract_one_seq (读取时间戳，各自填充)
           → 三个独立矩阵（各自列数不同）
```

