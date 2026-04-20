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
-np.array(arr, dtype=np.int64)：把一个普通 Python 列表转成 NumPy 数组，并指定数据类型为 64 位整数。NumPy 数组比 Python 列表快得多，是数值计算的标准工具。

### 辅助函数（二）：extract_sparse_feats（按顺序取值组成列表）

-在上个步骤建成的字典里，按照指定顺序取出值组成列表。
-如果某个指定的ID顺序不存在，就默认为0.
-feat_dict.get(fid, default)：字典的 get 方法。

```Python
[int(feat_dict.get(fid, default)) for fid in feat_ids]
"""列表推导式"""
```
【相互等价】
```Python
result = []
for fid in feat_ids:
    result.append(int(feat_dict.get(fid, default)))
return result
```

-关键在于：【----for---变量---in---可迭代对象】。

### 辅助函数（三）：extract_seq_feats（-按特征维度储存转化成按时间步*特征维度的矩阵-做截断-左填充）【复杂】

-关于预处理的部分，我们其实只需要理解其中的逻辑就可以，大致知道预处理的整体架构在干些什么，其中对于各类函数的使用属于Python课程，重点看有关于数学思想建模的地方。就像鸡蛋壳里面的蛋清，蛋清里面还有蛋黄，论营养我们还得着重看蛋黄部分。

-那么，我们大致浏览一遍这个最复杂的辅助函数的三段功能都是怎么构造实现的。

```Python
"""收集原始数据"""
feat_map = {}
for feat in seq_data:
    fid = int(feat['feature_id'])
    if fid in feat_ids:                        # 只要我们关心的特征
        arr = feat.get('int_array', None)
        if arr is not None:
            feat_map[fid] = np.array(arr, dtype=np.int64)
        else:
            feat_map[fid] = np.array([], dtype=np.int64)
```

-其实显而易见，无论是输入还是输出结果，核心还是在于[for---in---]遍历。遍历原始序列数据，把需要的特征 ID 对应的数组收集到 [at_map]字典中。

```Python
"""处理空数据的边界情况"""
if len(feat_map) == 0:
    return np.zeros((max_len, len(feat_ids)), dtype=np.int64), 0
```

-如果做项目时候，不考虑空数据的处理问题，会boom。因此要考虑全面。在这里是：如果该样本没有任何序列数据，直接返回一个全零矩阵和长度 0。|||
\\\np.zeros((max_len, len(feat_ids)), dtype=np.int64)：创建一个形状为 (max_len, 特征数) 的全零矩阵
\\\-有一点说明：（ x , y ）:x{矩阵的行数} ；  y{矩阵的列数}。

```Python
""" 截断+左填充，组装矩阵 """
actual_len = len(next(iter(feat_map.values())))
start = max(0, actual_len - max_len)
result = np.zeros((max_len, len(feat_ids)), dtype=np.int64)

for col_idx, fid in enumerate(feat_ids):
    arr = feat_map.get(fid, np.zeros(actual_len, dtype=np.int64))
    truncated = arr[start:]
    pad_len = max_len - len(truncated)
    if pad_len > 0:
        result[pad_len:, col_idx] = truncated
    else:
        result[:, col_idx] = truncated

seq_len = min(actual_len, max_len)
return result, seq_len
```

--[feat_map.value()]返回字典所有值的视图。
--[iter()]把他变成迭代器。
--[next()]取迭代器的第一个元素。
--TT:取字典中任意一个值。
--T:获取序列真实长度。
--[enumerate(feat_ids)]遍历列表时同时获取下标和值。

### 词表构建

```Python
def build_vocab(df):
    vocab = {}

    def update_vocab(fid, val):
        if isinstance(val, (int, float, np.integer, np.floating)):
            val = int(val)
            vocab[fid] = max(vocab.get(fid, 0), val + 1)
        elif isinstance(val, np.ndarray) and val.size > 0:
            vocab[fid] = max(vocab.get(fid, 0), int(val.max()) + 1)

    for idx in range(len(df)):
        item_feat = extract_feat_dict(df.iloc[idx]['item_feature'])
        for fid, val in item_feat.items():
            update_vocab(fid, val)
        user_feat = extract_feat_dict(df.iloc[idx]['user_feature'])
        for fid, val in user_feat.items():
            update_vocab(fid, val)
        seq = df.iloc[idx]['seq_feature']
        for seq_key in ['action_seq', 'content_seq', 'item_seq']:
            for feat in seq[seq_key]:
                fid = int(feat['feature_id'])
                arr = feat.get('int_array', None)
                if arr is not None:
                    arr = np.array(arr, dtype=np.int64)
                    update_vocab(fid, arr)

    vocab['item_id'] = int(df['item_id'].max()) + 1
    return vocab
```

---作用：遍历整个数据集，找出每个 feature_id 对应的最大值，+1 后作为 Embedding 层的词表大小。\\\
---这里+1是必要的，为了避免出现大于规定范围的值长度出现，因此+1可以很好的规避正好取到最大值的情况，防止发生错误。


---这里有一个嵌套函数：
```Python
def build_vocab(df):
    vocab = {}

    def update_vocab(fid, val):
        vocab[fid] = max(vocab.get(fid, 0), val + 1)
    ...

"""
---两个函数的关系：
【build_vocab】 是外层函数，它里面定义了 【vocab = {}】 这个变量。
【update_vocab】 是内层函数，它直接访问并修改了外层的 【vocab】。

【vocab.get(fid, 0)】：如果 【vocab】 里有 【fid】 这个键，就返回它的值；没有的话，返回 0。
【val + 1】：把传进来的 【val】 加 1。
【max(a, b)】：取两个数里的最大值，再赋值给 【vocab[fid]】。
所以这段代码的作用是：给每个 fid 存一个不断更新的最大值。

"""
```

```Python
"""
【什么是闭包？】
---闭包的核心定义：
---内层函数（update_vocab）引用了外层函数（build_vocab）的变量（vocab），并且内层函数可以在外部被调用时，依然记住并访问外层函数的变量。
1. 用生活场景类比
你可以把 【build_vocab】 想象成一个带保险箱的房间：
房间里有个保险箱 【vocab = {}】，存着重要数据。
【update_vocab】 是房间里的专属管理员，它知道保险箱的密码，能直接往里存东西、改东西。
管理员不用每次开保险箱都让别人把密码（也就是 【vocab】）传给它，它天生就知道。
2. 没有闭包会怎么样？
如果不用闭包，你得把 vocab 当成参数传来传去，代码会变得很啰嗦：
"""
# 不用闭包的写法，很麻烦
def update_vocab(vocab, fid, val):
    vocab[fid] = max(vocab.get(fid, 0), val + 1)

def build_vocab(df):
    vocab = {}
    # 每次调用 update_vocab 都要把 vocab 传进去
    update_vocab(vocab, "apple", 5)
    update_vocab(vocab, "apple", 7)
    # ...
"""
---闭包的好处就是：内层函数自动 “记住” 了外层的变量，不用你手动传来传去，代码更干净。

"""
```

---言说回正题，好处是什么？
1.不用传参：【update_vocab】 不用每次都把 【vocab】 当参数传进去，代码更简洁。
2.数据隔离：【vocab】 是 【build_vocab】 里的局部变量，外面不能直接改，只能通过 【update_vocab 来修改，更安全。
3.封装性好：把 “创建字典” 和 “更新字典” 的逻辑放在一起，结构更清晰。

















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

