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


### 核心处理+保存

---一个新的函数：process_one_sample (502)(大大)（比八不）

---把前面所有辅助函数串起来，处理一条原始数据，输出一个干净的字典。这是一个"胶水函数"，把零件组装成成品。\\\

```Python
def process_one_sample(row):
    action_type = int(row['label'][0]['action_type'])
    label = 1.0 if action_type == 1 else 0.0

    item_id = int(row['item_id'])

    item_feat_dict = extract_feat_dict(row['item_feature'])
    item_sparse = extract_sparse_feats(item_feat_dict, ITEM_SPARSE_FEAT_IDS)

    user_feat_dict = extract_feat_dict(row['user_feature'])
    user_sparse = extract_sparse_feats(user_feat_dict, USER_SPARSE_FEAT_IDS)

    seq = row['seq_feature']
    action_seq, action_seq_len = extract_seq_feats(
        seq['action_seq'], ACTION_SEQ_FEAT_IDS, MAX_ACTION_SEQ_LEN)
    content_seq, content_seq_len = extract_seq_feats(
        seq['content_seq'], CONTENT_SEQ_FEAT_IDS, MAX_CONTENT_SEQ_LEN)
    item_seq, item_seq_len = extract_seq_feats(
        seq['item_seq'], ITEM_SEQ_FEAT_IDS, MAX_ITEM_SEQ_LEN)

    timestamp = int(row['timestamp'])

    return {
        'item_id': item_id,
        'item_sparse': np.array(item_sparse, dtype=np.int64),
        'user_sparse': np.array(user_sparse, dtype=np.int64),
        'action_seq': action_seq,
        'action_seq_len': action_seq_len,
        'content_seq': content_seq,
        'content_seq_len': content_seq_len,
        'item_seq': item_seq,
        'item_seq_len': item_seq_len,
        'label': label,
        'timestamp': timestamp,
    }

```

---[还是像背单词一样，看英文知道对应的中文意思是什么就行。
    阅读短文就像阅读代码，我只要知道这篇文章讲的大概是个什么内容，整体先后是按照什么逻辑顺序串连起来的就足够用了。]


---让我们提高速度，浏览一下这个胶水粘起来的过程。

---第二个新函数，preprocess_and_save (读取、构建词表、逐样本处理、排序划分、保存)

```Python
def preprocess_and_save(parquet_path, save_dir="interformer_data", train_ratio=0.8):
    os.makedirs(save_dir, exist_ok=True)

    # 1. 读取
    df = pd.read_parquet(parquet_path)

    # 2. 构建 vocab
    vocab = build_vocab(df)

    # 3. 逐样本处理
    all_samples = []
    for idx in range(len(df)):
        sample = process_one_sample(df.iloc[idx])
        all_samples.append(sample)
        if (idx + 1) % 200 == 0:
            print(f"      已处理 {idx+1}/{len(df)}")

    # 4. 按时间排序 → 划分
    all_samples.sort(key=lambda x: x['timestamp'])
    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # 5. 保存
    torch.save(vocab, os.path.join(save_dir, "vocab.pt"))
    torch.save(train_samples, os.path.join(save_dir, "train_data.pt"))
    torch.save(val_samples, os.path.join(save_dir, "val_data.pt"))
    torch.save(config, os.path.join(save_dir, "config.pt"))
    ...

```

---读取和构建vocab不用多说，从第三步逐样本处理开始：
```Python
def preprocess_and_save(parquet_path, save_dir="interformer_data", train_ratio=0.8):
    os.makedirs(save_dir, exist_ok=True)
"""
3 个入参说明：
【parquet_path】：必填，你的原始数据文件路径。【parquet】是大数据场景常用的列式存储格式，比 【csv】 体积更小、读取速度更快，【pandas】 原生支持。
【save_dir】：选填，处理好的文件保存的文件夹，默认叫【interformer_data】，对应你的模型名称。
【train_ratio=0.8】：选填，训练集占总数据的比例，默认 80% 数据用来训练模型，剩下 20% 用来验证效果，是机器学习的常规划分比例。
【os.makedirs(save_dir, exist_ok=True)】：创建保存用的文件夹。【exist_ok=True】是工程常用写法：如果文件夹已经存在，就不报错、直接复用；不存在就新建，避免重复运行代码时因为文件夹已存在而崩溃。

"""
```

```Python

# 3. 逐样本处理
all_samples = []
for idx in range(len(df)):
    sample = process_one_sample(df.iloc[idx])
    all_samples.append(sample)
    if (idx + 1) % 200 == 0:
        print(f"    已处理 {idx+1}/{len(df)}")
"""
这一步的核心是：把表格里的每一行数据，处理成模型能直接读取的标准格式，统一存起来。
【all_samples = []】：先创建一个空列表，用来装所有处理好的样本。
【for idx in range(len(df))】：循环遍历表格的每一行，idx是行号，从 0 到总行数 - 1。
【sample = process_one_sample(df.iloc[idx])】：
【df.iloc[idx]】：取出表格里第idx行的完整数据；
【process_one_sample】是配套的单样本处理函数，作用是把单行原始数据，转换成模型需要的格式。
【all_samples.append(sample)】：把处理好的单条样本，添加到总列表里。
最后的if判断：进度提示功能。如果数据量很大，循环会跑很久，这段代码每处理 200 条样本，就打印一次当前进度，让你知道程序没有卡死，实时看到处理进度。

"""
```

```Py

# 4. 按时间排序 → 划分
all_samples.sort(key=lambda x: x['timestamp'])
split_idx = int(len(all_samples) * train_ratio)
train_samples = all_samples[:split_idx]
val_samples = all_samples[split_idx:]


```

---这是时序模型 / 推荐系统最核心的规范操作，必须重点理解：
【all_samples.sort(key=lambda x: x['timestamp'])】：把所有样本，按照【timestamp】（时间戳）从小到大排序，也就是按「从早到晚」的时间顺序排列。

```Py
"""
为什么必须按时间划分？
时序推荐模型的目标是「用用户过去的行为，预测未来的行为」，如果随机划分数据集，会把未来的数据放到训练集里，让模型提前 “看到答案”，训练出来的效果是虚假的，上线后完全不能用。按时间划分是时序任务的铁则。
划分逻辑：
【split_idx】：计算划分的分界点，比如总共有 10000 条样本，【train_ratio=0.8】，分界点就是 8000；
【train_samples】：前 80% 的样本（时间更早的），作为训练集，给模型学习规律；
【val_samples】：后 20% 的样本（时间更晚的），作为验证集，用来测试模型在 “未来数据” 上的真实效果，避免过拟合。

"""
```

---读取原始 parquet 数据 → 构建特征词表 → 逐行处理成模型标准格式 → 按时间合规划分训练 / 验证集 → 保存所有文件，供后续训练直接使用

### Dataset + DataLoader

--InterFormerDataset--
```Py
"""---来点小粑粑：
item_id：目标商品 ID，模型要预测用户对这个商品的行为。
item_sparse：商品的离散特征（比如品类、品牌）。
user_sparse：用户的离散特征（比如性别、年龄、城市）。
action_seq：用户历史行为序列（比如用户之前点击过的商品 ID 序列）。
action_seq_len：这个行为序列的真实长度（用来做 padding mask）。
content_seq：用户浏览 / 交互过的内容序列（比如文章、视频 ID）。
content_seq_len：内容序列的真实长度。
item_seq：用户交互过的商品序列（和 action_seq 可能是同一套，也可能是不同的）。
item_seq_len：商品序列的真实长度。
label：标签（比如用户是否点击了目标商品，0 或 1；或者用户的评分）。
"""
```

---这是一个PyTorch 标准的 Dataset 类，专门用来把你上一步预处理好的推荐系统数据，包装成 PyTorch 模型训练能直接吃的格式。只有包装成这种格式，后面的 DataLoader 才能帮你自动做分批次、打乱顺序、并行加载这些训练必备操作。

```Py
"""
PyTorch Dataset 是什么？
先铺垫一个基础概念：
PyTorch 训练模型的标准流程是：
数据文件 → Dataset → DataLoader → 模型训练
Dataset：负责定义 “如何获取一条样本”，是一个抽象接口，你必须继承它并实现固定的几个方法。
DataLoader：在 Dataset 的基础上，帮你自动做分批次（batch）、打乱（shuffle）、多线程加载，是训练的 “数据搬运工”。


"""
```

```Py
class InterFormerDataset(Dataset): #1. 类定义与继承

"""
InterFormerDataset：这是你自定义的数据集类名，对应你的模型 InterFormer，一看就是专门为这个模型定制的。
(Dataset)：表示这个类继承了 PyTorch 自带的 torch.utils.data.Dataset 基类。
继承之后，你的类就拥有了 Dataset 的所有特性，DataLoader 才能识别它。
继承的硬性要求：必须实现 __init__、__len__、__getitem__ 这三个方法。


"""
```

```Py

def __init__(self, samples):
    self.samples = samples  #__init__ 初始化方法


```

```Py
def __len__(self):
    return len(self.samples) #__len__ 方法

"""这个方法告诉 PyTorch：你的数据集总共有多少条样本。
比如你的 train_data.pt 里有 8000 条样本，调用 len(dataset) 就会返回 8000。
DataLoader 分批次的时候，会用这个值来计算一个 epoch 有多少个 batch。"""
```

核心来喽！！！！！小飞棍来喽！！！！乐迪起飞咯！！！！
```Py
def __getitem__(self, idx):
    s = self.samples[idx]
    return {
        'item_id':           torch.tensor(s['item_id'], dtype=torch.long),
        'item_sparse':       torch.tensor(s['item_sparse'], dtype=torch.long),
        'user_sparse':       torch.tensor(s['user_sparse'], dtype=torch.long),
        'action_seq':        torch.tensor(s['action_seq'], dtype=torch.long),
        'action_seq_len':    torch.tensor(s['action_seq_len'], dtype=torch.long),
        'content_seq':       torch.tensor(s['content_seq'], dtype=torch.long),
        'content_seq_len':   torch.tensor(s['content_seq_len'], dtype=torch.long),
        'item_seq':          torch.tensor(s['item_seq'], dtype=torch.long),
        'item_seq_len':      torch.tensor(s['item_seq_len'], dtype=torch.long),
        'label':             torch.tensor(s['label'], dtype=torch.float32),
    }


```

---这个方法是 【Dataset】 的核心：当你用 【dataset[idx]】 取第 【idx】 条样本时，它会返回模型需要的、格式正确的数据。


---假设你已经有了预处理好的训练数据，完整使用流程是这样的：
```Py
# 1. 加载预处理好的数据
train_samples = torch.load("interformer_data/train_data.pt")
val_samples = torch.load("interformer_data/val_data.pt")

# 2. 创建 Dataset 对象
train_dataset = InterFormerDataset(train_samples)
val_dataset = InterFormerDataset(val_samples)

# 3. 用 DataLoader 包装 Dataset
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# 4. 训练时直接迭代 DataLoader
for batch in train_loader:
    # batch 就是一个字典，每个值都是一个 batch_size 维度的张量
    item_id = batch['item_id']
    label = batch['label']
    # 喂给模型计算损失...

```

---load_dataloaders--
作用：从 .pt 文件加载数据，创建 DataLoader。

```Py
def load_dataloaders(save_dir="interformer_data", batch_size=32):
    vocab = torch.load(os.path.join(save_dir, "vocab.pt"), weights_only=False)
    train_samples = torch.load(os.path.join(save_dir, "train_data.pt"), weights_only=False)
    val_samples = torch.load(os.path.join(save_dir, "val_data.pt"), weights_only=False)
    config = torch.load(os.path.join(save_dir, "config.pt"), weights_only=False)

    train_loader = DataLoader(
        InterFormerDataset(train_samples),
        batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        InterFormerDataset(val_samples),
        batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    )
    return train_loader, val_loader, vocab, config

```

--预处理到此--

从嵌套的 parquet 结构中，把每条样本的特征提取为固定形状的 numpy 数组，保存为 .pt 文件，供模型训练时直接加载。



## 模型定义部分（核心）

### 先说个小知识（烟头叔叔）

---PyTorch 中的 nn.Module

```Py
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()                    # 必须调用父类构造函数
        self.linear = nn.Linear(10, 5)        # 定义一个零件

    def forward(self, x):                     # 定义数据流
        return self.linear(x)                 # x 流过这个零件

```

---super().__init__()：调用父类 nn.Module 的构造函数，完成内部初始化（注册参数、子模块等）。每个 nn.Module 子类的 __init__ 第一行都必须写这个，否则会报错。

---调用模型时直接写 model(x) 而不是 model.forward(x)。PyTorch 重载了 __call__ 方法，model(x) 会自动调用 forward(x) 并附加一些内部逻辑（如 hooks）。

### 言归正传

InterFormer的模型定义部分一共是由四个模块组成，每个模块各成一片天地却又彼此构成联系；精妙的架构设计，下面让我们好好看看各个模块之间到底是如何设计的。

### 模块1：基础组件

---SelfGating ——(压力板)【用来控制门的】|
```Py
class SelfGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)
```

--作用：对输入向量的每个维度，学习一个 0~1 之间的"开关"，决定保留多少信息。
---这里，有些新思想（冷启动）；
---首先，我们先得了解一下什么叫做冷启动，冷启动，就是因目标对象（用户 / 广告 / 系统）缺乏足量、有效的历史交互数据，导致模型无法精准学习其特征与规律，进而出现预估偏差大、推荐效果差、无法精准匹配的核心问题，是广告推荐领域公认的核心学术与工业痛点。

【下面，广告冷启动分为三种细化：】
1. 广告 / 物品冷启动（广告场景最核心、最普遍的类型）
定义：新上线的广告创意、商品、投放计划，没有历史曝光、点击、转化数据，或只有极少量样本，模型无法学习该广告的转化特征、受众偏好，无法精准推给高转化潜力的用户。
广告场景实例：广告主刚上传一条新的游戏广告素材，还没有任何用户点击过，模型无法判断这条素材适合推给 18-25 岁男性还是 30-40 岁用户，只能泛流量投放，转化效果远低于预期，这就是典型的广告冷启动。
细分场景：素材冷启动、商品冷启动、投放计划冷启动、新广告主冷启动（首次投放的广告主，无历史投放数据）。
2. 用户冷启动
定义：平台的新注册用户，没有任何历史广告交互行为（点击、转化、加购、停留等），或只有极少量行为，模型无法捕捉用户的个性化偏好，无法给用户推送符合其需求的广告。
实例：用户刚下载一个 APP、完成注册，第一次打开应用，没有任何点击记录，模型只能基于用户的非序列静态特征（注册时填的年龄、性别、城市、设备信息）做泛化推荐，无法实现个性化精准匹配，这就是用户冷启动。
3. 系统冷启动（最极端的类型）
定义：一个全新的广告平台 / 推荐系统刚上线，既没有用户历史数据，也没有广告历史数据，甚至连基础的用户画像、类目体系都不完善，整个系统从零开始，完全没有可用于模型训练的历史数据。
实例：一个新上线的短视频 APP，搭建了自有的广告投放系统，没有任何历史用户与广告数据，整个推荐体系从零搭建，这就是系统冷启动。
