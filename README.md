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
