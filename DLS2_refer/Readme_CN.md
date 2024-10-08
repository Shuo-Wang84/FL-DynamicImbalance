# 一、实验设置

## 1、人工数据集

### 1.1、生成数据集源数据

```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

base_path = 'E:/FedStream/data_set_syn/Synthetic0310/'
dataset_path = 'Make_CF/Syn130K_10F_2C/'
path = os.path.join(base_path,dataset_path)
if not os.path.exists(path):
    os.makedirs(path)

source_path = os.path.join(path,'source.csv')
df = pd.read_csv(source_path , comment='@', header=None)
print(df.head())
```

### 1.2、将元数据组成`Balanced`

10个客户端，每个客户端都分配100个数据块，每个数据块都是平衡的。

```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

base_path = 'E:/FedStream/data_set_syn/Synthetic0310/'
source_dataset_path = 'Make_CF/Syn130K_10F_2C/'
path = os.path.join(base_path,source_dataset_path)
if not os.path.exists(path):
    os.makedirs(path)

source_path = os.path.join(path,'source.csv')
df = pd.read_csv(source_path)

# 分离特征和目标变量
X = df.iloc[:, :-1]  # 特征
y = df.iloc[:, -1]   # 目标变量

# 标准化特征数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 保存训练集和测试集
train_dir = f'{path}/train'
test_dir = f'{path}/test'
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

train_df = pd.DataFrame(X_train,columns=X.columns)
train_df['target'] = y_train.reset_index(drop=True).astype(int)
train_df.to_csv(os.path.join(train_dir, 'train.csv'), index=False)

# 将测试集的特征数据和目标变量合并到同一个 DataFrame
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['target'] = y_test.reset_index(drop=True).astype(int) # very important
test_df.to_csv(os.path.join(test_dir, 'test.csv'), index=False)

# 分割训练集并重新组合
class_dir = f'{path}/130K_binary_class'
if not os.path.exists(class_dir):
    os.makedirs(class_dir)

for c in range(2):  # 假设有两个类别，可以根据实际情况修改
    class_df = train_df[train_df['target'] == c]
    class_df.to_csv(os.path.join(class_dir, f'class_{c}.csv'), index=False)

# Severity [98:2]VH(very high),[95:5]H(High),[90:10]M(middle),[75:25]L(Low),Balanced(B)--(5)
# Coverage one_L,half_H,all_A -(3)
# static(S)/Dynamic(D)
# experiment_1: Severity(static) impact federated-system
"""
    Coverage : ALL ,Severity VH , H ,M ,L(static 4)
    想要说明，在所有客户端都是同样严重程度下，对模型性能的影响
"""
# experiment_2: CoverType(static) impact federated-system
""" Coverage : Half ,Severity VH , H (static 2)"""
""" Coverage : One ,Severity VH , H (static 2)"""
""" 想要说明，同样的严重程度，全部，半数，一个对全局的影响"""
# experiment_3:Dynamic - Frequency  - abrupt/incremental(2*2)
""" [50:50] <-> [95:5] at(every per 10 round)"""
""" [50:50] <-> [95:5] at(every per 20 round)"""
# Severity
""" [90:10] <-> [98:2] at(every per 10 round)"""
""" [98:2] <-> [98:2] at(every per 20 round)"""
# experiment_3:Dynamic - Recurrence
# experiment_4:Dynamic - incremental - direction

setting = 'Balanced'
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratio = [50,50]
per_round_samples = sum(class_ratio)

rounds = int(len(X_train)/clients/sum(class_ratio))  # 轮次数量
for r in range(rounds):
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
```

### 1.3、讨论严重性`Severity`

#### 1.3.1 所有客户端都是同样严重性的情况

（1）`COvA_SeverityVH`，实验设置 10个客户端，所有客户端都有100个数据块，每个数据块的’class 0 : clsss 1"类别比例是98：2

```python
setting = 'CovA_SeverityVH'
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratio = [98:2]
per_round_samples = sum(class_ratio)

rounds = int(len(X_train)/clients/sum(class_ratio))  # 轮次数量
for r in range(rounds):
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
```

（2）`COvA_SeverityH`，实验设置 10个客户端，所有客户端都有100个数据块，每个数据块’class 0 : clsss 1"的类别比例是95：5
```python
setting = 'CovA_SeverityH'
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratio = [95,5]
per_round_samples = sum(class_ratio)

rounds = int(len(X_train)/clients/sum(class_ratio))  # 轮次数量
for r in range(rounds):
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
```

（3）`COvA_SeverityM`，实验设置 10个客户端，所有客户端都有100个数据块，每个数据块’class 0 : clsss 1"的类别比例是90：10
```python
setting = 'CovA_SeverityM'
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratio = [90,10]
per_round_samples = sum(class_ratio)

rounds = int(len(X_train)/clients/sum(class_ratio))  # 轮次数量
for r in range(rounds):
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
```

（4）`COvA_SeverityL`，实验设置 10个客户端，所有客户端都有100个数据块，每个数据块’class 0 : clsss 1"的类别比例是75：25
```python
setting = 'CovA_SeverityL'
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratio = [75,25]
per_round_samples = sum(class_ratio)

rounds = int(len(X_train)/clients/sum(class_ratio))  # 轮次数量
for r in range(rounds):
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
```

（5）`COvA_SeverityVHR`，实验设置 10个客户端，所有客户端都有100个数据块，每个数据块’class 0 : clsss 1"的类别比例是2:98,这个的多数类与少数类之比刚好与`covA_SeverityVH`相反
```python
setting = 'CovA_SeverityVH'
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratio = [2:98]
per_round_samples = sum(class_ratio)

rounds = int(len(X_train)/clients/sum(class_ratio))  # 轮次数量
for r in range(rounds):
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
```

### 1.4、速度`Speed`

1.`Abrupt/Sudden`,突然型变化，突然从一种不平衡（也许是平衡状态）状态变成另外一种

```python
setting = 'CovA_Abrupt_BtoVH'
# 9个稍微不平衡，一个平衡
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratios = [[50, 50]] * 10
per_round_samples = sum(class_ratios[1])

# rounds = int(len(X_train)/clients/sum(class_ratios[1]))  # 轮次数量
for r in range(100):
    print("round:",r)
    if r >= 20:
        class_ratios = [[98,2]] * 10
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        class_ratio = class_ratios[c]
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
# 目的，为了验证，一半客户端平衡数据对稍微不平衡数据集的影响
```

2.`Gradual`,两种不平衡状态交替出现

```python
# 2024 05 20 18:18 Gradual
import os
import pandas as pd
base_path = 'E:/FedStream/data_set_syn/Synthetic0310/'
source_dataset_path = 'Make_CF/Syn130K_10F_2C/'
path = os.path.join(base_path,source_dataset_path)
class_dir = f'{path}/130K_binary_class'
if not os.path.exists(path):
    os.makedirs(path)
setting = 'Gradual2'
# 9个稍微不平衡，一个平衡
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratios = [[50, 50]] * 10
per_round_samples = sum(class_ratios[1])

# rounds = int(len(X_train)/clients/sum(class_ratios[1]))  # 轮次数量
for r in range(100):
    # [0-15]B,[15-20]VH,[20-30]M,[30-35]VH,[35-60]B,[60-65]VH,[65-75]M,[75-80]VH,[80-100]B
    if r<=19:
        class_ratios = [[50, 50]] * 10
    elif r>19 and r<=45:
        if r in [20,26,32,37,41,44]:
            class_ratios = [[98,2]] * 10
        else:
            class_ratios = [[50, 50]] * 10
    else:
        if r in [48,52,57,63,70]:
            class_ratios = [[50, 50]] * 10
        else:
            class_ratios = [[98,2]] * 10
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        class_ratio = class_ratios[c]
        # if c ==0:
        print(f"round {r}: client :{c},class_ratio :{class_ratio}")
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)

```

3.`Incremental`,从一种不平衡状态变成了另外一种，中间不平衡程度呈现过渡式变化

```python
# 增量型变化 step 10,20,50 direction
setting = 'covA_Incremental_10_dirF'
# [50,50]->[95,5] ,happend 1 times,direction forward/backward
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)
clients = 10  # 客户端数量

class_ratios = [[50,50]]*10 # incremnetal 45/10 ,45/20 ,45/50 (4.5)
per_round_samples = sum(class_ratios[1])
# 48/10
# 0   1   2     3   4    5    6    7    8    9   10
# 50 54.5 59  63.5  67  71.5  76  80.5  85  89.5  95
#
# 50 55  60  65  70  75  80  85  90  95  100
# rounds = int(len(X_train)/clients/sum(class_ratios[1]))  # 轮次数量
for r in range(100):
    # if r>=20 :
    #     class_ratio = [95,5]
    for c in range(clients):
        if  (r >= 20 and r < 31) :
            class_ratios[c] = [int(50+4.8*(r-20)) ,int(50-4.8*(r-20))]
        elif r>=31 :
            class_ratios[c] = [98,2]
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        class_ratio = class_ratios[c]
        print(f"round {r}: client :{c},class_ratio :{class_ratio}")
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
```

### 1.5、频率`Frequency`

同样一个变化会周期性相同的发生；类似于Balance-->imbalanceA-->Balance-->imbalanceA

```python
setting = 'Rc1c2_Frequency_10VH'
# 9个稍微不平衡，一个平衡
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
# class_ratios = [[50, 50]] * 9 + [[95, 5]] * 1
# [50,50] ->[95,5]->[50,50] # forward per 5,10,20 round
# [95,5] ->[50,50] ->[95,95] # backward per 5,10,20 round
per_round_samples = sum(class_ratio)

# rounds = int(len(X_train)/clients/sum(class_ratio))  # 轮次数量
for r in range(100):
    if int(r/10)%2 == 0 :
        class_ratio = [98,2]
    elif int(r/10)%2 == 1 :
        class_ratio = [2,98]
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        # class_ratio = class_ratios[c]
        # print(f" round {r} , client {c} : class_ratio: {class_ratio}")
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
# 目的，为了验证，一半客户端平衡数据对稍微不平衡数据集的影响
```

```python
setting = 'Abrupt_Frequency10_DirF'
# 9个稍微不平衡，一个平衡
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
# class_ratios = [[50, 50]] * 9 + [[95, 5]] * 1
# [50,50] ->[95,5]->[50,50] # forward per 5,10,20 round
# [95,5] ->[50,50] ->[95,95] # backward per 5,10,20 round
per_round_samples = sum(class_ratio)

rounds = int(len(X_train)/clients/sum(class_ratio))  # 轮次数量
for r in range(rounds):
    if int(r/10)%2 == 0 :
        class_ratio = [50,50]
    elif int(r/10)%2 == 1 :
        class_ratio = [95,5]
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        # class_ratio = class_ratios[c]
        # print(f" round {r} , client {c} : class_ratio: {class_ratio}")
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
# 目的，为了验证，一半客户端平衡数据对稍微不平衡数据集的影响
```

```python
setting = 'Abrupt_Frequency20_DirF'
# 9个稍微不平衡，一个平衡
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
# class_ratios = [[50, 50]] * 9 + [[95, 5]] * 1
# [50,50] ->[95,5]->[50,50] # forward per 5,10,20 round
# [95,5] ->[50,50] ->[95,95] # backward per 5,10,20 round
per_round_samples = sum(class_ratio)

rounds = int(len(X_train)/clients/sum(class_ratio))  # 轮次数量
for r in range(rounds):
    if int(r/20)%2 == 0 :
        class_ratio = [50,50]
    elif int(r/20)%2 == 1 :
        class_ratio = [95,5]
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        # class_ratio = class_ratios[c]
        # print(f" round {r} , client {c} : class_ratio: {class_ratio}")
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
# 目的，为了验证，一半客户端平衡数据对稍微不平衡数据集的影响
```

### 1.6、重复性`Recurrence`

重复性和频率最大的区别是，重复性类似于季节变化这种，多个变化在一个时间段内成组出现，两次出现的间隔不确定

```python
# 本次实验主要生成重复性
# [0-15]B,[15-20]VH,[20-30]M,[30-35]VH,[35-60]B,[60-65]VH,[65-75]M,[75-80]VH,[80-100]B
# 可预测性 生成不可预测性
# [0-10]B,[10-15]VH,[15-20]M,[20-50]B,[50-55]M,[55-85]B,[85-90]VH,[90-100]B
# 方向性(保持全局平衡，全局不平衡的)
# covH (half VH->VHR,VHR->VH)全局的角度来看没有大小类之分（98/2 ->2/98,2/98 ->98/2）(100,100)
# covH (half B->M,M->B)全局角度来看数据是第一类为大类 (50/50 ->98/2,98/2->50/50)(140/60)
# 相关性
# 每个客户端随机选择[B,VH,VHR,H,M,L]
# recurrence
import os
import pandas as pd
base_path = 'E:/FedStream/data_set_syn/Synthetic0310/'
source_dataset_path = 'Make_CF/Syn130K_10F_2C/'
path = os.path.join(base_path,source_dataset_path)
class_dir = f'{path}/130K_binary_class'
if not os.path.exists(path):
    os.makedirs(path)
setting = 'Recurrence'
# 9个稍微不平衡，一个平衡
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratios = [[50, 50]] * 10
per_round_samples = sum(class_ratios[1])

# rounds = int(len(X_train)/clients/sum(class_ratios[1]))  # 轮次数量
for r in range(100):
    # [0-15]B,[15-20]VH,[20-30]M,[30-35]VH,[35-60]B,[60-65]VH,[65-75]M,[75-80]VH,[80-100]B
    if r >= 15 and r<20: # 15 16 17 18 19
        class_ratios = [[98,2]] * 10
    elif r>=20 and r<30:
        class_ratios = [[90,10]] * 10
    elif r>=30 and r<35:
        class_ratios = [[98,2]] * 10
    elif r>=35 and r<60:
        class_ratios = [[50, 50]] * 10
    if r >= 65 and r<70: # 15 16 17 18 19
        class_ratios = [[98,2]] * 10
    elif r>=70 and r<80:
        class_ratios = [[90,10]] * 10
    elif r>=80 and r<85:
        class_ratios = [[98,2]] * 10
    elif r>=85 and r<100:
        class_ratios = [[50, 50]] * 10
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        class_ratio = class_ratios[c]
        # if c ==0:
        print(f"round {r}: client :{c},class_ratio :{class_ratio}")
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)

```

1.7、预测性`Predictablity`

```python
# 可预测性 生成不可预测性
# [0-10]B,[10-15]VH,[15-20]M,[20-50]B,[50-55]M,[55-85]B,[85-90]VH,[90-100]B
# 方向性(保持全局平衡，全局不平衡的)
# covH (half VH->VHR,VHR->VH)全局的角度来看没有大小类之分（98/2 ->2/98,2/98 ->98/2）(100,100)
# covH (half B->M,M->B)全局角度来看数据是第一类为大类 (50/50 ->98/2,98/2->50/50)(140/60)
# 相关性
# 每个客户端随机选择[B,VH,VHR,H,M,L]
# recurrence
import os
import pandas as pd
base_path = 'E:/FedStream/data_set_syn/Synthetic0310/'
source_dataset_path = 'Make_CF/Syn130K_10F_2C/'
path = os.path.join(base_path,source_dataset_path)
class_dir = f'{path}/130K_binary_class'
if not os.path.exists(path):
    os.makedirs(path)
setting = 'unPredictability'
# 9个稍微不平衡，一个平衡
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratios = [[50, 50]] * 10
per_round_samples = sum(class_ratios[1])

# rounds = int(len(X_train)/clients/sum(class_ratios[1]))  # 轮次数量
for r in range(100):
    # [0-10]B,[10-15]VH,[15-20]M,[20-50]B,[50-60]VH,[60-80]B,[80-85]H,[85-90]VH,[90-100]B
    if r >= 10 and r<15: # 15 16 17 18 19
        class_ratios = [[98,2]] * 10
    elif r>=15 and r<20:
        class_ratios = [[90,10]] * 10
    elif r>=20 and r<50:
        class_ratios = [[50, 50]] * 10
    if r >= 50 and r<60: # 15 16 17 18 19
        class_ratios = [[98,2]] * 10
    elif r>=60 and r<80:
        class_ratios = [[50,50]] * 10
    elif r>=80 and r<85:
        class_ratios = [[95,5]] * 10
    elif r>=85 and r<90:
        class_ratios = [[98, 2]] * 10
    elif r>=90 and r<100:
        class_ratios = [[50, 50]] * 10
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        class_ratio = class_ratios[c]
        # if c ==0:
        print(f"round {r}: client :{c},class_ratio :{class_ratio}")
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)

```

### 1.7 覆盖性`Coverage`

在此我们单讨论客户端是平衡和不平衡的情况，为了方便观察覆盖性对全局的影响，我们让客户端在整个训练周期的不平衡率都一样，在只有10个客户端的联邦系统中讨论所有客户端都是平衡状态，9个是平衡一个不平衡，一半平衡另一半不平衡，一个平衡9个不平衡，都不平衡。

`Balanced`:

```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

base_path = 'E:/FedStream/data_set_syn/Synthetic0310/'
source_dataset_path = 'Make_CF/Syn130K_10F_2C/'
path = os.path.join(base_path,source_dataset_path)
if not os.path.exists(path):
    os.makedirs(path)

source_path = os.path.join(path,'source.csv')
df = pd.read_csv(source_path)

# 分离特征和目标变量
X = df.iloc[:, :-1]  # 特征
y = df.iloc[:, -1]   # 目标变量

# 标准化特征数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 保存训练集和测试集
train_dir = f'{path}/train'
test_dir = f'{path}/test'
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

train_df = pd.DataFrame(X_train,columns=X.columns)
train_df['target'] = y_train.reset_index(drop=True).astype(int)
train_df.to_csv(os.path.join(train_dir, 'train.csv'), index=False)

# 将测试集的特征数据和目标变量合并到同一个 DataFrame
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['target'] = y_test.reset_index(drop=True).astype(int) # very important
test_df.to_csv(os.path.join(test_dir, 'test.csv'), index=False)

# 分割训练集并重新组合
class_dir = f'{path}/130K_binary_class'
if not os.path.exists(class_dir):
    os.makedirs(class_dir)

for c in range(2):  # 假设有两个类别，可以根据实际情况修改
    class_df = train_df[train_df['target'] == c]
    class_df.to_csv(os.path.join(class_dir, f'class_{c}.csv'), index=False)

# Severity [98:2]VH(very high),[95:5]H(High),[90:10]M(middle),[75:25]L(Low),Balanced(B)--(5)
# Coverage one_L,half_H,all_A -(3)
# static(S)/Dynamic(D)
# experiment_1: Severity(static) impact federated-system
"""
    Coverage : ALL ,Severity VH , H ,M ,L(static 4)
    想要说明，在所有客户端都是同样严重程度下，对模型性能的影响
"""
# experiment_2: CoverType(static) impact federated-system
""" Coverage : Half ,Severity VH , H (static 2)"""
""" Coverage : One ,Severity VH , H (static 2)"""
""" 想要说明，同样的严重程度，全部，半数，一个对全局的影响"""
# experiment_3:Dynamic - Frequency  - abrupt/incremental(2*2)
""" [50:50] <-> [95:5] at(every per 10 round)"""
""" [50:50] <-> [95:5] at(every per 20 round)"""
# Severity
""" [90:10] <-> [98:2] at(every per 10 round)"""
""" [98:2] <-> [98:2] at(every per 20 round)"""
# experiment_3:Dynamic - Recurrence
# experiment_4:Dynamic - incremental - direction

setting = 'Balanced'
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratio = [50,50]
per_round_samples = sum(class_ratio)

rounds = int(len(X_train)/clients/sum(class_ratio))  # 轮次数量
for r in range(rounds):
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
```

`Cov9B_SeverityVH`
```python
import os
import pandas as pd
base_path = 'E:/FedStream/data_set_syn/Synthetic0310/'
source_dataset_path = 'Make_CF/Syn130K_10F_2C/'
class_dir = f'{path}/130K_binary_class'
path = os.path.join(base_path,source_dataset_path)
if not os.path.exists(path):
    os.makedirs(path)
setting = 'Cov9B_SeverityVH'
# 9个稍微不平衡，一个平衡
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratios = [[50, 50]] * 9 + [[98, 2]] * 1
per_round_samples = sum(class_ratios[1])

# rounds = int(len(X_train)/clients/sum(class_ratios[1]))  # 轮次数量
for r in range(100):
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        class_ratio = class_ratios[c]
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
# 目的，为了验证，一半客户端平衡数据对稍微不平衡数据集的影响
```
`CovH_SeverityVH`
```python
import os
import pandas as pd
base_path = 'E:/FedStream/data_set_syn/Synthetic0310/'
source_dataset_path = 'Make_CF/Syn130K_10F_2C/'
class_dir = f'{path}/130K_binary_class'
path = os.path.join(base_path,source_dataset_path)
if not os.path.exists(path):
    os.makedirs(path)
setting = 'Cov9B_SeverityVH'
# 9个稍微不平衡，一个平衡
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratios = [[50, 50]] * 5 + [[98, 2]] * 5
per_round_samples = sum(class_ratios[1])

# rounds = int(len(X_train)/clients/sum(class_ratios[1]))  # 轮次数量
for r in range(100):
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        class_ratio = class_ratios[c]
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
# 目的，为了验证，一半客户端平衡数据对稍微不平衡数据集的影响
```
`Cov1B_SeverityVH`
```python
import os
import pandas as pd
base_path = 'E:/FedStream/data_set_syn/Synthetic0310/'
source_dataset_path = 'Make_CF/Syn130K_10F_2C/'
class_dir = f'{path}/130K_binary_class'
path = os.path.join(base_path,source_dataset_path)
if not os.path.exists(path):
    os.makedirs(path)
setting = 'Cov9B_SeverityVH'
# 9个稍微不平衡，一个平衡
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratios = [[50, 50]] * 1 + [[98, 2]] * 9
per_round_samples = sum(class_ratios[1])

# rounds = int(len(X_train)/clients/sum(class_ratios[1]))  # 轮次数量
for r in range(100):
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        class_ratio = class_ratios[c]
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
# 目的，为了验证，一半客户端平衡数据对稍微不平衡数据集的影响
```
`CovA_SeverityVH`
```python
import os
import pandas as pd
base_path = 'E:/FedStream/data_set_syn/Synthetic0310/'
source_dataset_path = 'Make_CF/Syn130K_10F_2C/'
class_dir = f'{path}/130K_binary_class'
path = os.path.join(base_path,source_dataset_path)
if not os.path.exists(path):
    os.makedirs(path)
setting = 'Cov9B_SeverityVH'
# 9个稍微不平衡，一个平衡
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratios = [[98, 2]] * 10
per_round_samples = sum(class_ratios[1])

# rounds = int(len(X_train)/clients/sum(class_ratios[1]))  # 轮次数量
for r in range(100):
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        class_ratio = class_ratios[c]
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
# 目的，为了验证，一半客户端平衡数据对稍微不平衡数据集的影响
```

`CovA_SeverityVHR`

```python
import os
import pandas as pd
base_path = 'E:/FedStream/data_set_syn/Synthetic0310/'
source_dataset_path = 'Make_CF/Syn130K_10F_2C/'
class_dir = f'{path}/130K_binary_class'
path = os.path.join(base_path,source_dataset_path)
if not os.path.exists(path):
    os.makedirs(path)
setting = 'Cov9B_SeverityVH'
# 9个稍微不平衡，一个平衡
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratios = [[2,98]] * 10
per_round_samples = sum(class_ratios[1])

# rounds = int(len(X_train)/clients/sum(class_ratios[1]))  # 轮次数量
for r in range(100):
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        class_ratio = class_ratios[c]
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
# 目的，为了验证，一半客户端平衡数据对稍微不平衡数据集的影响
```

### 1.8 异步性`asynchronous`

所有异步我们设计的是98/2 -> 2/98变化这种，为了让变化前后数据量不变

```python
# 同步异步 step 3,5,7
setting = 'Rc1c2_Asyn_step7'
# [50,50]->[95,5] ,happend 1 times,direction forward/backward
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)
clients = 10  # 客户端数量
class_ratios = [[98,2]]*clients
per_round_samples = 100

# rounds = int(len(X_train)/clients/sum(class_ratio))  # 轮次数量
for r in range(100):
    # if r>=20 :
    #     class_ratio = [95,5]
    for c in range(clients):
        if  r > (20+7*c):
            class_ratios[c] = [2,98]
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        class_ratio = class_ratios[c]
        print(f"round {r}: client :{c},class_ratio :{class_ratio}")
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
```

```python
# 同步异步 step 3,5,7
setting = 'Rc1c2_Asyn_step5'
# [50,50]->[95,5] ,happend 1 times,direction forward/backward
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)
clients = 10  # 客户端数量
class_ratios = [[98,2]]*clients
per_round_samples = 100

# rounds = int(len(X_train)/clients/sum(class_ratio))  # 轮次数量
for r in range(100):
    # if r>=20 :
    #     class_ratio = [95,5]
    for c in range(clients):
        if  r > (20+5*c):
            class_ratios[c] = [2,98]
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        class_ratio = class_ratios[c]
        print(f"round {r}: client :{c},class_ratio :{class_ratio}")
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
```

### 1.9 方向性`Direction`

为了研究不同的客户端不平衡变化方向对全局性能的影响

```python
# 方向性(保持全局平衡，全局不平衡的)
# covH (half VH->VHR,VHR->VH)全局的角度来看没有大小类之分（98/2 ->2/98,2/98 ->98/2）(100,100)
# covH (half B->M,M->B)全局角度来看数据是第一类为大类 (50/50 ->98/2,98/2->50/50)(140/60)
# 相关性
# 每个客户端随机选择[B,VH,VHR,H,M,L]
# recurrence
import os
import pandas as pd
base_path = 'E:/FedStream/data_set_syn/Synthetic0310/'
source_dataset_path = 'Make_CF/Syn130K_10F_2C/'
path = os.path.join(base_path,source_dataset_path)
class_dir = f'{path}/130K_binary_class'
if not os.path.exists(path):
    os.makedirs(path)
setting = 'Direction_VHvsVHR_GB'
# 9个稍微不平衡，一个平衡
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratios = [[98, 2]] * 5 + [[2, 98]] * 5
per_round_samples = sum(class_ratios[1])

# rounds = int(len(X_train)/clients/sum(class_ratios[1]))  # 轮次数量
for r in range(100):
    # （98/2 ->2/98,2/98 ->98/2）
    if r >= 40: # 15 16 17 18 19
        class_ratios = [[2, 98]] * 5 + [[98, 2]] * 5

    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        class_ratio = class_ratios[c]
        if c ==0 or c ==6:
            print(f"round {r}: client :{c},class_ratio :{class_ratio}")
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)
```

```python
# 方向性(保持全局平衡，全局不平衡的)

# covH (half B->M,M->B)全局角度来看数据是第一类为大类 (50/50 ->98/2,98/2->50/50)(140/60)
# 相关性
# 每个客户端随机选择[B,VH,VHR,H,M,L]
# recurrence
import os
import pandas as pd
base_path = 'E:/FedStream/data_set_syn/Synthetic0310/'
source_dataset_path = 'Make_CF/Syn130K_10F_2C/'
path = os.path.join(base_path,source_dataset_path)
class_dir = f'{path}/130K_binary_class'
if not os.path.exists(path):
    os.makedirs(path)
setting = 'Direction_BvsVH_noGB'
# 9个稍微不平衡，一个平衡
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratios = [[98, 2]] * 5 + [[50, 50]] * 5
per_round_samples = sum(class_ratios[1])

# rounds = int(len(X_train)/clients/sum(class_ratios[1]))  # 轮次数量
for r in range(100):
    # （98/2 ->2/98,2/98 ->98/2）
    if r >= 40: # 15 16 17 18 19
        class_ratios = [[50, 50]] * 5 + [[98, 2]] * 5
    for c in range(clients):
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        class_ratio = class_ratios[c]
        if c ==0 or c ==6:
            print(f"round {r}: client :{c},class_ratio :{class_ratio}")
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)

```

### 1.10、相关性`correction`

```python
# 方向性(保持全局平衡，全局不平衡的)

# covH (half B->M,M->B)全局角度来看数据是第一类为大类 (50/50 ->98/2,98/2->50/50)(140/60)
# 相关性
# 每个客户端随机选择[B,VH,VHR,H,M,L]
# recurrence
import os
import pandas as pd
import random
base_path = 'E:/FedStream/data_set_syn/Synthetic0310/'
source_dataset_path = 'Make_CF/Syn130K_10F_2C/'
path = os.path.join(base_path,source_dataset_path)
class_dir = f'{path}/130K_binary_class'
if not os.path.exists(path):
    os.makedirs(path)
setting = 'Correlation2'
# 9个稍微不平衡，一个平衡
client_dataset_dir = f'{path}/130K_binary_client/{setting}/'
if not os.path.exists(client_dataset_dir):
    os.makedirs(client_dataset_dir)

clients = 10  # 客户端数量
class_ratiox = [50,50]
class_ratios = [[50, 50]] * 10
choose_list = [[50,50],[75,25],[90,10],[95,5],[98,2]]
per_round_samples = sum(class_ratios[1])

# rounds = int(len(X_train)/clients/sum(class_ratios[1]))  # 轮次数量
for r in range(100):
    # （98/2 ->2/98,2/98 ->98/2）
    change_flag = int(r/5)
    flag = r/5
    for c in range(clients):
        if flag == change_flag:
            class_ratiox = random.choices(choose_list, k=1)[0]
        #     print(f'r  : ',class_ratiox)
        # if (flag>change_flag and flag<(change_flag+1)):
            class_ratios[c] = class_ratiox
        dir = os.path.join(client_dataset_dir, f'client_{c}')
        if not os.path.exists(dir):
            os.makedirs(dir)
        single_df = pd.DataFrame()
        class_ratio = class_ratios[c]
        # if c ==0:
        print(f"round {r}: client :{c},class_ratio :{class_ratio}")
        for cc in range(2):
            class_df = pd.read_csv(os.path.join(class_dir, f'class_{cc}.csv'))
            sample_size = int(class_ratio[cc] / sum(class_ratio) * per_round_samples)  # 根据比例计算样本数量
            sample_df = class_df.sample(n=sample_size, replace=True)
            single_df = pd.concat([single_df, sample_df], ignore_index=True)

        single_df.to_csv(os.path.join(dir, f'round_{r}.csv'), index=False)

```

## 2、真实数据集Electricity的处理

### 2.1、读取源文件

```python
# 1.加载原始数据
from scipy.io import arff
import pandas as pd

# download site :https://moa.cms.waikato.ac.nz/datasets/
resource_path = 'E:/FedStream/real_data_set/elecNormNew_arff/elecNormNew.arff'
# 读取 arff 文件
data, meta = arff.loadarff(resource_path)

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 输出前6行数据
print(df.head(6))

```

### 2.2、字符数字化

```python
# 2.数据格式转换
import numpy as np

# 假设 df 是您的 DataFrame，且它包含 'day' 和 'class' 列

# day 列转换
# 将字节字符串转换为整数
df['day'] = df['day'].apply(lambda x: int(x.decode('utf-8')))

# 如果您想要一个列表而不是 DataFrame 列，可以这样做
day_list = df['day'].tolist()

# 确保 day_list 是从 1 到 7 的整数列表
day_list = [i if i in range(1, 8) else None for i in day_list]  # 如果存在非 1-7 的值，将其替换为 None

# class 列转换
# 创建映射字典
class_mapping = {b'UP': 0, b'DOWN': 1}

# 应用映射到 class 列
df['class'] = df['class'].map(lambda x: class_mapping[x])

# 如果您想要一个列表而不是 DataFrame 列，可以这样做
class_list = df['class'].tolist()

# 输出转换后的 day 和 class 信息
# print("day:", day_list)
# print("class:", class_list)
```

### 2.3、测试集训练集划分

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 假设 df 是你的 DataFrame

# 确保所有的列都是数值型的，否则你需要处理非数值型数据
# 例如，你可以使用 LabelEncoder 或 OneHotEncoder 来编码分类变量

# 分割 DataFrame 到特征 X 和目标 y
# 假设最后一列是目标变量，其他列是特征
X = df.iloc[:, :-1]  # 特征
y = df.iloc[:, -1]   # 目标

# 使用 train_test_split 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 如果你想要保持 DataFrame 的格式，你可以将 numpy 数组转换回 DataFrame
train_df = pd.DataFrame(X_train, columns=X.columns)
train_df['target'] = y_train

test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['target'] = y_test

# 现在 train_df 包含训练集的特征和目标，test_df 包含测试集的特征和目标
# 将训练集保存为 CSV 文件
train_df.to_csv('E:/Real DataSet/elecNorm/train_dataset.csv', index=False)

# 将测试集保存为 CSV 文件
test_df.to_csv('E:/Real DataSet/elecNorm/test_dataset.csv', index=False)
```

### 2.4、联邦数据集划分

```python
import os
client_nums = 10
per_round_data_nums = 100
base_path = 'E:/Real DataSet/elecNorm/Electricity_client/'
for r in range(100):    # 0,1,2...36
    for c in range(client_nums):     # 0,1,2...9
        start_index = (r*10 + c)*per_round_data_nums
        # end_index = start_index+per_round_data_nums
        path = os.path.join(base_path,f'client_{c}')
        if not os.path.exists(path):
            os.makedirs(path)
        selected_data = train_df.iloc[start_index:start_index + per_round_data_nums]
        file_path = os.path.join(path,f'round_{r}.csv')
        selected_data.to_csv(file_path, index=False)
        print(f"client: {c} round:{r}")
```

```python

```

## 3、PokerHand数据集处理

### 2.1、读取源文件

```python
import os
import pandas as pd
from scipy.io import arff
import numpy as np
source_path = 'E:/FedStream/real_data_set/poker_lsn_arff/poker-lsn.arff'
# 读取 arff 文件
data, meta = arff.loadarff(source_path)

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 输出前6行数据
print(df.head(6))

```

### 2.2、字符数字化

```python
class_counts = df['class'].value_counts()
print(class_counts)
# 将字节字符串转换为整数
for i in range(5):
    df[f's{i+1}']= df[f's{i+1}'].apply(lambda x: int(x.decode('utf-8')))
df['class']= df['class'].apply(lambda x: int(x.decode('utf-8')))
print(df.head(5))
```

### 2.3、类别映射

```python
class_counts = df['class'].value_counts()
print(class_counts)
"""
0    415526 没有任何特殊牌型
1    350426 一对（One Pair）	即有两张相同数字的牌
2     39432 两对（Two Pairs）	即有两对相同数字的牌
3     17541 三条（Three of a Kind）	即有三张相同数字的牌
4      3225 顺子（Straight）	即五张连续数字的牌
5      1657 同花（Flush）	即五张同一花色的牌
6      1186 葫芦（Full House）	即三张相同数字和一对相同数字的牌(应该不能保留)
7       195 四条（Four of a kind）	即有四张相同数字的牌
8        11 同花顺（Straight Flush）	即既是顺子又是同花的牌
9         2 皇家同花顺（Royal Flush）	即从10到A的同花顺
0,     1(对子),     2(两对),    (3+6+7 有3张数字相同的牌),(4+5+8+9同花和顺子)
415526 350426 39432 17541+1186+195(18922)  3225+1657+11+2(4895)
84.88  71.588 8.05  3.8655                  1
"""
# 829201
# 85+ 71.5+8+4+1 = 170
# [170,143,16,7,2]
#round client   sample_nums
# 829  10       100
# 415  10       200
# 243  10       340
# 415526(0无),350426(1对)，39432(两对2)，17541+1186+195(3张相同数字3，6，7)，3225+1657+11+2（同花和顺子4，5，8，9）
# 415526(0无),350426(1对)，39432(两对2)，17541+1186+195(3张相同数字3，6，7)，3225+1657+11+2（同花和顺子4，5，8，9）
# 0,     1(对子),     2(两对),    (3+6+7 有3张数字相同的牌),(4+5+8+9同花和顺子)
# 415526 350426       39432     17541+1186+195(18922)  3225+1657+11+2(4895)
# 84.88  71.588       8.05      3.8655                  1
# print(17541+195)
# print(sum([415526 ,350426,39432,17736,4895]))
# print(17736/4895*2)
# print(39432/4895*2)
# print(350426/4895*2)
# print(415526/4895*2)
# [169,143,16,7,2]
"""
# 0,     1(对子),     2(两对),    (3+6+7 有3张数字相同的牌),(4+5+8+9同花和顺子)
# 415526 350426 39432 17541+1186+195(18922)  3225+1657+11+2(4895)
# 84.88  71.588 8.05  3.8655                  1
"""
# 829201
# 85+ 71.5+8+4+1 = 170
# [170,143,16,7,2]
#round client   sample_nums
# 829  10       100
# 415  10       200
# 243  10       340
# 创建一个映射字典进行类别替换
import copy
dffive = copy.deepcopy(df)
class_mapping = {3: 3, 6: 3, 7: 3, 4: 4, 5: 4, 8: 4, 9: 4}

# 使用replace()函数替换类别
dffive['class'] = dffive['class'].replace(class_mapping)
class_countsf = dffive['class'].value_counts()
print(class_countsf)
dffive.to_csv('E:/FedStream/real_data_set/realdataset0427/pokerhand_fiveclass.csv', index=False)
```

### 2.4、测试集训练集划分

```python
from sklearn.model_selection import train_test_split

# 假设df是包含新的数据集的DataFrame，df['class']是类别列
# 按类别进行数据集拆分
train_data = pd.DataFrame()  # 存储训练集数据
test_data = pd.DataFrame()   # 存储测试集数据

# 按类别拆分数据集
for class_label in dffive['class'].unique():
    class_data = dffive[dffive['class'] == class_label]  # 获取特定类别的数据

    # 将特定类别的数据集拆分为训练集和测试集
    X_train, X_test = train_test_split(class_data, test_size=0.2, shuffle=False)

    # 将拆分后的数据添加到相应的数据集中
    train_data = train_data.append(X_train)
    test_data = test_data.append(X_test)

class_countsft = train_data['class'].value_counts()
print(class_countsft)
class_countsftt = test_data['class'].value_counts()
print(class_countsftt)
# 将训练集保存为 CSV 文件
train_data.to_csv('E:/FedStream/real_data_set/realdataset0427/pokerhand_five/train.csv', index=False)

# 将测试集保存为 CSV 文件
test_data.to_csv('E:/FedStream/real_data_set/realdataset0427/pokerhand_five/test.csv', index=False)
```

### 2.4、联邦数据集划分

```python
import os
import pandas as pd

# 假设以下变量已定义
class_ratios = [170, 143, 16, 7, 2]
class_labels = train_data['class'].unique()  # 确保dffive里有class_label这一列
total_samples_per_file = sum(class_ratios)  # 每个文件的总样本数

# 基本目录
base_dir = 'E:/FedStream/real_data_set/realdataset0427/pokerhand_five/pokerhand_client/'

# 创建文件夹（如果尚不存在）
for j in range(10):
    os.makedirs(os.path.join(base_dir, f"client_{j}"), exist_ok=True)

# 计算可以生成的总文件数
# 3916/2 = 1958
# 分割数据并保存为文件
for i in range(1958,1959):
    samples = []
    sample_counts = {label: 0 for label in class_labels}  # 初始化字典记录每个类别的样本数量

    train_data = dffive.copy()  # 每次循环重新复制数据

    for label, ratio in zip(class_labels, class_ratios):
        class_subset = train_data[train_data['class'] == label]
        if len(class_subset) < ratio:
            print(f"Not enough samples for class {label} in round {i}")
            continue  # 如果剩余样本不足，则跳过此类别
        sample = class_subset.sample(n=ratio, replace=False)
        train_data = train_data.drop(sample.index)  # 实现不放回抽样
        samples.append(sample)
        sample_counts[label] += len(sample)
        print(f'class: {label}, number: {len(sample)}')

    # 合并样本并保存到文件
    result_df = pd.concat(samples, ignore_index=True)
    client_id = i % 10
    round_number = i // 10
    file_path = os.path.join(base_dir, f"client_{client_id}", f"round_{round_number}.csv")
    result_df.to_csv(file_path, index=False)
    print(f'round: {round_number}, client: {client_id}')
    print(f"Round {i}: {sample_counts}")
```

# 二、联邦训练、测试

## 1、合成数据集

```python
 # 上诉代码没有共享模型
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,recall_score, accuracy_score, roc_auc_score,cohen_kappa_score,confusion_matrix,precision_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import copy
import os
import random
from sklearn.cluster import KMeans
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class IncrementalSampling(object):
    def __init__(self,prototype_nums = 10):
        self.incremental_data_map = {}
        self.data  = np.array([])
        self.label = np.array([])
        self.k_value = prototype_nums
    def split_current_data_by_class(self):
        """把当前轮的数据按类别划分"""
        # 获取唯一的类别标签
        current_unique_labels = np.unique(self.label)
        # 按类别分割当前轮获得的数据
        data_by_class = {}  # map {"label":[the data belong to the label]}
        for current_data_label in current_unique_labels:
            indices = np.where(self.label == current_data_label)[0]
            data_by_class[current_data_label] = self.data[indices]
        return data_by_class, current_unique_labels
    def data_combined(self):
        """把当前轮的数据，和类map中的数据进行合并"""
        current_data_map_by_class, current_unique_labels = self.split_current_data_by_class()
        # 考虑到有新类出现的情况
        # 如果current_unique_labels有新label,直接扩充incremental_data_map
        # 如果label 是incremental_data_map中已经有的，扩张incremental_data_map对应label中data的长度
        for new_data_label in current_unique_labels:
            if new_data_label in self.incremental_data_map:
                # 增量map中已经有这个标签的数据了，那就扩充这个数据
                self.incremental_data_map[new_data_label] = np.concatenate(
                    (self.incremental_data_map[new_data_label], current_data_map_by_class[new_data_label])
                )
            else:
                # 如果增量map中没有这个标签的的数据，就扩充增量map
                self.incremental_data_map[new_data_label] = current_data_map_by_class[new_data_label]
    def cluster_data(self,data, num_clusters):
        # kmeans = KMeans(n_clusters=num_clusters,n_init='auto')
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data)
        cluster_centers = kmeans.cluster_centers_
        return  cluster_centers
    def reCludter(self):
        # incremental_data_map的大小进行压缩self.reCluster_instance_nums
        new_cluster = {}
        for label, data in self.incremental_data_map.items():
            if len(data) <= self.k_value:
                # 全部保存
                new_cluster[label] = data
            else:
                sampled_nums = self.k_value
                clusters = self.cluster_data(data = data, num_clusters=sampled_nums)
                new_cluster[label] = clusters
        self.incremental_data_map = new_cluster
    def compute_sampling_nums(self,combinde_map):
        # initialize
        min_length = float('inf')  # 初始化最小长度为正无穷大
        max_length = 0  # 初始化最大长度为0
        for label, data in combinde_map.items():
            data_length = len(data)
            if data_length < min_length:
                min_length = data_length
            if data_length > max_length:
                max_length = data_length
        return min_length, max_length
    def data_sampling(self,sampling_nums):
        resampling_data = []
        resampling_label = []
        for label, data in self.incremental_data_map.items():
            if len(data) > sampling_nums:
                # 那就是下采样了,不放回采样
                sampled_indices = random.sample(range(len(data)), sampling_nums)
                sampled_data = [data[i] for i in sampled_indices]
                resampling_data.extend(sampled_data)
                resampling_label.extend([label] * sampling_nums)
            elif len(data) == sampling_nums:
                # 直接复制
                resampling_data.extend(data)
                resampling_label.extend([label] * sampling_nums)
            else:
                # 上采样,保存原样本
                resampling_data.extend(data)
                # 随机有放回的找差额部分
                sampled_data = random.choices(data, k=(sampling_nums-len(data)))
                resampling_data.extend(sampled_data)
                resampling_label.extend([label] * sampling_nums)
        # 洗牌
        combined_data = list(zip(resampling_data, resampling_label))
        random.shuffle(combined_data)
        resampling_data, resampling_label = zip(*combined_data)
        return resampling_data,resampling_label
    def fit(self,incremental_prototype_map,data,label,sampling_strategy = "OverSampling"):
        self.incremental_data_map = incremental_prototype_map
        self.data = data
        self.label = label
        self.data_combined()
        resampling_data = []
        resampling_label = []
        min_length, max_length =self.compute_sampling_nums(self.incremental_data_map)
        if sampling_strategy.lower() == "oversampling":
            resampling_data,resampling_label = self.data_sampling(max_length)
        elif sampling_strategy.lower() == "downsampling":
            resampling_data,resampling_label = self.data_sampling(min_length)
        else:
            print("No sampling measures have been taken")
        self.reCludter()
        return resampling_data,resampling_label,self.incremental_data_map

class Triplets(object):
    def __init__(self, n_neighbors=5, random=True, len_lim=True, **kwargs):
        self.n_neighbors = n_neighbors
        self.random = random
        self.len_lim = len_lim

    def fit_resample(self, x, y):
        strategy = self._sample_strategy(y)
        self.n_neighbors = max(self.n_neighbors, self.counts.max() // self.counts.min())

        gen_x = []
        gen_y = []
        # 这里的代码平衡状态会报错
        for c, size in enumerate(strategy):
            if size == 0: continue
            weight = self._weights(x, y, c)
            gen_x_c, gen_y_c = self._sample_one(x, y, c, size, weight)
            gen_x += gen_x_c
            gen_y += gen_y_c

        # 为了这个方法在平衡状态下不报错，我们特地在这里加了这段代码
        # To prevent errors in this method when in a balanced state, we intentionally added this code block
        if len(gen_x)==0:
            return x,y
        gen_x = np.vstack(gen_x)
        gen_y = np.array(gen_y)
        return np.concatenate((x, gen_x), axis=0), np.concatenate((y, gen_y), axis=0)

    def _sample_strategy(self, y):
        _, self.counts = np.unique(y, return_counts=True)
        return max(self.counts) - self.counts

    def _weights(self, x, y, c):
        return np.ones(self.counts[c])

    def _sample_one(self, x, y, c, size, weight):
        gen_x = []
        gen_y = []
        if size == 0: return gen_x, gen_y

        # get the indices of minority and majority instances
        min_idxs = np.where(y == c)[0]
        maj_idxs = np.where(y != c)[0]

        # find nearest majority neighbors for each minority instance
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(x[maj_idxs])
        _, indices = nbrs.kneighbors(x[min_idxs])

        # generate synthetic data
        for j in np.random.choice(len(min_idxs), size, p=weight / weight.sum()):
            tp1 = x[min_idxs[j]]
            tp2 = x[maj_idxs[indices[j][:5]]].mean(axis=0)
            # tp3_ord = np.random.randint(1, self.n_neighbors)
            tp3_ord = np.random.randint(self.n_neighbors)
            tp3 = x[maj_idxs[indices[j][tp3_ord]]]
            if (tp2 == tp1).all():
                gen_x.append(tp1)
                gen_y.append(c)
                continue

            offset = tp3 - tp2
            if self.len_lim: offset = offset * min(1, norm(tp1 - tp2) / norm(offset))
            coef = np.random.rand() if self.random is True else 1.0
            new_x = tp1 + coef * offset
            gen_x.append(new_x)
            gen_y.append(c)
        return gen_x, gen_y

def train_model_FedAvg_local(input_model,X_train_tensor, y_train_tensor, num_epochs):
    losses =[]
    model = input_model# copy.deepcopy(input_model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        losses.append(loss.item())  # 将每次训练的损失值添加到列表中

    return losses
def train_model_FedProx_local(input_model, X_train_tensor, y_train_tensor, num_epochs):
    mu = 0.1
    # because last round trained global model replaced local model,
    # that in this round the first local model is last round global model
    losses =[]
    model = input_model# copy.deepcopy(input_model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    global_weights = copy.deepcopy(list(model.parameters()))
    # not use deepcopy ,because model as parameter transport in this ,update model also update model
    # current_local_model = cmodel
    # model.train()
    # for epoch in range(num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        # FedProx
        prox_term = 0.0
        for p_i, param in enumerate(model.parameters()):
                prox_term += (mu / 2) * torch.norm((param - global_weights[p_i])) ** 2
        loss += prox_term

        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        losses.append(loss.item())  # 将每次训练的损失值添加到列表中
    #     total_loss = 0.0
    #     for step,(x,y) in enumerate(zip(X_train_tensor,y_train_tensor)):
    #         # current_local_model.train()
    #         output = model(x) # current_local_model(x)
    #         loss = criterion(output, y)
    #         total_loss+=loss.item()
    #         optimizer.zero_grad()
    #
    #         # fedprox
    #         prox_term = 0.0
    #         for p_i, param in enumerate(model.parameters()):
    #             prox_term += (mu / 2) * torch.norm((param - global_weights[p_i])) ** 2
    #         loss += prox_term
    #         loss.backward()
    #         optimizer.step()
    #     losses.append(total_loss)
    #     if epoch % 10 == 0:
    #         print(f"Epoch [{epoch+1}/{num_epochs}], Epoch Total Loss: {total_loss:.4f}")
    return losses

def train_model_FedNova_local(input_model,X_train_tensor, y_train_tensor, num_epochs):
    # because last round trained global model replaced local model,
    # that in this round the first local model is last round global model
    losses =[]
    model = input_model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    global_weights = copy.deepcopy(input_model.state_dict())
    tau = 0
    rho = 0.9
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        tau +=len(y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        losses.append(loss.item())
    # for epoch in range(num_epochs):
    #     total_loss = 0.0
    #     for step,(x,y) in enumerate(zip(X_train_tensor,y_train_tensor)):
    #         # current_local_model.train()
    #         # model.train()
    #         output = model(x) # current_local_model(x)
    #         loss = criterion(output, y)
    #         total_loss+=loss.item()
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         tau +=1
    #     losses.append(total_loss)
    #     if epoch % 10 == 0:
    #         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
    coeff = (tau - rho * (1 - pow(rho, tau)) / (1 - rho)) / (1 - rho)
    state_dict = model.state_dict()
    norm_grad = copy.deepcopy(global_weights)
    for key in norm_grad:
        norm_grad[key] = torch.div(global_weights[key] - state_dict[key], coeff)

    return losses, coeff, norm_grad,len(X_train_tensor)
# 定义模型参数共享函数
def share_params(model):
    params = model.state_dict()
    return {k: v.clone().detach().requires_grad_(True) for k, v in params.items()}

# # 定义模型参数聚合函数
def aggregate_params(params_list):
    aggregated_params = {}
    for key in params_list[0].keys():
        # 将参数转换为张量进行处理
        params_tensors = [params[key].clone().detach().float() for params in params_list]
        # 聚合参数
        aggregated_params[key] = sum(params_tensors) / len(params_tensors)
    return aggregated_params
def test(global_model,X_test,y_test):
    # 在全局模型上进行测试
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        outputs = global_model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        # 计算度量
        predictions = predicted.numpy() # 将张量转换为 NumPy 数组并去除零维数组
        true_labels = y_test_tensor.numpy()  # 将张量转换为 NumPy 数组并去除零维数组
        precision = precision_score(true_labels,predictions,zero_division=0.0,average=None)
        precision_micro = precision_score(true_labels,predictions,zero_division=0.0,average='micro')
        precision_macro = precision_score(true_labels,predictions,zero_division=0.0,average='macro')
        # recall
        recalls = recall_score(true_labels,predictions,zero_division=0.0,average=None)
        recalls_micro =recall_score(true_labels,predictions,zero_division=0.0,average='micro')
        recalls_macro =recall_score(true_labels,predictions,zero_division=0.0,average='macro')
        f1_scores = f1_score(true_labels, predictions, average=None)
        acc = accuracy_score(true_labels, predictions)
        kappa = cohen_kappa_score(true_labels,predictions)
        conf_matrix = confusion_matrix(true_labels,predictions)
        # 计算所有类别乘积的几何平均值作为 G-mean
        g_mean_all= np.power(np.prod(recalls), 1 / len(recalls))
        # AUC
        lb = LabelBinarizer()
        lb.fit(true_labels)
        true_labels_bin = lb.transform(true_labels)
        predictions_bin = lb.transform(predictions)
        auc = roc_auc_score(true_labels_bin, predictions_bin, average='weighted', multi_class='ovr')
        metrics = {
            'recall':recalls,
            'recall_micro':recalls_micro,
            'recall_macro':recalls_macro,
            'precision':precision,
            'precision_micro':precision_micro,
            'precision_macro':precision_macro,
            'f1_score':f1_scores,
            'g_mean':g_mean_all,
            'acc':acc,
            'auc':auc,
            'kappa':kappa,
            'confusion_matrix':conf_matrix
        }
        return metrics

def save_loss(loss_list,client_id,round_id,save_loss_path):
    if not os.path.exists(save_loss_path):
        os.makedirs(save_loss_path)
    # 构建文件路径
    file_path = os.path.join(save_loss_path, f"client_{client_id}.csv")

    if os.path.exists(file_path):
        # 如果文件存在，加载现有的 CSV 文件为 DataFrame
        df = pd.read_csv(file_path)
    else:
        # 如果文件不存在，直接创建新的 DataFrame
        df = pd.DataFrame()

    # 将损失值添加到 DataFrame 中
    column_name = f'round_{round_id}'
    df[column_name] = loss_list

    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(file_path, index=False)
def save_model(global_model,round_id,save_model_path):
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    model_path = os.path.join(save_model_path,f'round_{round_id}_gm.pt')
    torch.save(global_model,model_path)

def save_metrics(title, rounds, metrics, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    file_name = f"{title}.csv"
    file_path = os.path.join(save_folder, file_name)
    recalls = metrics['recall']
    class_nums = len(recalls)
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 如果文件存在，加载现有的 Excel 文件为 DataFrame
        df = pd.read_csv(file_path)
    else:
        # 如果文件不存在，直接创建新的 DataFrame
        columns = [
        'rounds', 'accuracy', 'auc', 'kappa', 'g_mean', 'recall_micro', 'precision_micro',
        'recall_macro', 'precision_macro'
        ]
        df = pd.DataFrame(columns=columns)
        for i in range(class_nums):  # 动态生成 f1-score 相关列名
            columns.append(f'f1_score_{i}')
            columns.append(f'recall_{i}')
            columns.append(f'precession_{i}')

    data = {
        'rounds': rounds,
        'accuracy': metrics['acc'],
        'auc': metrics['auc'],
        'kappa': metrics['kappa'],
        'g_mean':metrics['g_mean'],
        'recall_micro':metrics['recall_micro'],
        'precision_micro':metrics['precision_micro'],
        'recall_macro':metrics['recall_macro'],
        'precision_macro':metrics['precision_macro']
    }
    # 添加每个类别的 F1-score、G-mean 和 Recall 到 data 中
    for i in range(class_nums):  #类别数
        data[f'recall_{i}'] = metrics['recall'][i]
        data[f'precision_{i}'] = metrics['precision'][i]
        data[f'f1_score_{i}'] = metrics['f1_score'][i]
    # 创建新行并追加到 DataFrame
    new_row = pd.DataFrame(data, index=[0])
    df = pd.concat([df, new_row], ignore_index=True)

    # 将 DataFrame 保存为 Excel 文件
    df.to_csv(file_path, index=False)

from imblearn.over_sampling import SMOTE,RandomOverSampler,SMOTENC,SMOTEN,ADASYN,BorderlineSMOTE,KMeansSMOTE,SVMSMOTE
from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler,NearMiss,TomekLinks,EditedNearestNeighbours,RepeatedEditedNearestNeighbours,AllKNN,CondensedNearestNeighbour,OneSidedSelection,NeighbourhoodCleaningRule,InstanceHardnessThreshold
from imblearn.combine import SMOTEENN,SMOTETomek
# SMOTE,ROS,SMOTENC,SMOTEN,ADASYN,BorderlineSMOTE1,BorderlineSMOTE2,KMeansSMOTE,SVMSMOTE
# ClusterCentroids,RUS,NearMiss1,NearMiss2,NearMiss2,TomekLinks,ENN,RENN,AllKNN,CNN,OSS,NC,IHT
# SMOTEENN,SMOTETomek
def data_sampling(raw_X,raw_y,sampling_strategy):
    if sampling_strategy.upper() == 'NO' :
        return raw_X,raw_y
    # overSampling
    elif sampling_strategy.upper() == 'SMOTE': # overSampling
        """
            1.对于样本x ,按照欧氏距离找到离其距离最近的K个近邻样本
            2.确定采样比例，然后从K个近邻中选择x_n
            3.公式 x_new = x + rand(0,1)*(x_n-x)
        """
        smote = SMOTE( random_state=42)
        X_resampled, y_resampled = smote.fit_resample(raw_X,raw_y)
        return X_resampled, y_resampled
    elif sampling_strategy == "RandomOverSampler" or sampling_strategy.upper()=='ROS': # overSampling
        ros = RandomOverSampler(random_state=1)
        ros_data,ros_label = ros.fit_resample(raw_X,raw_y)
        return ros_data,ros_label
    elif sampling_strategy.upper() == 'SMOTENC':    # overSampling
        smotenc = SMOTENC(random_state=1,categorical_features=[0])
        smotenc_data,smotenc_label = smotenc.fit_resample(raw_X,raw_y)
        return smotenc_data,smotenc_label
    elif sampling_strategy.upper() == 'SMOTEN': # overSampling
        smoten = SMOTEN(random_state=1)
        smoten_data,smoten_label = smoten.fit_resample(raw_X,raw_y)
        return smoten_data,smoten_label
    elif sampling_strategy.upper() =='ADASYN':
        adasyn = ADASYN(random_state=1)
        adasyn_data,adasyn_label = adasyn.fit_resample(raw_X,raw_y)
        return adasyn_data,adasyn_label
    elif sampling_strategy == 'BorderlineSMOTE1' or sampling_strategy.upper()=='BSMOTE1':
        bsmote1 = BorderlineSMOTE(kind='borderline-1',random_state=1)
        bsmote1_data,bsmote1_label = bsmote1.fit_resample(raw_X,raw_y)
        return bsmote1_data,bsmote1_label
    elif sampling_strategy == 'BorderlineSMOTE2'or sampling_strategy.upper()=='BSMOTE2':
        bsmote2 = BorderlineSMOTE(kind='borderline-2',random_state=1)
        bsmote2_data,bsmote2_label = bsmote2.fit_resample(raw_X,raw_y)
        return bsmote2_data,bsmote2_label
    elif sampling_strategy == 'KMeansSMOTE' or sampling_strategy.upper() == 'KSMOTE':
        kmeanssmote = KMeansSMOTE(random_state=1)
        kmeanssmote_data,kmeanssmote_label = kmeanssmote.fit_resample(raw_X,raw_y)
        return kmeanssmote_data,kmeanssmote_label
    elif sampling_strategy == 'SVMSMOTE':
        svmsmote = SVMSMOTE(random_state=1)
        svmsmote_data,svmsmote_label = svmsmote.fit_resample(raw_X,raw_y)
        return svmsmote_data,svmsmote_label
    # downSampling
    elif sampling_strategy == 'ClusterCentroids': # down-sampling,generate
        clustercentroids = ClusterCentroids(random_state=1)
        clustercentroids_data,clustercentroids_label = clustercentroids.fit_resample(raw_X,raw_y)
        return clustercentroids_data,clustercentroids_label
    elif sampling_strategy=='RandomUnderSampler' or sampling_strategy.upper()=='RUS':
        rus = RandomUnderSampler(random_state=1)
        rus_data,rus_label = rus.fit_resample(raw_X,raw_y)
        return rus_data,rus_label
    elif sampling_strategy.upper() =='NEARMISS1':
        # 在k个少数类别样本中，选择出与他们-平均距离最近的多数类样本-进行保存
        nearmiss1 = NearMiss(version=1)
        nearmiss1_data,nearmiss1_label = nearmiss1.fit_resample(raw_X,raw_y)
        return nearmiss1_data,nearmiss1_label
    elif sampling_strategy.upper() =='NEARMISS2':
        # 选择K个距离最远的少数类别样本，然后根据这些样本选出的"平均距离最近"的样本进行保存
        nearmiss2 = NearMiss(version=2)
        nearmiss2_data,nearmiss2_label = nearmiss2.fit_resample(raw_X,raw_y)
        return nearmiss2_data,nearmiss2_label
    elif sampling_strategy.upper() =='NEARMISS3':
        # 1、对于每一个少数类别样本，保留其K个最近邻多数类样本；2、把到K个少数样本平均距离最大的多数类样本保存下来。
        nearmiss3 = NearMiss(version=3)
        nearmiss3_data,nearmiss3_label = nearmiss3.fit_resample(raw_X,raw_y)
        return nearmiss3_data,nearmiss3_label
    elif sampling_strategy == 'TomekLinks' or sampling_strategy.upper()=='TOMEK':
        # 它需要计算每个样本之间的距离，然后把互为最近邻且类别不同的一对样本拿出来，根据需求的选择将这一对数据进行剔除 or 把多数类样本剔除
        tomelink = TomekLinks(sampling_strategy='all')#sampling_strategy='all'表示全部删除，'auto'表示只删除多数类
        tomelink_data,tomelink_label = tomelink.fit_resample(raw_X,raw_y)
        return tomelink_data,tomelink_label
    elif sampling_strategy == 'EditedNearestNeighbours' or sampling_strategy.upper() == 'ENN':
        ENN = EditedNearestNeighbours()
        ENN_data,ENN_label = ENN.fit_resample(raw_X,raw_y)
        return ENN_data,ENN_label
    elif sampling_strategy == 'RepeatedEditedNearestNeighbours' or sampling_strategy.upper() == 'RENN':
        RENN = RepeatedEditedNearestNeighbours()
        RENN_data,RENN_label = RENN.fit_resample(raw_X,raw_y)
        return RENN_data,RENN_label
    elif sampling_strategy =='AllKNN':
        ## ENN的改进版本，和RepeatedEditedNearestNeighbours一样，会多次迭代ENN 算法，不同之处在于，他会每次增加KNN的K值
        allknn = AllKNN()
        allknn_data,allknn_label = allknn.fit_resample(raw_X,raw_y)
        return allknn_data,allknn_label
    elif sampling_strategy == 'CondensedNearestNeighbour'or sampling_strategy.upper() == 'CNN':
        ## 如果有样本无法和其他多数类样本聚类到一起，那么说明它极有可能是边界的样本，所以将这些样本加入到集合中
        CNN = CondensedNearestNeighbour(random_state=1)
        CNN_data,CNN_label = CNN.fit_resample(raw_X,raw_y)
        return CNN_data,CNN_label
    elif sampling_strategy == 'OneSidedSelection' or sampling_strategy.upper() == 'OSS':
        # OneSidedSelection = tomekLinks + CondensedNearestNeighbour,先使用自杀式的方式把大类数据中的其他值剔除，然后再使用CondensedNearestNeighbour的下采样
        OSS = OneSidedSelection(random_state=1)
        OSS_data,OSS_label = OSS.fit_resample(raw_X,raw_y)
        return OSS_data,OSS_label
    elif sampling_strategy == 'NeighbourhoodCleaningRule'or sampling_strategy.upper() == 'NC':
        # 若在大类的K-近邻中，少数类占多数，那就剔除这个多数类别的样本
        NC = NeighbourhoodCleaningRule()
        NC_data,NC_label = NC.fit_resample(raw_X,raw_y)
        return NC_data,NC_label
    elif sampling_strategy == 'InstanceHardnessThreshold' or sampling_strategy.upper() == 'IHT':
        # 默认算法是随机森林，通过分类算法给出样本阈值来剔除部分样本，（阈值较低的可以剔除）,慢
        IHT = InstanceHardnessThreshold(random_state=1)
        IHT_data,IHT_label = IHT.fit_resample(raw_X,raw_y)
        return IHT_data,IHT_label
    # hibird
    elif sampling_strategy.upper() =='SMOTEENN':
        se = SMOTEENN(random_state=1)
        se_data,se_label = se.fit_resample(raw_X,raw_y)
        return se_data,se_label
    elif sampling_strategy.upper() =='SMOTETOMEK':
        st = SMOTETomek(random_state=1)
        st_data,st_label = st.fit_resample(raw_X,raw_y)
        return st_data,st_label
    elif sampling_strategy == 'Triplets':
        print(" Triplets sampling")
        tpl = Triplets()
        tpl_data,tpl_label = tpl.fit_resample(raw_X,raw_y)
        return tpl_data,tpl_label
    else :
        print("skipped all the sampling strategy,but return the raw data and label")
        return raw_X,raw_y
def inremental_sampling(prototype_map,raw_X,raw_y):
    class_nums = len(np.unique(raw_y))
    print("class nums" ,class_nums)
    isap = IncrementalSampling()
    resampling_data,resampling_label,prototype_map = isap.fit(incremental_prototype_map=prototype_map,
                                                              data=raw_X,
                                                              label=raw_y)
    return resampling_data,resampling_label,prototype_map

def read_data_return_tensor(dataset_path, round_id, client_id, sampling_strategy='no',prototype_map = {}):
    folder_path = os.path.join(dataset_path, f'client_{client_id}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    open_file_path = os.path.join(folder_path, f'round_{round_id}.csv')
    data = pd.read_csv(open_file_path, header=None)
    # 排除标题行并打乱数据
    data_shuffled = shuffle(data.iloc[1:])  # 排除标题行并打乱数据

    # 提取特征和目标变量
    raw_X = data_shuffled.iloc[:, :-1].values  # 特征
    raw_y = data_shuffled.iloc[:, -1].values   # 目标变量
    raw_y = raw_y.astype(int)
    # print(len(raw_X),type(raw_X))
    # print(len(raw_X),type(raw_X))
    if sampling_strategy == 'IncrementalSampling':
        print("using IncrementalSampling")
        resampling_X,resampling_y,prototype_map = inremental_sampling(prototype_map,raw_X,raw_y)
    else:
        resampling_X,resampling_y = data_sampling(raw_X,raw_y,sampling_strategy)
        resampling_y.astype(float)
    resampling_X = np.array(resampling_X)  # 将列表转换为单个NumPy数组
    resampling_y = np.array(resampling_y)  # 将列表转换为单个NumPy数组

    X_train_tensor = torch.tensor(resampling_X, dtype=torch.float32)  # 特征张量
    y_train_tensor = torch.tensor(resampling_y, dtype=torch.long)   # 标签张量
    return X_train_tensor,y_train_tensor,prototype_map

def read_test_data(test_data_path):
    data = pd.read_csv(test_data_path, header=None)
    # 排除标题行并打乱数据
    data_shuffled = shuffle(data.iloc[1:])  # 排除标题行并打乱数据

    # 提取特征和目标变量
    X = data_shuffled.iloc[:, :-1].values  # 特征
    y = data_shuffled.iloc[:, -1].values   # 目标变量
    y = y.astype(float)

    X_test_tensor = torch.tensor(X, dtype=torch.float32)  # 特征张量
    y_test_tensor = torch.tensor(y, dtype=torch.long)   # 标签张量
    return X_test_tensor,y_test_tensor


```

```python
 def aggregate_fednova(local_params_list,gm):
    # (share_params(clients_models[i]),coeff, norm_grad, data_len) as input
    total_data_len = sum(data_len for _, _, _, data_len in local_params_list)
    global_model_state = gm.state_dict()
    nova_model_state = copy.deepcopy(global_model_state)
    # avg_loss = 0
    coeff = 0.0
    for clientID,(client_model,client_coeff,client_norm_grad,client_local_data_len) in enumerate(local_params_list):
        coeff = coeff + client_coeff*client_local_data_len/total_data_len
        for key in client_model.state_dict():
            if clientID == 0:
                nova_model_state[key] = client_norm_grad[key] * client_local_data_len/total_data_len
            else:
                nova_model_state[key] =nova_model_state[key]+ client_norm_grad[key] * client_local_data_len/total_data_len
        # avg_loss = avg_loss + cl
    for key in global_model_state:
        global_model_state[key] -= coeff*nova_model_state[key]

    return global_model_state
pp = []
def runFedNova(samplingName,settingName):
    num_clients = 10
    # 初始化全局模型和客户端模型
    input_size = 10
    hidden_size = 100
    output_size = 2
    global_model = MLP(input_size, hidden_size, output_size)
    client_prototype_map = [{} for _ in range(num_clients)]
    clients_models = [MLP(input_size, hidden_size, output_size) for _ in range(num_clients)]

    num_epochs = 200
    num_global_updates = 100
    base_path = 'E:/FedStream/'
    dataset_name = '130K_binary_client'
    setting_name = settingName # 'CovA_Abrupt_BtoVH'
    # SMOTE cant use in VH,H
    # sampling_strategy_name_list = ['no','RandomOverSampler','RandomUnderSampler','CondensedNearestNeighbour','Triplets','IncrementalSampling']
    # sampling_strategy_name = sampling_strategy_name_list[sampling_id]
    sampling_strategy_name = samplingName

    algorithm = 'FedNova'
    experiment_times = 'epoch200_1'
    save_loss_path = f'{base_path}/loss/{dataset_name}_Sampling/{algorithm}/{setting_name}/{sampling_strategy_name}_{experiment_times}'
    save_model_path = f'{base_path}/models/{dataset_name}_Sampling/{algorithm}/{setting_name}/{sampling_strategy_name}_{experiment_times}'
    save_metrics_path  = f'{base_path}/metrics/{dataset_name}_Sampling/{algorithm}/{setting_name}/{sampling_strategy_name}_{experiment_times}'
    read_data_path = f'E:/FedStream/data_set_syn/Synthetic0310/Make_CF/Syn130K_10F_2C/{dataset_name}/{setting_name}/'
    # 读取测试集CSV文件并转换为PyTorch张量
    test_path = 'E:/FedStream/data_set_syn/Synthetic0310/Make_CF/Syn130K_10F_2C/test/test.csv'
    X_test_tensor,y_test_tensor = read_test_data(test_path)

    for update in range(num_global_updates):
        print(f"round{update}")
        local_params_list = []

        for i in range(num_clients):
            # 在每个客户端训练本地模型
            print(f"client{i}")
            X_train_local,y_train_local,prototype_map_r = read_data_return_tensor(read_data_path,round_id=update,client_id=i,
                                                                            sampling_strategy=sampling_strategy_name,
                                                                            prototype_map=client_prototype_map[i])
            client_prototype_map[i] = prototype_map_r
            # 训练本地模型并获取损失值
            losses, coeff, norm_grad, data_len = train_model_FedNova_local(clients_models[i],
                                                               X_train_local,
                                                               y_train_local,
                                                               num_epochs=num_epochs)
            save_loss(loss_list=losses,client_id=i,round_id=update,save_loss_path=save_loss_path)

            local_metrics = test(copy.deepcopy(clients_models[i]),X_test_tensor,y_test_tensor)
            local_params_list.append((copy.deepcopy(clients_models[i]),coeff, norm_grad, data_len))
            save_metrics(title=f"client_{i}_metrics", rounds=update, metrics=local_metrics,save_folder = save_metrics_path)

        aggregated_params = aggregate_fednova(local_params_list,gm = copy.deepcopy(global_model))
        global_model.load_state_dict(aggregated_params)

        # 在每轮结束后发送全局模型参数给客户端
        gm = copy.deepcopy(global_model)
        save_model(copy.deepcopy(gm),update,save_model_path)
        for client_model in clients_models:
            client_model.load_state_dict(gm.state_dict())

        me = test(gm,X_test_tensor,y_test_tensor)
        save_metrics(title="global_back", rounds=update, metrics=me,save_folder = save_metrics_path)
        print("gme acc:" ,me['acc'])

# import time
# sampling_strategy_name_list = ['no','RandomOverSampler','RandomUnderSampler','CondensedNearestNeighbour','Triplets','IncrementalSampling']
# for id in range(6):
#     # runFedNova(sampling_id=id,settingName='CovA_Abrupt_BtoVH')
#     runFedNova(samplingName=sampling_strategy_name_list[id],settingName='CovA_Abrupt_VHtoB')
#     time.sleep(20)

#     runFedNova(sampling_id=id,settingName='Abrupt_Frequency10_DirF')
# print(pp)
import time
# sampling_strategy_name_list = ['no','RandomOverSampler','RandomUnderSampler','CondensedNearestNeighbour','Triplets','IncrementalSampling']
# dataset_list = ['CovA_Abrupt_BtoVH','CovA_Abrupt_BtoVH','CovA_Abrupt_BtoVH','CovA_Abrupt_BtoVH','covA_Abrupt_Asyn_step1','covA_Incremental_10_dirF','covA_Incremental_10_dirF','covA_Incremental_10_dirF','covA_Incremental_10_dirF','covA_Abrupt_Asyn_step1','covA_Abrupt_Asyn_step3','covA_Abrupt_Asyn_step3','covA_Abrupt_Asyn_step3','covA_Abrupt_Asyn_step3']
# ['covA_Abrupt_Asyn_step5','covA_Abrupt_Asyn_step5','covA_Abrupt_Asyn_step5','covA_Abrupt_Asyn_step5','Rc1c2_Frequency_5VH','Rc1c2_Frequency_10VH','Rc1c2_Frequency_15VH','Rc1c2_Asyn_step3','Rc1c2_Asyn_step5','Rc1c2_Asyn_step7','Rc1c2_Frequency_5VH','Rc1c2_Frequency_10VH','Rc1c2_Frequency_15VH','Rc1c2_Asyn_step3','Rc1c2_Asyn_step5','Rc1c2_Asyn_step7']
# sampling_strategy_name_list = ['no']
# dataset_list = ['covA_Incremental_20_dirF','covA_Incremental_20_dirF','covA_Incremental_20_dirF','covA_Incremental_20_dirF']
# sampling_strategy_name_list = ['RandomOverSampler','RandomUnderSampler','CondensedNearestNeighbour','Triplets','IncrementalSampling']
# dataset_list = ['covA_Incremental_50_dirF','covA_Incremental_50_dirF','covA_Incremental_50_dirF','covA_Incremental_50_dirF']
# sampling_strategy_name_list = ['no']
# dataset_list =['CovA_Abrupt_BtoH','CovA_Abrupt_BtoM','CovA_Abrupt_BtoL','CovA_Abrupt_BtoH','CovA_Abrupt_BtoM','CovA_Abrupt_BtoL','CovA_Abrupt_BtoH','CovA_Abrupt_BtoM','CovA_Abrupt_BtoL','CovA_Abrupt_BtoH','CovA_Abrupt_BtoM','CovA_Abrupt_BtoL','Cov1B_SeverityVH','Cov9B_SeverityVH','CovA_SeverityVH','CovH_SeverityVH','Cov1B_SeverityVH','Cov9B_SeverityVH','CovA_SeverityVH','CovH_SeverityVH','Cov1B_SeverityVH','Cov9B_SeverityVH','CovA_SeverityVH','CovH_SeverityVH','Cov1B_SeverityVH','Cov9B_SeverityVH','CovA_SeverityVH','CovH_SeverityVH','Cov1B_SeverityVH','Cov9B_SeverityVH','CovA_SeverityVH','CovH_SeverityVH','CovA_Abrupt_BtoH','CovA_Abrupt_BtoM','CovA_Abrupt_BtoL']
# ['covA_Abrupt_Asyn_step5','covA_Abrupt_Asyn_step5','covA_Abrupt_Asyn_step5','covA_Abrupt_Asyn_step5','Rc1c2_Frequency_5VH','Rc1c2_Frequency_10VH','Rc1c2_Frequency_15VH','Rc1c2_Asyn_step3','Rc1c2_Asyn_step5','Rc1c2_Asyn_step7','Rc1c2_Frequency_5VH','Rc1c2_Frequency_10VH','Rc1c2_Frequency_15VH','Rc1c2_Asyn_step3','Rc1c2_Asyn_step5','Rc1c2_Asyn_step7']
# for j ,settingname in enumerate(dataset_list):
#     for i,saplingname in enumerate(sampling_strategy_name_list):
#         runFedNova(samplingName=saplingname,settingName=settingname)
#         time.sleep(20)

```

## 2、Electricity数据集

```python
# 上诉代码没有共享模型
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree
from numpy.linalg import norm
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
from sklearn.metrics import f1_score,recall_score, accuracy_score, roc_auc_score,cohen_kappa_score,confusion_matrix,precision_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
import random
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,recall_score, accuracy_score, roc_auc_score,cohen_kappa_score,confusion_matrix,precision_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import copy
import os
import random
from sklearn.cluster import KMeans
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS, MeanShift ,SpectralClustering
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
import pandas as pd
from sklearn.utils import shuffle

import os
import torch
import hdbscan
import warnings
# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class IncrementalSampling(object):
    def __init__(self,prototype_nums = 30,cluster_strategy = 'kmeans'):
        self.incremental_data_map = {}
        self.data  = np.array([])
        self.label = np.array([])
        self.k_value = prototype_nums
        # 2024 06 07
        self.cluster_strategy = cluster_strategy
    def split_current_data_by_class(self):
        """把当前轮的数据按类别划分"""
        # 获取唯一的类别标签
        current_unique_labels = np.unique(self.label)
        # 按类别分割当前轮获得的数据
        data_by_class = {}  # map {"label":[the data belong to the label]}
        for current_data_label in current_unique_labels:
            indices = np.where(self.label == current_data_label)[0]
            data_by_class[current_data_label] = self.data[indices]
        return data_by_class, current_unique_labels
    def data_combined(self):
        """把当前轮的数据，和类map中的数据进行合并"""
        current_data_map_by_class, current_unique_labels = self.split_current_data_by_class()
        # 考虑到有新类出现的情况
        # 如果current_unique_labels有新label,直接扩充incremental_data_map
        # 如果label 是incremental_data_map中已经有的，扩张incremental_data_map对应label中data的长度
        for new_data_label in current_unique_labels:
            if new_data_label in self.incremental_data_map:
                # 增量map中已经有这个标签的数据了，那就扩充这个数据
                self.incremental_data_map[new_data_label] = np.concatenate(
                    (self.incremental_data_map[new_data_label], current_data_map_by_class[new_data_label])
                )
            else:
                # 如果增量map中没有这个标签的的数据，就扩充增量map
                self.incremental_data_map[new_data_label] = current_data_map_by_class[new_data_label]
    def cluster_data(self,data, num_clusters):
        # ['kmeans','spectral','kmeans++','kmedoids','OPTICS','meanshift','gmm','hdbscan']
        # kmeans = KMeans(n_clusters=num_clusters,n_init='auto')
        cluster_centers = np.ndarray([])
        if self.cluster_strategy == 'kmeans':
            print("using  kmeans")
            with warnings.catch_warnings():
            # KMeans(n_clusters=num_clusters,n_init='auto')
                kmeans = KMeans(n_clusters=num_clusters,n_init='auto')
                # kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(data)
                cluster_centers = kmeans.cluster_centers_
        elif self.cluster_strategy == 'spectral':
            print("using spectual")
            with warnings.catch_warnings():
                # 报错，小类样本太少，无法聚类
                print('cluster : spectral',{num_clusters})
                spectral = SpectralClustering(n_clusters=num_clusters, random_state=0, affinity='nearest_neighbors').fit(data)
                labels_spectral = spectral.labels_
                cluster_centers = np.array([data[labels_spectral == i].mean(axis=0) for i in range(num_clusters)])
                print(len(labels_spectral))
                print(len(cluster_centers))
        elif self.cluster_strategy.lower() == 'hdbscan':
            print("hdbscan")
            with warnings.catch_warnings():
                min_cluster_size = 2
                hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(data)
                labels_hdbscan = hdb.labels_
                cluster_centers = np.array([data[labels_hdbscan == i].mean(axis=0) for i in range(len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0))])
        elif self.cluster_strategy.lower() == 'kmeans++' or self.cluster_strategy.lower() == 'kmeansplusplus' :
            print("using kmeans++")
            with warnings.catch_warnings():# n_init='auto'
                # kmeans_plus_plus = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0).fit(data)
                kmeans_plus_plus = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0,n_init='auto').fit(data)
                cluster_centers = kmeans_plus_plus.cluster_centers_
        elif self.cluster_strategy.lower() =='kmedoids':
            print("using medorid")
            with warnings.catch_warnings():
                # 有几个空簇
                print('cluster : spectral',{num_clusters})
                kmedoids = KMedoids(n_clusters=num_clusters-10, random_state=0).fit(data)
                labelxx = kmedoids.labels_
                cluster_centers = kmedoids.cluster_centers_
                print(len(cluster_centers))
        elif self.cluster_strategy.upper() =='OPTICS':
            print("using optics")
            with warnings.catch_warnings():
                min_samples = 2
                optics = OPTICS(min_samples=min_samples).fit(data)
                labels_optics = optics.labels_
                cluster_centers = np.array([data[labels_optics == i].mean(axis=0) for i in range(len(set(labels_optics)) - (1 if -1 in labels_optics else 0))])
                print(len(cluster_centers))
        elif self.cluster_strategy.lower() =='meanshift':
            print("using meanshift")
            with warnings.catch_warnings():
                bandwidth = 0.1
                mean_shift = MeanShift(bandwidth=bandwidth).fit(data)
                cluster_centers = mean_shift.cluster_centers_
        elif self.cluster_strategy.lower() =='gmm':
            print("using gmm")
            with warnings.catch_warnings():
                gmm = GaussianMixture(n_components=num_clusters, random_state=0).fit(data)
                cluster_centers = gmm.means_  # 高斯混合模型的质心是每个成分的均值
        return  cluster_centers
    def reCludter(self):
        # incremental_data_map的大小进行压缩self.reCluster_instance_nums
        new_cluster = {}
        for label, data in self.incremental_data_map.items():
            if len(data) <= self.k_value:
                # 全部保存
                new_cluster[label] = data
            else:
                sampled_nums = self.k_value
                clusters = self.cluster_data(data = data, num_clusters=sampled_nums)
                new_cluster[label] = clusters
        self.incremental_data_map = new_cluster
    def compute_sampling_nums(self,combinde_map):
        # initialize
        min_length = float('inf')  # 初始化最小长度为正无穷大
        max_length = 0  # 初始化最大长度为0
        for label, data in combinde_map.items():
            data_length = len(data)
            if data_length < min_length:
                min_length = data_length
            if data_length > max_length:
                max_length = data_length
        return min_length, max_length
    def data_sampling(self,sampling_nums):
        resampling_data = []
        resampling_label = []
        for label, data in self.incremental_data_map.items():
            if len(data) > sampling_nums:
                # 那就是下采样了,不放回采样
                sampled_indices = random.sample(range(len(data)), sampling_nums)
                sampled_data = [data[i] for i in sampled_indices]
                resampling_data.extend(sampled_data)
                resampling_label.extend([label] * sampling_nums)
            elif len(data) == sampling_nums:
                # 直接复制
                resampling_data.extend(data)
                resampling_label.extend([label] * sampling_nums)
            else:
                # 上采样,保存原样本
                resampling_data.extend(data)
                # 随机有放回的找差额部分
                sampled_data = random.choices(data, k=(sampling_nums-len(data)))
                resampling_data.extend(sampled_data)
                resampling_label.extend([label] * sampling_nums)
        # 洗牌
        combined_data = list(zip(resampling_data, resampling_label))
        random.shuffle(combined_data)
        resampling_data, resampling_label = zip(*combined_data)
        return resampling_data,resampling_label
    def fit(self,incremental_prototype_map,data,label,sampling_strategy = "OverSampling"):
        self.incremental_data_map = incremental_prototype_map
        self.data = data
        self.label = label
        self.data_combined()
        resampling_data = []
        resampling_label = []
        min_length, max_length =self.compute_sampling_nums(self.incremental_data_map)
        if sampling_strategy.lower() == "oversampling":
            resampling_data,resampling_label = self.data_sampling(max_length)
        elif sampling_strategy.lower() == "downsampling":
            resampling_data,resampling_label = self.data_sampling(min_length)
        else:
            print("No sampling measures have been taken")
        self.reCludter()
        return resampling_data,resampling_label,self.incremental_data_map
# ['kmeans','spectral','kmeans++','kmedoids','OPTICS','meanshift','gmm']
class Incremental_sampling2(object):
    def __init__(self,save_prototype_nums = 30):
        self.save_prototype_nums = save_prototype_nums
        self.incremental_prototypes = {}
        self.data  = np.array([])
        self.label = np.array([])
        self.combined_map = {}
    def split_current_data_by_class(self):
        """把当前轮的数据按类别划分"""
        # 获取唯一的类别标签
        current_unique_labels = np.unique(self.label)
        # 按类别分割当前轮获得的数据
        data_by_class = {}  # map {"label":[the data belong to the label]}
        for current_data_label in current_unique_labels:
            indices = np.where(self.label == current_data_label)[0]
            data_by_class[current_data_label] = self.data[indices]
        return data_by_class, current_unique_labels
    def data_combined(self):
        """
        上一轮获得的数据原型和这一轮的新数据进行合并
        1.首先，按不同类别把数据和原型进行分开
        2.判断是不是新出现的类别的数据
            2.1、新类别数据,在原型map中直接扩展一个新类的map,{'新类':新类的数据}
            2.2、原先类别的数据,在对应类的数据上进行扩展，{'已有类':已有数据+新数据}
        """
        current_data_map_by_class, current_unique_labels = self.split_current_data_by_class()
        for new_data_label in current_unique_labels:
            if new_data_label in self.incremental_prototypes:
                # 增量map中已经有这个标签的数据了，那就扩充这个数据
                self.incremental_prototypes[new_data_label] = np.concatenate(
                    (self.incremental_prototypes[new_data_label], current_data_map_by_class[new_data_label])
                )
            else:
                # 如果增量map中没有这个标签的的数据，就扩充增量map
                self.incremental_prototypes[new_data_label] = current_data_map_by_class[new_data_label]
        self.combined_map = self.incremental_prototypes
    def cut_down_nearest_data_eu(self):
        """
        把每个类对应的原型中减去超出save_prototype_nums的样本
        判断规则：
            从离得最近的开始删除
        """
        for label in self.incremental_prototypes:
            data = self.incremental_prototypes[label]
            if len(data) > self.save_prototype_nums:
                # 计算成对距离
                pairwise_distances = squareform(pdist(data, 'euclidean'))
                np.fill_diagonal(pairwise_distances, np.inf)  # 将自身距离设置为无穷大，忽略自身距离

                while len(data) > self.save_prototype_nums:
                    # 找到距离最小的一对
                    min_dist_indices = np.unravel_index(np.argmin(pairwise_distances), pairwise_distances.shape)
                    # 保留一个样本，删除另一个
                    data = np.delete(data, min_dist_indices[1], axis=0)
                    # 从距离矩阵中删除对应的行和列
                    pairwise_distances = np.delete(pairwise_distances, min_dist_indices[1], axis=0)
                    pairwise_distances = np.delete(pairwise_distances, min_dist_indices[1], axis=1)

                # 使用简化后的数据更新原型
                self.incremental_prototypes[label] = data
    def cut_down_nearest_data_kdtree(self):
        """
        把每个类对应的原型中减去超出save_prototype_nums的样本
        判断规则：
            从离得最近的开始删除
        """
        for label in self.incremental_prototypes:
            # self.incremental_prototypes 在这之前已经和新数据合并了
            data = self.incremental_prototypes[label]
            if len(data) > self.save_prototype_nums:
                while len(data) > self.save_prototype_nums:
                    # 建立 KD 树
                    kdtree = KDTree(data)
                    # 查询每个点的最近邻
                    distances, indices = kdtree.query(data, k=2)  # k=2 因为第一个最近邻是点本身

                    # 找到最近的两个点
                    min_dist_idx = np.argmin(distances[:, 1])  # distances[:, 1] 是每个点的最近邻距离
                    nearest_idx = indices[min_dist_idx, 1]

                    # 删除其中一个点
                    data = np.delete(data, nearest_idx, axis=0)

                # 使用简化后的数据更新原型
                self.incremental_prototypes[label] = data
    def cut_down_nearest_data_nn(self):
        """
        把每个类对应的原型中减去超出save_prototype_nums的样本
        判断规则：
            使用批量删除
        """
        for label in self.incremental_prototypes:
            data = self.incremental_prototypes[label]
            if len(data) > self.save_prototype_nums:
                while len(data) > self.save_prototype_nums:
                    # 建立 NearestNeighbors 模型
                    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(data)
                    distances, indices = nbrs.kneighbors(data)

                    # 找到距离最小的一对
                    min_dist_idx = np.argmin(distances[:, 1])
                    nearest_idx = indices[min_dist_idx, 1]

                    # 批量删除，尽量减少删除操作次数
                    delete_indices = [min_dist_idx, nearest_idx]
                    data = np.delete(data, delete_indices, axis=0)

                # 使用简化后的数据更新原型
                self.incremental_prototypes[label] = data
    def cut_down_nearest_data(self):
        """
        把每个类对应的原型中减去超出save_prototype_nums的样本
        判断规则：
            保留距离最远的样本
        """
        for label in self.incremental_prototypes:
            data = self.incremental_prototypes[label]
            if len(data) > self.save_prototype_nums:
                # 计算成对距离矩阵
                pairwise_distances = squareform(pdist(data, 'euclidean'))

                # 对距离矩阵进行排序，获取距离最远的样本索引
                farthest_indices = np.argsort(-pairwise_distances.sum(axis=1))

                # 保留距离最远的前save_prototype_nums个样本
                keep_indices = farthest_indices[:self.save_prototype_nums]
                data = data[keep_indices]

                # 使用简化后的数据更新原型
                self.incremental_prototypes[label] = data
    def compute_sampling_nums(self,combined_map):
        # initialize
        min_length = float('inf')  # 初始化最小长度为正无穷大
        max_length = 0  # 初始化最大长度为0
        for label, data in combined_map.items():
            data_length = len(data)
            if data_length < min_length:
                min_length = data_length
            if data_length > max_length:
                max_length = data_length
        return min_length, max_length
    def triplet_sampling(self, num_cluster, n_neighbors=5, randomOr=True, len_lim=True):
        """Triplets 数据采样"""
        gen_x = []
        gen_y = []
        for label in self.incremental_prototypes:
            data_min = self.incremental_prototypes[label]
            if len(data_min) < num_cluster:
                size = num_cluster - len(data_min)
                weight = np.ones(len(data_min))
                # 收集多数类样本
                data_maj = np.vstack([self.incremental_prototypes[l] for l in self.incremental_prototypes if l != label])
                gen_x_c, gen_y_c = self._sample_one(data_min, data_maj, label, size, weight, n_neighbors, randomOr, len_lim)
                gen_x += gen_x_c
                gen_y += gen_y_c
        resampling_data = []
        resampling_label = []
        for label in self.combined_map: # incremental_prototypes样本太少了，只是为了生成样本用的
            data = self.incremental_prototypes[label]
            resampling_data.append(data)
            resampling_label.extend([label] * len(data))
        resampling_data = np.vstack(resampling_data)
        resampling_label = np.array(resampling_label)
        if len(gen_x) > 0:
            gen_x = np.vstack(gen_x)
            gen_y = np.array(gen_y)
            resampling_data = np.concatenate((resampling_data, gen_x), axis=0)
            resampling_label = np.concatenate((resampling_label, gen_y), axis=0)
        # 洗牌
        combined_data = list(zip(resampling_data, resampling_label))
        random.shuffle(combined_data)
        resampling_data, resampling_label = zip(*combined_data)
        return resampling_data,resampling_label
    def _sample_one(self, data_min, data_maj, label, size, weight, n_neighbors, randomOr, len_lim):
        gen_x = []
        gen_y = []
        if size == 0: return gen_x, gen_y

        min_idxs = np.arange(len(data_min))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data_maj)
        _, indices = nbrs.kneighbors(data_min)

        for j in np.random.choice(len(min_idxs), size, p=weight / weight.sum()):
            tp1 = data_min[min_idxs[j]]
            tp2 = data_maj[indices[j][:5]].mean(axis=0)
            tp3_ord = np.random.randint(n_neighbors)
            tp3 = data_maj[indices[j][tp3_ord]]
            if (tp2 == tp1).all():
                gen_x.append(tp1)
                gen_y.append(label)
                continue

            offset = tp3 - tp2
            offset_norm = norm(offset)
            if offset_norm == 0:
                continue

            tp1_tp2_norm = norm(tp1 - tp2)
            if tp1_tp2_norm == 0:
                continue

            if len_lim: offset = offset * min(1, tp1_tp2_norm / offset_norm)
            coef = np.random.rand() if randomOr else 1.0
            new_x = tp1 + coef * offset
            gen_x.append(new_x)
            gen_y.append(label)

        return gen_x, gen_y

    def random_sampling(self,num_cluster,sampling_strategy):
        """
        这里需要对数据采样
        首先遍历self.incremental_prototypes ,每个类
        以及每个类的数据的长度
        然后比较每个类的数据的长度和num_cluster之间的差距
        差额部分使用triplets的核心算法对齐进行生成样本
        """
        resampling_data = []
        resampling_label = []
        if sampling_strategy == "ros-p":
            # 随机上采样，差额数据从原型数据中随机复制
            for label, data in self.combined_map.items():
                if len(data) < num_cluster:  # 需要上采样的数据的条件
                    prototype_data = self.incremental_prototypes[label]
                    sampling_nums = num_cluster - len(data)

                    if len(prototype_data) < sampling_nums:
                        sampled_data = []
                        while len(sampled_data) < sampling_nums:
                            # 原型中样本还没有sampling_nums多时，直接复制原型中的数据
                            needed = sampling_nums - len(sampled_data)
                            sampled_indices = random.sample(range(len(prototype_data)), min(needed, len(prototype_data)))
                            sampled_data.extend([prototype_data[i] for i in sampled_indices])
                    else:
                        sampled_indices = random.sample(range(len(prototype_data)), sampling_nums)
                        sampled_data = [prototype_data[i] for i in sampled_indices]

                    resampling_data.extend(data)
                    resampling_label.extend([label] * len(data))
                    resampling_data.extend(sampled_data)
                    resampling_label.extend([label] * sampling_nums)
                elif len(data) == num_cluster:
                    resampling_data.extend(data)
                    resampling_label.extend([label] * len(data))
        elif sampling_strategy == "ros-h" :
            # 随机上采样，从混合数据中采样差额数据
            for label, data in self.combined_map.items():
                data_len = len(data)
                if data_len < num_cluster:
                    # 那就是下采样了,不放回采样
                    sampling_nums = num_cluster-data_len
                    if data_len < sampling_nums:
                        sampled_data = []
                        while len(sampled_data) < sampling_nums:
                            # 原型中样本还没有sampling_nums多时，直接复制原型中的数据
                            needed = sampling_nums - len(sampled_data)
                            sampled_indices = random.sample(range(data_len), min(needed, data_len))
                            sampled_data.extend([data[i] for i in sampled_indices])
                    else:
                        sampled_indices = random.sample(range(data_len), sampling_nums)
                        sampled_data = [data[i] for i in sampled_indices]
                    resampling_data.extend(data)
                    resampling_label.extend([label] * len(data))
                    resampling_data.extend(sampled_data)
                    resampling_label.extend([label] * sampling_nums)
                elif len(data) == num_cluster:
                    # 直接复制
                    resampling_data.extend(data)
                    resampling_label.extend([label] * len(data))
        # 洗牌
        combined_data = list(zip(resampling_data, resampling_label))
        random.shuffle(combined_data)
        resampling_data, resampling_label = zip(*combined_data)
        return resampling_data,resampling_label
    def fit(self,new_data,new_data_label,last_round_prototype,sampling_strategy='tpl'):
        self.incremental_prototypes = last_round_prototype
        self.data = new_data
        self.label = new_data_label
        self.data_combined()
        resampling_data = []
        resampling_label = []
        min_length, max_length =self.compute_sampling_nums(self.incremental_prototypes)
        if sampling_strategy.lower() == "tpl":
            resampling_data,resampling_label = self.triplet_sampling(max_length)
        elif sampling_strategy.lower() == "ros-p":
            # rest data copied from prototype
            resampling_data,resampling_label = self.random_sampling(max_length,sampling_strategy = "ros-p")
            print('use ros p')
        elif sampling_strategy.lower() == "ros-h":
            # rest data copied from hybrid data(combined data)
            resampling_data,resampling_label = self.random_sampling(max_length,sampling_strategy = "ros-h")
            print('use ros h')
        return resampling_data,resampling_label,self.incremental_prototypes
class Triplets(object):
    def __init__(self, n_neighbors=5, random=True, len_lim=True, **kwargs):
        self.n_neighbors = n_neighbors
        self.random = random
        self.len_lim = len_lim

    def fit_resample(self, x, y):
        strategy = self._sample_strategy(y)
        self.n_neighbors = max(self.n_neighbors, self.counts.max() // self.counts.min())

        gen_x = []
        gen_y = []
        # 这里的代码平衡状态会报错
        for c, size in enumerate(strategy):
            if size == 0: continue
            weight = self._weights(x, y, c)
            gen_x_c, gen_y_c = self._sample_one(x, y, c, size, weight)
            gen_x += gen_x_c
            gen_y += gen_y_c

        # 为了这个方法在平衡状态下不报错，我们特地在这里加了这段代码
        # To prevent errors in this method when in a balanced state, we intentionally added this code block
        if len(gen_x)==0:
            return x,y
        gen_x = np.vstack(gen_x)
        gen_y = np.array(gen_y)
        return np.concatenate((x, gen_x), axis=0), np.concatenate((y, gen_y), axis=0)

    def _sample_strategy(self, y):
        _, self.counts = np.unique(y, return_counts=True)
        return max(self.counts) - self.counts

    def _weights(self, x, y, c):
        return np.ones(self.counts[c])

    def _sample_one(self, x, y, c, size, weight):
        gen_x = []
        gen_y = []
        if size == 0: return gen_x, gen_y

        # get the indices of minority and majority instances
        min_idxs = np.where(y == c)[0]
        maj_idxs = np.where(y != c)[0]

        # find nearest majority neighbors for each minority instance
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(x[maj_idxs])
        _, indices = nbrs.kneighbors(x[min_idxs])

        # generate synthetic data
        for j in np.random.choice(len(min_idxs), size, p=weight / weight.sum()):
            tp1 = x[min_idxs[j]]
            tp2 = x[maj_idxs[indices[j][:5]]].mean(axis=0)
            # tp3_ord = np.random.randint(1, self.n_neighbors)
            tp3_ord = np.random.randint(self.n_neighbors)
            tp3 = x[maj_idxs[indices[j][tp3_ord]]]
            if (tp2 == tp1).all():
                gen_x.append(tp1)
                gen_y.append(c)
                continue

            offset = tp3 - tp2
            if self.len_lim: offset = offset * min(1, norm(tp1 - tp2) / norm(offset))
            coef = np.random.rand() if self.random is True else 1.0
            new_x = tp1 + coef * offset
            gen_x.append(new_x)
            gen_y.append(c)
        return gen_x, gen_y

def train_model_FedAvg_local(input_model,X_train_tensor, y_train_tensor, num_epochs):
    losses =[]
    model = input_model# copy.deepcopy(input_model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        losses.append(loss.item())  # 将每次训练的损失值添加到列表中

    return losses
def train_model_FedProx_local(input_model, X_train_tensor, y_train_tensor, num_epochs):
    mu = 0.1
    # because last round trained global model replaced local model,
    # that in this round the first local model is last round global model
    losses =[]
    model = input_model# copy.deepcopy(input_model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    global_weights = copy.deepcopy(list(model.parameters()))
    # not use deepcopy ,because model as parameter transport in this ,update model also update model
    # current_local_model = cmodel
    # model.train()
    # for epoch in range(num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        # FedProx
        prox_term = 0.0
        for p_i, param in enumerate(model.parameters()):
                prox_term += (mu / 2) * torch.norm((param - global_weights[p_i])) ** 2
        loss += prox_term

        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        losses.append(loss.item())  # 将每次训练的损失值添加到列表中
    #     total_loss = 0.0
    #     for step,(x,y) in enumerate(zip(X_train_tensor,y_train_tensor)):
    #         # current_local_model.train()
    #         output = model(x) # current_local_model(x)
    #         loss = criterion(output, y)
    #         total_loss+=loss.item()
    #         optimizer.zero_grad()
    #
    #         # fedprox
    #         prox_term = 0.0
    #         for p_i, param in enumerate(model.parameters()):
    #             prox_term += (mu / 2) * torch.norm((param - global_weights[p_i])) ** 2
    #         loss += prox_term
    #         loss.backward()
    #         optimizer.step()
    #     losses.append(total_loss)
    #     if epoch % 10 == 0:
    #         print(f"Epoch [{epoch+1}/{num_epochs}], Epoch Total Loss: {total_loss:.4f}")
    return losses

def train_model_FedNova_local(input_model,X_train_tensor, y_train_tensor, num_epochs):
    # because last round trained global model replaced local model,
    # that in this round the first local model is last round global model
    losses =[]
    model = input_model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    global_weights = copy.deepcopy(input_model.state_dict())
    tau = 0
    rho = 0.9
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        tau +=len(y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        losses.append(loss.item())
    # for epoch in range(num_epochs):
    #     total_loss = 0.0
    #     for step,(x,y) in enumerate(zip(X_train_tensor,y_train_tensor)):
    #         # current_local_model.train()
    #         # model.train()
    #         output = model(x) # current_local_model(x)
    #         loss = criterion(output, y)
    #         total_loss+=loss.item()
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         tau +=1
    #     losses.append(total_loss)
    #     if epoch % 10 == 0:
    #         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
    coeff = (tau - rho * (1 - pow(rho, tau)) / (1 - rho)) / (1 - rho)
    state_dict = model.state_dict()
    norm_grad = copy.deepcopy(global_weights)
    for key in norm_grad:
        norm_grad[key] = torch.div(global_weights[key] - state_dict[key], coeff)

    return losses, coeff, norm_grad,len(X_train_tensor)
# 定义模型参数共享函数
def share_params(model):
    params = model.state_dict()
    # return {k: v.clone().detach().requires_grad_(True) for k, v in params.items()}
    return {k: v.clone().detach().requires_grad_(False) for k, v in params.items()}

# # 定义模型参数聚合函数
def aggregate_params(params_list):
    aggregated_params = {}
    for key in params_list[0].keys():
        # 将参数转换为张量进行处理
        params_tensors = [params[key].clone().detach().float() for params in params_list]
        # 聚合参数
        aggregated_params[key] = sum(params_tensors) / len(params_tensors)
    return aggregated_params
def test(global_model,X_test,y_test):
    # 在全局模型上进行测试
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        outputs = global_model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        # 计算度量
        predictions = predicted.numpy() # 将张量转换为 NumPy 数组并去除零维数组
        true_labels = y_test_tensor.numpy()  # 将张量转换为 NumPy 数组并去除零维数组
        precision = precision_score(true_labels,predictions,zero_division=0.0,average=None)
        precision_micro = precision_score(true_labels,predictions,zero_division=0.0,average='micro')
        precision_macro = precision_score(true_labels,predictions,zero_division=0.0,average='macro')
        # recall
        recalls = recall_score(true_labels,predictions,zero_division=0.0,average=None)
        recalls_micro =recall_score(true_labels,predictions,zero_division=0.0,average='micro')
        recalls_macro =recall_score(true_labels,predictions,zero_division=0.0,average='macro')
        f1_scores = f1_score(true_labels, predictions, average=None)
        acc = accuracy_score(true_labels, predictions)
        kappa = cohen_kappa_score(true_labels,predictions)
        conf_matrix = confusion_matrix(true_labels,predictions)
        # 计算所有类别乘积的几何平均值作为 G-mean
        g_mean_all= np.power(np.prod(recalls), 1 / len(recalls))
        # AUC
        lb = LabelBinarizer()
        lb.fit(true_labels)
        true_labels_bin = lb.transform(true_labels)
        predictions_bin = lb.transform(predictions)
        auc = roc_auc_score(true_labels_bin, predictions_bin, average='weighted', multi_class='ovr')
        metrics = {
            'recall':recalls,
            'recall_micro':recalls_micro,
            'recall_macro':recalls_macro,
            'precision':precision,
            'precision_micro':precision_micro,
            'precision_macro':precision_macro,
            'f1_score':f1_scores,
            'g_mean':g_mean_all,
            'acc':acc,
            'auc':auc,
            'kappa':kappa,
            'confusion_matrix':conf_matrix
        }
        return metrics

def save_loss(loss_list,client_id,round_id,save_loss_path):
    if not os.path.exists(save_loss_path):
        os.makedirs(save_loss_path)
    # 构建文件路径
    file_path = os.path.join(save_loss_path, f"client_{client_id}.csv")

    if os.path.exists(file_path):
        # 如果文件存在，加载现有的 CSV 文件为 DataFrame
        df = pd.read_csv(file_path)
    else:
        # 如果文件不存在，直接创建新的 DataFrame
        df = pd.DataFrame()

    # 将损失值添加到 DataFrame 中
    column_name = f'round_{round_id}'
    df[column_name] = loss_list

    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(file_path, index=False)
def save_model(global_model,round_id,save_model_path):
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    model_path = os.path.join(save_model_path,f'round_{round_id}_gm.pt')
    torch.save(global_model,model_path)

def save_metrics(title, rounds, metrics, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    file_name = f"{title}.csv"
    file_path = os.path.join(save_folder, file_name)
    # print(file_path)
    recalls = metrics['recall']
    class_nums = len(recalls)
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 如果文件存在，加载现有的 Excel 文件为 DataFrame
        df = pd.read_csv(file_path)
    else:
        # 如果文件不存在，直接创建新的 DataFrame
        columns = [
        'rounds', 'accuracy', 'auc', 'kappa', 'g_mean', 'recall_micro', 'precision_micro',
        'recall_macro', 'precision_macro'
        ]
        df = pd.DataFrame(columns=columns)
        for i in range(class_nums):  # 动态生成 f1-score 相关列名
            columns.append(f'f1_score_{i}')
            columns.append(f'recall_{i}')
            columns.append(f'precession_{i}')

    data = {
        'rounds': rounds,
        'accuracy': metrics['acc'],
        'auc': metrics['auc'],
        'kappa': metrics['kappa'],
        'g_mean':metrics['g_mean'],
        'recall_micro':metrics['recall_micro'],
        'precision_micro':metrics['precision_micro'],
        'recall_macro':metrics['recall_macro'],
        'precision_macro':metrics['precision_macro']
    }
    # 添加每个类别的 F1-score、G-mean 和 Recall 到 data 中
    for i in range(class_nums):  #类别数
        data[f'recall_{i}'] = metrics['recall'][i]
        data[f'precision_{i}'] = metrics['precision'][i]
        data[f'f1_score_{i}'] = metrics['f1_score'][i]
    # 创建新行并追加到 DataFrame
    new_row = pd.DataFrame(data, index=[0])
    df = pd.concat([df, new_row], ignore_index=True)

    # 将 DataFrame 保存为 Excel 文件
    df.to_csv(file_path, index=False)

from imblearn.over_sampling import SMOTE,RandomOverSampler,SMOTENC,SMOTEN,ADASYN,BorderlineSMOTE,KMeansSMOTE,SVMSMOTE
from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler,NearMiss,TomekLinks,EditedNearestNeighbours,RepeatedEditedNearestNeighbours,AllKNN,CondensedNearestNeighbour,OneSidedSelection,NeighbourhoodCleaningRule,InstanceHardnessThreshold
from imblearn.combine import SMOTEENN,SMOTETomek
# SMOTE,ROS,SMOTENC,SMOTEN,ADASYN,BorderlineSMOTE1,BorderlineSMOTE2,KMeansSMOTE,SVMSMOTE
# ClusterCentroids,RUS,NearMiss1,NearMiss2,NearMiss2,TomekLinks,ENN,RENN,AllKNN,CNN,OSS,NC,IHT
# SMOTEENN,SMOTETomek
def data_sampling(raw_X,raw_y,sampling_strategy):
    if sampling_strategy.upper() == 'NO' :
        return raw_X,raw_y
    # overSampling
    elif sampling_strategy.upper() == 'SMOTE': # overSampling
        """
            1.对于样本x ,按照欧氏距离找到离其距离最近的K个近邻样本
            2.确定采样比例，然后从K个近邻中选择x_n
            3.公式 x_new = x + rand(0,1)*(x_n-x)
        """
        smote = SMOTE( random_state=42)
        X_resampled, y_resampled = smote.fit_resample(raw_X,raw_y)
        return X_resampled, y_resampled
    elif sampling_strategy == "RandomOverSampler" or sampling_strategy.upper()=='ROS': # overSampling
        ros = RandomOverSampler(random_state=1)
        ros_data,ros_label = ros.fit_resample(raw_X,raw_y)
        return ros_data,ros_label
    elif sampling_strategy.upper() == 'SMOTENC':    # overSampling
        smotenc = SMOTENC(random_state=1,categorical_features=[0])
        smotenc_data,smotenc_label = smotenc.fit_resample(raw_X,raw_y)
        return smotenc_data,smotenc_label
    elif sampling_strategy.upper() == 'SMOTEN': # overSampling
        smoten = SMOTEN(random_state=1)
        smoten_data,smoten_label = smoten.fit_resample(raw_X,raw_y)
        return smoten_data,smoten_label
    elif sampling_strategy.upper() =='ADASYN':
        adasyn = ADASYN(random_state=1)
        adasyn_data,adasyn_label = adasyn.fit_resample(raw_X,raw_y)
        return adasyn_data,adasyn_label
    elif sampling_strategy == 'BorderlineSMOTE1' or sampling_strategy.upper()=='BSMOTE1':
        bsmote1 = BorderlineSMOTE(kind='borderline-1',random_state=1)
        bsmote1_data,bsmote1_label = bsmote1.fit_resample(raw_X,raw_y)
        return bsmote1_data,bsmote1_label
    elif sampling_strategy == 'BorderlineSMOTE2'or sampling_strategy.upper()=='BSMOTE2':
        bsmote2 = BorderlineSMOTE(kind='borderline-2',random_state=1)
        bsmote2_data,bsmote2_label = bsmote2.fit_resample(raw_X,raw_y)
        return bsmote2_data,bsmote2_label
    elif sampling_strategy == 'KMeansSMOTE' or sampling_strategy.upper() == 'KSMOTE':
        kmeanssmote = KMeansSMOTE(random_state=1)
        kmeanssmote_data,kmeanssmote_label = kmeanssmote.fit_resample(raw_X,raw_y)
        return kmeanssmote_data,kmeanssmote_label
    elif sampling_strategy == 'SVMSMOTE':
        svmsmote = SVMSMOTE(random_state=1)
        svmsmote_data,svmsmote_label = svmsmote.fit_resample(raw_X,raw_y)
        return svmsmote_data,svmsmote_label
    # downSampling
    elif sampling_strategy == 'ClusterCentroids': # down-sampling,generate
        clustercentroids = ClusterCentroids(random_state=1)
        clustercentroids_data,clustercentroids_label = clustercentroids.fit_resample(raw_X,raw_y)
        return clustercentroids_data,clustercentroids_label
    elif sampling_strategy=='RandomUnderSampler' or sampling_strategy.upper()=='RUS':
        rus = RandomUnderSampler(random_state=1)
        rus_data,rus_label = rus.fit_resample(raw_X,raw_y)
        return rus_data,rus_label
    elif sampling_strategy.upper() =='NEARMISS1':
        # 在k个少数类别样本中，选择出与他们-平均距离最近的多数类样本-进行保存
        nearmiss1 = NearMiss(version=1)
        nearmiss1_data,nearmiss1_label = nearmiss1.fit_resample(raw_X,raw_y)
        return nearmiss1_data,nearmiss1_label
    elif sampling_strategy.upper() =='NEARMISS2':
        # 选择K个距离最远的少数类别样本，然后根据这些样本选出的"平均距离最近"的样本进行保存
        nearmiss2 = NearMiss(version=2)
        nearmiss2_data,nearmiss2_label = nearmiss2.fit_resample(raw_X,raw_y)
        return nearmiss2_data,nearmiss2_label
    elif sampling_strategy.upper() =='NEARMISS3':
        # 1、对于每一个少数类别样本，保留其K个最近邻多数类样本；2、把到K个少数样本平均距离最大的多数类样本保存下来。
        nearmiss3 = NearMiss(version=3)
        nearmiss3_data,nearmiss3_label = nearmiss3.fit_resample(raw_X,raw_y)
        return nearmiss3_data,nearmiss3_label
    elif sampling_strategy == 'TomekLinks' or sampling_strategy.upper()=='TOMEK':
        # 它需要计算每个样本之间的距离，然后把互为最近邻且类别不同的一对样本拿出来，根据需求的选择将这一对数据进行剔除 or 把多数类样本剔除
        tomelink = TomekLinks(sampling_strategy='all')#sampling_strategy='all'表示全部删除，'auto'表示只删除多数类
        tomelink_data,tomelink_label = tomelink.fit_resample(raw_X,raw_y)
        return tomelink_data,tomelink_label
    elif sampling_strategy == 'EditedNearestNeighbours' or sampling_strategy.upper() == 'ENN':
        ENN = EditedNearestNeighbours()
        ENN_data,ENN_label = ENN.fit_resample(raw_X,raw_y)
        return ENN_data,ENN_label
    elif sampling_strategy == 'RepeatedEditedNearestNeighbours' or sampling_strategy.upper() == 'RENN':
        RENN = RepeatedEditedNearestNeighbours()
        RENN_data,RENN_label = RENN.fit_resample(raw_X,raw_y)
        return RENN_data,RENN_label
    elif sampling_strategy =='AllKNN':
        ## ENN的改进版本，和RepeatedEditedNearestNeighbours一样，会多次迭代ENN 算法，不同之处在于，他会每次增加KNN的K值
        allknn = AllKNN()
        allknn_data,allknn_label = allknn.fit_resample(raw_X,raw_y)
        return allknn_data,allknn_label
    elif sampling_strategy == 'CondensedNearestNeighbour'or sampling_strategy.upper() == 'CNN':
        ## 如果有样本无法和其他多数类样本聚类到一起，那么说明它极有可能是边界的样本，所以将这些样本加入到集合中
        CNN = CondensedNearestNeighbour(random_state=1)
        CNN_data,CNN_label = CNN.fit_resample(raw_X,raw_y)
        return CNN_data,CNN_label
    elif sampling_strategy == 'OneSidedSelection' or sampling_strategy.upper() == 'OSS':
        # OneSidedSelection = tomekLinks + CondensedNearestNeighbour,先使用自杀式的方式把大类数据中的其他值剔除，然后再使用CondensedNearestNeighbour的下采样
        OSS = OneSidedSelection(random_state=1)
        OSS_data,OSS_label = OSS.fit_resample(raw_X,raw_y)
        return OSS_data,OSS_label
    elif sampling_strategy == 'NeighbourhoodCleaningRule'or sampling_strategy.upper() == 'NC':
        # 若在大类的K-近邻中，少数类占多数，那就剔除这个多数类别的样本
        NC = NeighbourhoodCleaningRule()
        NC_data,NC_label = NC.fit_resample(raw_X,raw_y)
        return NC_data,NC_label
    elif sampling_strategy == 'InstanceHardnessThreshold' or sampling_strategy.upper() == 'IHT':
        # 默认算法是随机森林，通过分类算法给出样本阈值来剔除部分样本，（阈值较低的可以剔除）,慢
        IHT = InstanceHardnessThreshold(random_state=1)
        IHT_data,IHT_label = IHT.fit_resample(raw_X,raw_y)
        return IHT_data,IHT_label
    # hibird
    elif sampling_strategy.upper() =='SMOTEENN':
        se = SMOTEENN(random_state=1)
        se_data,se_label = se.fit_resample(raw_X,raw_y)
        return se_data,se_label
    elif sampling_strategy.upper() =='SMOTETOMEK':
        st = SMOTETomek(random_state=1)
        st_data,st_label = st.fit_resample(raw_X,raw_y)
        return st_data,st_label
    elif sampling_strategy == 'Triplets':
        print(" Triplets sampling")
        tpl = Triplets()
        tpl_data,tpl_label = tpl.fit_resample(raw_X,raw_y)
        return tpl_data,tpl_label
    else :
        print("skipped all the sampling strategy,but return the raw data and label")
        return raw_X,raw_y
# ['kmeans','spectral','kmeans++','kmedoids','OPTICS','meanshift','gmm']
def inremental_sampling(prototype_map,raw_X,raw_y,cluster_strategy):
    class_nums = len(np.unique(raw_y))
    print("class nums" ,class_nums)
    if cluster_strategy == 'ros-p' or cluster_strategy == 'ros-h' or cluster_strategy=='tpl':
        isap = Incremental_sampling2()
        resampling_data,resampling_label,prototype_map = isap.fit(new_data=raw_X,new_data_label=raw_y,last_round_prototype=prototype_map,sampling_strategy=cluster_strategy)
    else :
        isap = IncrementalSampling(cluster_strategy=cluster_strategy)
        resampling_data,resampling_label,prototype_map = isap.fit(incremental_prototype_map=prototype_map,data=raw_X,label=raw_y)
    return resampling_data,resampling_label,prototype_map
# ['kmeans','spectral','kmeans++','kmedoids','OPTICS','meanshift','gmm']
def read_data_return_tensor(dataset_path, round_id, client_id, sampling_strategy='no',prototype_map = {},cluster_strategy='kmeans'):
    folder_path = os.path.join(dataset_path, f'client_{client_id}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    open_file_path = os.path.join(folder_path, f'round_{round_id}.csv')
    data = pd.read_csv(open_file_path, header=0)
    # 排除标题行并打乱数据
    data_shuffled = shuffle(data)  # 排除标题行并打乱数据

    # 提取特征和目标变量
    raw_X = data_shuffled.iloc[:, :-1].values  # 特征
    raw_y = data_shuffled.iloc[:, -1].values   # 目标变量
    raw_y = raw_y.astype(float)
    # print(len(raw_X),type(raw_X))
    # print(len(raw_X),type(raw_X))
    if sampling_strategy == 'IncrementalSampling':
        print("using IncrementalSampling")
        resampling_X,resampling_y,prototype_map = inremental_sampling(prototype_map,raw_X,raw_y,cluster_strategy=cluster_strategy)
    else:
        resampling_X,resampling_y = data_sampling(raw_X,raw_y,sampling_strategy)
        resampling_y.astype(float)
    resampling_X = np.array(resampling_X)  # 将列表转换为单个NumPy数组
    resampling_y = np.array(resampling_y)  # 将列表转换为单个NumPy数组

    X_train_tensor = torch.tensor(resampling_X, dtype=torch.float32)  # 特征张量
    y_train_tensor = torch.tensor(resampling_y, dtype=torch.long)   # 标签张量
    return X_train_tensor,y_train_tensor,prototype_map
def read_test_data(test_data_path):
    data = pd.read_csv(test_data_path, header=0)
    # 排除标题行并打乱数据
    data_shuffled = shuffle(data)  # 排除标题行并打乱数据

    # 提取特征和目标变量
    X = data_shuffled.iloc[:, :-1].values.astype(float)  # 特征
    # print(X)
    y = data_shuffled.iloc[:, -1].values   # 目标变量
    y = y.astype(float)

    X_test_tensor = torch.tensor(X, dtype=torch.float32)  # 特征张量
    y_test_tensor = torch.tensor(y, dtype=torch.long)   # 标签张量
    return X_test_tensor,y_test_tensor
```

#### Run code

```python
def aggregate_fednova(local_params_list,gm):
    # (share_params(clients_models[i]),coeff, norm_grad, data_len) as input
    total_data_len = sum(data_len for _, _, _, data_len in local_params_list)
    global_model_state = gm.state_dict()
    nova_model_state = copy.deepcopy(global_model_state)
    # avg_loss = 0
    coeff = 0.0
    for clientID,(client_model,client_coeff,client_norm_grad,client_local_data_len) in enumerate(local_params_list):
        coeff = coeff + client_coeff*client_local_data_len/total_data_len
        for key in client_model.state_dict():
            if clientID == 0:
                nova_model_state[key] = client_norm_grad[key] * client_local_data_len/total_data_len
            else:
                nova_model_state[key] =nova_model_state[key]+ client_norm_grad[key] * client_local_data_len/total_data_len
        # avg_loss = avg_loss + cl
    for key in global_model_state:
        global_model_state[key] -= coeff*nova_model_state[key]

    return global_model_state
pp = []
def runFedNova(samplingName,settingName,cluster_strategy):
    sampling_strategy_name=samplingName
    num_clients = 10
    # 初始化全局模型和客户端模型
    input_size = 8
    hidden_size = 100
    output_size = 2
    global_model = MLP(input_size, hidden_size, output_size)
    client_prototype_map = [{} for _ in range(num_clients)]
    clients_models = [MLP(input_size, hidden_size, output_size) for _ in range(num_clients)]

    num_epochs = 400
    num_global_updates = 36
    # 'E:/FedStream/real_data_set/realdataset0427/elecNorm/Electricity_client_random/'
    base_path ='/root/FedStream/' # 'E:/FedStream/'
    dataset_name = 'elecNorm'
    setting_name = settingName # 'CovA_Abrupt_BtoVH'
    algorithm = 'Fednova_kvalue30'
    experiment_times = 'epoch200'
    save_loss_path = f'{base_path}/loss/{dataset_name}_Sampling/{algorithm}/{setting_name}/{sampling_strategy_name}_{cluster_strategy}fa_{experiment_times}'
    save_model_path = f'{base_path}/models/{dataset_name}_Sampling/{algorithm}/{setting_name}/{sampling_strategy_name}_{cluster_strategy}fa_{experiment_times}'
    save_metrics_path = f'{base_path}/metrics/{dataset_name}_Sampling/{algorithm}/{setting_name}/{sampling_strategy_name}_{cluster_strategy}fa_{experiment_times}'
    read_data_path = f'/root/FedStream/real_data_set/realdataset0427/{dataset_name}/{setting_name}/'
    test_path = '/root/FedStream/real_data_set/realdataset0427/elecNorm/test_dataset.csv'
    X_test_tensor, y_test_tensor = read_test_data(test_path)

    for update in range(num_global_updates):
        print(f"round{update}")
        local_params_list = []

        for i in range(num_clients):
            # 在每个客户端训练本地模型
            print(f"client{i}")
            X_train_local,y_train_local,prototype_map_r = read_data_return_tensor(read_data_path,round_id=update,client_id=i,sampling_strategy=sampling_strategy_name,prototype_map=client_prototype_map[i],cluster_strategy=cluster_strategy)
            client_prototype_map[i] = prototype_map_r
            # 训练本地模型并获取损失值
            losses, coeff, norm_grad, data_len = train_model_FedNova_local(clients_models[i],X_train_local,y_train_local,num_epochs=num_epochs)
            # y_train_tensor = torch.sub(y_train_tensor, 1)
            # losses, coeff, norm_grad, data_len = train_model_FedNova_local(clients_models[i],X_train_local,torch.sub(y_train_local, 1),num_epochs=num_epochs)
            # save_loss(loss_list=losses,client_id=i,round_id=update,save_loss_path=save_loss_path)

            local_metrics = test(copy.deepcopy(clients_models[i]),X_test_tensor,y_test_tensor)
            local_params_list.append((copy.deepcopy(clients_models[i]),coeff, norm_grad, data_len))
            save_metrics(title=f"client_{i}_metrics", rounds=update, metrics=local_metrics,save_folder = save_metrics_path)

        aggregated_params = aggregate_fednova(local_params_list,gm = copy.deepcopy(global_model))
        global_model.load_state_dict(aggregated_params)

        # 在每轮结束后发送全局模型参数给客户端
        gm = copy.deepcopy(global_model)
        save_model(copy.deepcopy(gm),update,save_model_path)
        for client_model in clients_models:
            client_model.load_state_dict(gm.state_dict())

        me = test(gm,X_test_tensor,y_test_tensor) # torch.sub(y_train_tensor, 1)
        # me = test(gm,X_test_tensor,torch.sub(y_test_tensor, 1))
        save_metrics(title="global_back", rounds=update, metrics=me,save_folder = save_metrics_path)
        print("gme acc:" ,me)

def train_model_FedScaFFold_local(input_model, X_train_tensor, y_train_tensor, num_epochs, c_global, c_local, lr=0.01):
    model = input_model
    global_weights = copy.deepcopy(input_model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    # 初始化控制变量差异
    if not c_local:
        c_local = [torch.zeros_like(param) for param in model.parameters()]

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()

        # 计算并加上控制变量差异
        c_diff = [c_g - c_l for c_g, c_l in zip(c_global, c_local)]
        for param, c_d in zip(model.parameters(), c_diff):
            param.grad += c_d.data

        optimizer.step()

    # 计算 y_delta（模型参数的变化量）
    y_delta = [param.data - global_weights[name].data for name, param in model.named_parameters()]

    # 更新本地控制变量
    coef = 1 / (num_epochs * lr)
    c_local = [c_l - c_g - coef * delta for c_l, c_g, delta in zip(c_local, c_global, y_delta)]

    return model.state_dict(), y_delta, c_local
# 定义服务器端聚合函数
def scaffold_aggregator(local_params):
    global_params = local_params[0][0]
    global_c = local_params[0][2]
    num_clients = len(local_params)

    # 初始化全局y_delta和c_delta
    avg_y_delta = [torch.zeros_like(param) for param in global_params.values()]
    avg_c_delta = [torch.zeros_like(c) for c in global_c]

    for params, y_delta, c_local in local_params:
        for i, delta in enumerate(y_delta):
            avg_y_delta[i] += delta / num_clients
        for i, c_delta in enumerate(c_local):
            avg_c_delta[i] += c_delta / num_clients

    # 更新全局模型参数
    for (name, param), delta in zip(global_params.items(), avg_y_delta):
        param.data += delta

    # 更新全局控制变量
    for i, delta in enumerate(avg_c_delta):
        global_c[i] += delta

    return global_params, global_c

# 运行SCAFFOLD算法
def runFedScaFFold(samplingName, settingName,cluster_strategy):
    sampling_strategy_name=samplingName
    num_clients = 10
    # 初始化全局模型和客户端模型
    input_size = 8
    hidden_size = 100
    output_size = 2
    global_model = MLP(input_size, hidden_size, output_size)
    client_prototype_map = [{} for _ in range(num_clients)]
    clients_models = [MLP(input_size, hidden_size, output_size) for _ in range(num_clients)]

    c_global= [torch.zeros_like(param) for param in global_model.parameters()]
    c_locals = [[] for _ in range(num_clients)]

    num_epochs = 400
    num_global_updates = 36
    # 'E:/FedStream/real_data_set/realdataset0427/elecNorm/Electricity_client_random/'
    base_path ='/root/FedStream/' # 'E:/FedStream/'
    dataset_name = 'elecNorm'
    setting_name = settingName # 'CovA_Abrupt_BtoVH'
    algorithm = 'scoffod_kvalue30'# reweight
    experiment_times = 'epoch200'
    save_loss_path = f'{base_path}/loss/{dataset_name}_Sampling/{algorithm}/{setting_name}/{sampling_strategy_name}_{cluster_strategy}fa_{experiment_times}'
    save_model_path = f'{base_path}/models/{dataset_name}_Sampling/{algorithm}/{setting_name}/{sampling_strategy_name}_{cluster_strategy}fa_{experiment_times}'
    save_metrics_path = f'{base_path}/metrics/{dataset_name}_Sampling/{algorithm}/{setting_name}/{sampling_strategy_name}_{cluster_strategy}fa_{experiment_times}'
    read_data_path = f'/root/FedStream/real_data_set/realdataset0427/{dataset_name}/{setting_name}/'
    test_path = '/root/FedStream/real_data_set/realdataset0427/elecNorm/test_dataset.csv'
    X_test_tensor, y_test_tensor = read_test_data(test_path)

    for update in range(num_global_updates):
        print(f"round {update}")
        local_params_list = []

        for i in range(num_clients):
            print(f"client {i}")
            X_train_local, y_train_local, prototype_map_r = read_data_return_tensor(
                read_data_path, round_id=update, client_id=i,
                sampling_strategy=sampling_strategy_name,
                prototype_map=client_prototype_map[i],cluster_strategy=cluster_strategy
            )
            client_prototype_map[i] = prototype_map_r

            model_weights, y_delta, updated_c_local = train_model_FedScaFFold_local(
                clients_models[i], X_train_local, y_train_local, num_epochs=num_epochs,
                c_global=c_global, c_local=c_locals[i], lr=0.01
            )
            local_params_list.append((model_weights, y_delta, updated_c_local))
            c_locals[i] = updated_c_local

            local_metrics = test(copy.deepcopy(clients_models[i]), X_test_tensor, y_test_tensor)
            save_metrics(title=f"client_{i}_metrics", rounds=update, metrics=local_metrics, save_folder=save_metrics_path)

        # 聚合本地模型参数到全局模型
        aggregated_params, updated_c_global = scaffold_aggregator(local_params_list)
        global_model.load_state_dict(aggregated_params)
        c_global = updated_c_global

        # 在每轮结束后发送全局模型参数给客户端
        gm = copy.deepcopy(global_model)
        save_model(copy.deepcopy(gm), update, save_model_path)
        for client_model in clients_models:
            client_model.load_state_dict(gm.state_dict())

        me = test(gm, X_test_tensor, y_test_tensor)
        save_metrics(title="global_back", rounds=update, metrics=me, save_folder=save_metrics_path)
        print(me)

import time

dataset_list = ['Electricity_client_random','Electricity_client_random','Electricity_client_random','Electricity_client_random','Electricity_client_random']
cluster_strategy_list = ['kmeans','kmeans++','ros-p','gmm','OPTICS','meanshift','ros-h','tpl']

total_start_time = time.time()  # 记录整个循环的开始时间
ex_time_list = []
import gc
for j, settingname in enumerate(dataset_list):
    for i, cluster_strategy in enumerate(cluster_strategy_list):
        start_time = time.time()  # 记录当前迭代的开始时间
        # runFedNova(samplingName='IncrementalSampling', settingName=settingname,cluster_strategy=cluster_strategy)
        # end_time = time.time()  # 记录当前迭代的结束时间
        # execution_time = end_time - start_time  # 计算当前迭代的执行时间
        # ex_time_list.append(execution_time)
        # gc.collect()
        # time.sleep(10)
        runFedScaFFold(samplingName='IncrementalSampling', settingName=settingname,cluster_strategy=cluster_strategy)
        end_time = time.time()  # 记录当前迭代的结束时间
        execution_time = end_time - start_time  # 计算当前迭代的执行时间
        ex_time_list.append(execution_time)
        gc.collect()
        time.sleep(10)

total_end_time = time.time()  # 记录整个循环的结束时间
total_execution_time = total_end_time - total_start_time  # 计算整个循环的执行时间

print(f"Total execution time: {total_execution_time} seconds")
print(ex_time_list)

```

## 3、PokerHand数据集

```python
# 上诉代码没有共享模型
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree
from numpy.linalg import norm
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
from sklearn.metrics import f1_score,recall_score, accuracy_score, roc_auc_score,cohen_kappa_score,confusion_matrix,precision_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
import random
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,recall_score, accuracy_score, roc_auc_score,cohen_kappa_score,confusion_matrix,precision_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import copy
import os
import random
from sklearn.cluster import KMeans
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS, MeanShift ,SpectralClustering
from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
import pandas as pd
from sklearn.utils import shuffle

import os
import torch
import hdbscan
import warnings
# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class IncrementalSampling(object):
    def __init__(self,prototype_nums = 30,cluster_strategy = 'kmeans'):
        self.incremental_data_map = {}
        self.data  = np.array([])
        self.label = np.array([])
        self.k_value = prototype_nums
        # 2024 06 07
        self.cluster_strategy = cluster_strategy
    def split_current_data_by_class(self):
        """把当前轮的数据按类别划分"""
        # 获取唯一的类别标签
        current_unique_labels = np.unique(self.label)
        # 按类别分割当前轮获得的数据
        data_by_class = {}  # map {"label":[the data belong to the label]}
        for current_data_label in current_unique_labels:
            indices = np.where(self.label == current_data_label)[0]
            data_by_class[current_data_label] = self.data[indices]
        return data_by_class, current_unique_labels
    def data_combined(self):
        """把当前轮的数据，和类map中的数据进行合并"""
        current_data_map_by_class, current_unique_labels = self.split_current_data_by_class()
        # 考虑到有新类出现的情况
        # 如果current_unique_labels有新label,直接扩充incremental_data_map
        # 如果label 是incremental_data_map中已经有的，扩张incremental_data_map对应label中data的长度
        for new_data_label in current_unique_labels:
            if new_data_label in self.incremental_data_map:
                # 增量map中已经有这个标签的数据了，那就扩充这个数据
                self.incremental_data_map[new_data_label] = np.concatenate(
                    (self.incremental_data_map[new_data_label], current_data_map_by_class[new_data_label])
                )
            else:
                # 如果增量map中没有这个标签的的数据，就扩充增量map
                self.incremental_data_map[new_data_label] = current_data_map_by_class[new_data_label]
    def cluster_data(self,data, num_clusters):
        # ['kmeans','spectral','kmeans++','kmedoids','OPTICS','meanshift','gmm','hdbscan']
        # kmeans = KMeans(n_clusters=num_clusters,n_init='auto')
        cluster_centers = np.ndarray([])
        if self.cluster_strategy == 'kmeans':
            print("using  kmeans")
            with warnings.catch_warnings():
            # KMeans(n_clusters=num_clusters,n_init='auto')
                kmeans = KMeans(n_clusters=num_clusters,n_init='auto')
                # kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(data)
                cluster_centers = kmeans.cluster_centers_
        elif self.cluster_strategy == 'spectral':
            print("using spectual")
            with warnings.catch_warnings():
                # 报错，小类样本太少，无法聚类
                print('cluster : spectral',{num_clusters})
                spectral = SpectralClustering(n_clusters=num_clusters, random_state=0, affinity='nearest_neighbors').fit(data)
                labels_spectral = spectral.labels_
                cluster_centers = np.array([data[labels_spectral == i].mean(axis=0) for i in range(num_clusters)])
                print(len(labels_spectral))
                print(len(cluster_centers))
        elif self.cluster_strategy.lower() == 'hdbscan':
            print("hdbscan")
            with warnings.catch_warnings():
                min_cluster_size = 2
                hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(data)
                labels_hdbscan = hdb.labels_
                cluster_centers = np.array([data[labels_hdbscan == i].mean(axis=0) for i in range(len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0))])
        elif self.cluster_strategy.lower() == 'kmeans++' or self.cluster_strategy.lower() == 'kmeansplusplus' :
            print("using kmeans++")
            with warnings.catch_warnings():
                # kmeans_plus_plus = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0).fit(data)
                kmeans_plus_plus = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0,n_init='auto').fit(data)
                cluster_centers = kmeans_plus_plus.cluster_centers_
        elif self.cluster_strategy.lower() =='kmedoids':
            print("using medorid")
            with warnings.catch_warnings():
                # 有几个空簇
                print('cluster : spectral',{num_clusters})
                kmedoids = KMedoids(n_clusters=num_clusters-10, random_state=0).fit(data)
                labelxx = kmedoids.labels_
                cluster_centers = kmedoids.cluster_centers_
                print(len(cluster_centers))
        elif self.cluster_strategy.upper() =='OPTICS':
            print("using optics")
            with warnings.catch_warnings():
                min_samples = 2
                optics = OPTICS(min_samples=min_samples).fit(data)
                labels_optics = optics.labels_
                cluster_centers = np.array([data[labels_optics == i].mean(axis=0) for i in range(len(set(labels_optics)) - (1 if -1 in labels_optics else 0))])
                print(len(cluster_centers))
        elif self.cluster_strategy.lower() =='meanshift':
            print("using meanshift")
            with warnings.catch_warnings():
                bandwidth = 0.1
                mean_shift = MeanShift(bandwidth=bandwidth).fit(data)
                cluster_centers = mean_shift.cluster_centers_
        elif self.cluster_strategy.lower() =='gmm':
            print("using gmm")
            with warnings.catch_warnings():
                gmm = GaussianMixture(n_components=num_clusters, random_state=0).fit(data)
                cluster_centers = gmm.means_  # 高斯混合模型的质心是每个成分的均值
        return  cluster_centers
    def reCludter(self):
        # incremental_data_map的大小进行压缩self.reCluster_instance_nums
        new_cluster = {}
        for label, data in self.incremental_data_map.items():
            if len(data) <= self.k_value:
                # 全部保存
                new_cluster[label] = data
            else:
                sampled_nums = self.k_value
                clusters = self.cluster_data(data = data, num_clusters=sampled_nums)
                new_cluster[label] = clusters
        self.incremental_data_map = new_cluster
    def compute_sampling_nums(self,combinde_map):
        # initialize
        min_length = float('inf')  # 初始化最小长度为正无穷大
        max_length = 0  # 初始化最大长度为0
        for label, data in combinde_map.items():
            data_length = len(data)
            if data_length < min_length:
                min_length = data_length
            if data_length > max_length:
                max_length = data_length
        return min_length, max_length
    def data_sampling(self,sampling_nums):
        resampling_data = []
        resampling_label = []
        for label, data in self.incremental_data_map.items():
            if len(data) > sampling_nums:
                # 那就是下采样了,不放回采样
                sampled_indices = random.sample(range(len(data)), sampling_nums)
                sampled_data = [data[i] for i in sampled_indices]
                resampling_data.extend(sampled_data)
                resampling_label.extend([label] * sampling_nums)
            elif len(data) == sampling_nums:
                # 直接复制
                resampling_data.extend(data)
                resampling_label.extend([label] * sampling_nums)
            else:
                # 上采样,保存原样本
                resampling_data.extend(data)
                # 随机有放回的找差额部分
                sampled_data = random.choices(data, k=(sampling_nums-len(data)))
                resampling_data.extend(sampled_data)
                resampling_label.extend([label] * sampling_nums)
        # 洗牌
        combined_data = list(zip(resampling_data, resampling_label))
        random.shuffle(combined_data)
        resampling_data, resampling_label = zip(*combined_data)
        return resampling_data,resampling_label
    def fit(self,incremental_prototype_map,data,label,sampling_strategy = "OverSampling"):
        self.incremental_data_map = incremental_prototype_map
        self.data = data
        self.label = label
        self.data_combined()
        resampling_data = []
        resampling_label = []
        min_length, max_length =self.compute_sampling_nums(self.incremental_data_map)
        if sampling_strategy.lower() == "oversampling":
            resampling_data,resampling_label = self.data_sampling(max_length)
        elif sampling_strategy.lower() == "downsampling":
            resampling_data,resampling_label = self.data_sampling(min_length)
        else:
            print("No sampling measures have been taken")
        self.reCludter()
        return resampling_data,resampling_label,self.incremental_data_map
# ['kmeans','spectral','kmeans++','kmedoids','OPTICS','meanshift','gmm']
class Incremental_sampling2(object):
    def __init__(self,save_prototype_nums = 30):
        self.save_prototype_nums = save_prototype_nums
        self.incremental_prototypes = {}
        self.data  = np.array([])
        self.label = np.array([])
        self.combined_map = {}
    def split_current_data_by_class(self):
        """把当前轮的数据按类别划分"""
        # 获取唯一的类别标签
        current_unique_labels = np.unique(self.label)
        # 按类别分割当前轮获得的数据
        data_by_class = {}  # map {"label":[the data belong to the label]}
        for current_data_label in current_unique_labels:
            indices = np.where(self.label == current_data_label)[0]
            data_by_class[current_data_label] = self.data[indices]
        return data_by_class, current_unique_labels
    def data_combined(self):
        """
        上一轮获得的数据原型和这一轮的新数据进行合并
        1.首先，按不同类别把数据和原型进行分开
        2.判断是不是新出现的类别的数据
            2.1、新类别数据,在原型map中直接扩展一个新类的map,{'新类':新类的数据}
            2.2、原先类别的数据,在对应类的数据上进行扩展，{'已有类':已有数据+新数据}
        """
        current_data_map_by_class, current_unique_labels = self.split_current_data_by_class()
        for new_data_label in current_unique_labels:
            if new_data_label in self.incremental_prototypes:
                # 增量map中已经有这个标签的数据了，那就扩充这个数据
                self.incremental_prototypes[new_data_label] = np.concatenate(
                    (self.incremental_prototypes[new_data_label], current_data_map_by_class[new_data_label])
                )
            else:
                # 如果增量map中没有这个标签的的数据，就扩充增量map
                self.incremental_prototypes[new_data_label] = current_data_map_by_class[new_data_label]
        self.combined_map = self.incremental_prototypes
    def cut_down_nearest_data_eu(self):
        """
        把每个类对应的原型中减去超出save_prototype_nums的样本
        判断规则：
            从离得最近的开始删除
        """
        for label in self.incremental_prototypes:
            data = self.incremental_prototypes[label]
            if len(data) > self.save_prototype_nums:
                # 计算成对距离
                pairwise_distances = squareform(pdist(data, 'euclidean'))
                np.fill_diagonal(pairwise_distances, np.inf)  # 将自身距离设置为无穷大，忽略自身距离

                while len(data) > self.save_prototype_nums:
                    # 找到距离最小的一对
                    min_dist_indices = np.unravel_index(np.argmin(pairwise_distances), pairwise_distances.shape)
                    # 保留一个样本，删除另一个
                    data = np.delete(data, min_dist_indices[1], axis=0)
                    # 从距离矩阵中删除对应的行和列
                    pairwise_distances = np.delete(pairwise_distances, min_dist_indices[1], axis=0)
                    pairwise_distances = np.delete(pairwise_distances, min_dist_indices[1], axis=1)

                # 使用简化后的数据更新原型
                self.incremental_prototypes[label] = data
    def cut_down_nearest_data_kdtree(self):
        """
        把每个类对应的原型中减去超出save_prototype_nums的样本
        判断规则：
            从离得最近的开始删除
        """
        for label in self.incremental_prototypes:
            # self.incremental_prototypes 在这之前已经和新数据合并了
            data = self.incremental_prototypes[label]
            if len(data) > self.save_prototype_nums:
                while len(data) > self.save_prototype_nums:
                    # 建立 KD 树
                    kdtree = KDTree(data)
                    # 查询每个点的最近邻
                    distances, indices = kdtree.query(data, k=2)  # k=2 因为第一个最近邻是点本身

                    # 找到最近的两个点
                    min_dist_idx = np.argmin(distances[:, 1])  # distances[:, 1] 是每个点的最近邻距离
                    nearest_idx = indices[min_dist_idx, 1]

                    # 删除其中一个点
                    data = np.delete(data, nearest_idx, axis=0)

                # 使用简化后的数据更新原型
                self.incremental_prototypes[label] = data
    def cut_down_nearest_data_nn(self):
        """
        把每个类对应的原型中减去超出save_prototype_nums的样本
        判断规则：
            使用批量删除
        """
        for label in self.incremental_prototypes:
            data = self.incremental_prototypes[label]
            if len(data) > self.save_prototype_nums:
                while len(data) > self.save_prototype_nums:
                    # 建立 NearestNeighbors 模型
                    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(data)
                    distances, indices = nbrs.kneighbors(data)

                    # 找到距离最小的一对
                    min_dist_idx = np.argmin(distances[:, 1])
                    nearest_idx = indices[min_dist_idx, 1]

                    # 批量删除，尽量减少删除操作次数
                    delete_indices = [min_dist_idx, nearest_idx]
                    data = np.delete(data, delete_indices, axis=0)

                # 使用简化后的数据更新原型
                self.incremental_prototypes[label] = data
    def cut_down_nearest_data(self):
        """
        把每个类对应的原型中减去超出save_prototype_nums的样本
        判断规则：
            保留距离最远的样本
        """
        for label in self.incremental_prototypes:
            data = self.incremental_prototypes[label]
            if len(data) > self.save_prototype_nums:
                # 计算成对距离矩阵
                pairwise_distances = squareform(pdist(data, 'euclidean'))

                # 对距离矩阵进行排序，获取距离最远的样本索引
                farthest_indices = np.argsort(-pairwise_distances.sum(axis=1))

                # 保留距离最远的前save_prototype_nums个样本
                keep_indices = farthest_indices[:self.save_prototype_nums]
                data = data[keep_indices]

                # 使用简化后的数据更新原型
                self.incremental_prototypes[label] = data
    def compute_sampling_nums(self,combined_map):
        # initialize
        min_length = float('inf')  # 初始化最小长度为正无穷大
        max_length = 0  # 初始化最大长度为0
        for label, data in combined_map.items():
            data_length = len(data)
            if data_length < min_length:
                min_length = data_length
            if data_length > max_length:
                max_length = data_length
        return min_length, max_length
    def triplet_sampling(self, num_cluster, n_neighbors=5, randomOr=True, len_lim=True):
        """Triplets 数据采样"""
        gen_x = []
        gen_y = []
        for label in self.incremental_prototypes:
            data_min = self.incremental_prototypes[label]
            if len(data_min) < num_cluster:
                size = num_cluster - len(data_min)
                weight = np.ones(len(data_min))
                # 收集多数类样本
                data_maj = np.vstack([self.incremental_prototypes[l] for l in self.incremental_prototypes if l != label])
                gen_x_c, gen_y_c = self._sample_one(data_min, data_maj, label, size, weight, n_neighbors, randomOr, len_lim)
                gen_x += gen_x_c
                gen_y += gen_y_c
        resampling_data = []
        resampling_label = []
        for label in self.combined_map: # incremental_prototypes样本太少了，只是为了生成样本用的
            data = self.incremental_prototypes[label]
            resampling_data.append(data)
            resampling_label.extend([label] * len(data))
        resampling_data = np.vstack(resampling_data)
        resampling_label = np.array(resampling_label)
        if len(gen_x) > 0:
            gen_x = np.vstack(gen_x)
            gen_y = np.array(gen_y)
            resampling_data = np.concatenate((resampling_data, gen_x), axis=0)
            resampling_label = np.concatenate((resampling_label, gen_y), axis=0)
        # 洗牌
        combined_data = list(zip(resampling_data, resampling_label))
        random.shuffle(combined_data)
        resampling_data, resampling_label = zip(*combined_data)
        return resampling_data,resampling_label
    def _sample_one(self, data_min, data_maj, label, size, weight, n_neighbors, randomOr, len_lim):
        gen_x = []
        gen_y = []
        if size == 0: return gen_x, gen_y

        min_idxs = np.arange(len(data_min))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data_maj)
        _, indices = nbrs.kneighbors(data_min)

        for j in np.random.choice(len(min_idxs), size, p=weight / weight.sum()):
            tp1 = data_min[min_idxs[j]]
            tp2 = data_maj[indices[j][:5]].mean(axis=0)
            tp3_ord = np.random.randint(n_neighbors)
            tp3 = data_maj[indices[j][tp3_ord]]
            if (tp2 == tp1).all():
                gen_x.append(tp1)
                gen_y.append(label)
                continue

            offset = tp3 - tp2
            offset_norm = norm(offset)
            if offset_norm == 0:
                continue

            tp1_tp2_norm = norm(tp1 - tp2)
            if tp1_tp2_norm == 0:
                continue

            if len_lim: offset = offset * min(1, tp1_tp2_norm / offset_norm)
            coef = np.random.rand() if randomOr else 1.0
            new_x = tp1 + coef * offset
            gen_x.append(new_x)
            gen_y.append(label)

        return gen_x, gen_y

    def random_sampling(self,num_cluster,sampling_strategy):
        """
        这里需要对数据采样
        首先遍历self.incremental_prototypes ,每个类
        以及每个类的数据的长度
        然后比较每个类的数据的长度和num_cluster之间的差距
        差额部分使用triplets的核心算法对齐进行生成样本
        """
        resampling_data = []
        resampling_label = []
        if sampling_strategy == "ros-p":
            # 随机上采样，差额数据从原型数据中随机复制
            for label, data in self.combined_map.items():
                if len(data) < num_cluster:  # 需要上采样的数据的条件
                    prototype_data = self.incremental_prototypes[label]
                    sampling_nums = num_cluster - len(data)

                    if len(prototype_data) < sampling_nums:
                        sampled_data = []
                        while len(sampled_data) < sampling_nums:
                            # 原型中样本还没有sampling_nums多时，直接复制原型中的数据
                            needed = sampling_nums - len(sampled_data)
                            sampled_indices = random.sample(range(len(prototype_data)), min(needed, len(prototype_data)))
                            sampled_data.extend([prototype_data[i] for i in sampled_indices])
                    else:
                        sampled_indices = random.sample(range(len(prototype_data)), sampling_nums)
                        sampled_data = [prototype_data[i] for i in sampled_indices]

                    resampling_data.extend(data)
                    resampling_label.extend([label] * len(data))
                    resampling_data.extend(sampled_data)
                    resampling_label.extend([label] * sampling_nums)
                elif len(data) == num_cluster:
                    resampling_data.extend(data)
                    resampling_label.extend([label] * len(data))
        elif sampling_strategy == "ros-h" :
            # 随机上采样，从混合数据中采样差额数据
            for label, data in self.combined_map.items():
                data_len = len(data)
                if data_len < num_cluster:
                    # 那就是下采样了,不放回采样
                    sampling_nums = num_cluster-data_len
                    if data_len < sampling_nums:
                        sampled_data = []
                        while len(sampled_data) < sampling_nums:
                            # 原型中样本还没有sampling_nums多时，直接复制原型中的数据
                            needed = sampling_nums - len(sampled_data)
                            sampled_indices = random.sample(range(data_len), min(needed, data_len))
                            sampled_data.extend([data[i] for i in sampled_indices])
                    else:
                        sampled_indices = random.sample(range(data_len), sampling_nums)
                        sampled_data = [data[i] for i in sampled_indices]
                    resampling_data.extend(data)
                    resampling_label.extend([label] * len(data))
                    resampling_data.extend(sampled_data)
                    resampling_label.extend([label] * sampling_nums)
                elif len(data) == num_cluster:
                    # 直接复制
                    resampling_data.extend(data)
                    resampling_label.extend([label] * len(data))
        # 洗牌
        combined_data = list(zip(resampling_data, resampling_label))
        random.shuffle(combined_data)
        resampling_data, resampling_label = zip(*combined_data)
        return resampling_data,resampling_label
    def fit(self,new_data,new_data_label,last_round_prototype,sampling_strategy='tpl'):
        self.incremental_prototypes = last_round_prototype
        self.data = new_data
        self.label = new_data_label
        self.data_combined()
        resampling_data = []
        resampling_label = []
        min_length, max_length =self.compute_sampling_nums(self.incremental_prototypes)
        if sampling_strategy.lower() == "tpl":
            resampling_data,resampling_label = self.triplet_sampling(max_length)
        elif sampling_strategy.lower() == "ros-p":
            # rest data copied from prototype
            resampling_data,resampling_label = self.random_sampling(max_length,sampling_strategy = "ros-p")
            print('use ros p')
        elif sampling_strategy.lower() == "ros-h":
            # rest data copied from hybrid data(combined data)
            resampling_data,resampling_label = self.random_sampling(max_length,sampling_strategy = "ros-h")
            print('use ros h')
        return resampling_data,resampling_label,self.incremental_prototypes
class Triplets(object):
    def __init__(self, n_neighbors=5, random=True, len_lim=True, **kwargs):
        self.n_neighbors = n_neighbors
        self.random = random
        self.len_lim = len_lim

    def fit_resample(self, x, y):
        strategy = self._sample_strategy(y)
        self.n_neighbors = max(self.n_neighbors, self.counts.max() // self.counts.min())

        gen_x = []
        gen_y = []
        # 这里的代码平衡状态会报错
        for c, size in enumerate(strategy):
            if size == 0: continue
            weight = self._weights(x, y, c)
            gen_x_c, gen_y_c = self._sample_one(x, y, c, size, weight)
            gen_x += gen_x_c
            gen_y += gen_y_c

        # 为了这个方法在平衡状态下不报错，我们特地在这里加了这段代码
        # To prevent errors in this method when in a balanced state, we intentionally added this code block
        if len(gen_x)==0:
            return x,y
        gen_x = np.vstack(gen_x)
        gen_y = np.array(gen_y)
        return np.concatenate((x, gen_x), axis=0), np.concatenate((y, gen_y), axis=0)

    def _sample_strategy(self, y):
        _, self.counts = np.unique(y, return_counts=True)
        return max(self.counts) - self.counts

    def _weights(self, x, y, c):
        return np.ones(self.counts[c])

    def _sample_one(self, x, y, c, size, weight):
        gen_x = []
        gen_y = []
        if size == 0: return gen_x, gen_y

        # get the indices of minority and majority instances
        min_idxs = np.where(y == c)[0]
        maj_idxs = np.where(y != c)[0]

        # find nearest majority neighbors for each minority instance
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(x[maj_idxs])
        _, indices = nbrs.kneighbors(x[min_idxs])

        # generate synthetic data
        for j in np.random.choice(len(min_idxs), size, p=weight / weight.sum()):
            tp1 = x[min_idxs[j]]
            tp2 = x[maj_idxs[indices[j][:5]]].mean(axis=0)
            # tp3_ord = np.random.randint(1, self.n_neighbors)
            tp3_ord = np.random.randint(self.n_neighbors)
            tp3 = x[maj_idxs[indices[j][tp3_ord]]]
            if (tp2 == tp1).all():
                gen_x.append(tp1)
                gen_y.append(c)
                continue

            offset = tp3 - tp2
            if self.len_lim: offset = offset * min(1, norm(tp1 - tp2) / norm(offset))
            coef = np.random.rand() if self.random is True else 1.0
            new_x = tp1 + coef * offset
            gen_x.append(new_x)
            gen_y.append(c)
        return gen_x, gen_y

def train_model_FedAvg_local(input_model,X_train_tensor, y_train_tensor, num_epochs):
    losses =[]
    model = input_model# copy.deepcopy(input_model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        losses.append(loss.item())  # 将每次训练的损失值添加到列表中

    return losses
def train_model_FedProx_local(input_model, X_train_tensor, y_train_tensor, num_epochs):
    mu = 0.1
    # because last round trained global model replaced local model,
    # that in this round the first local model is last round global model
    losses =[]
    model = input_model# copy.deepcopy(input_model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    global_weights = copy.deepcopy(list(model.parameters()))
    # not use deepcopy ,because model as parameter transport in this ,update model also update model
    # current_local_model = cmodel
    # model.train()
    # for epoch in range(num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        # FedProx
        prox_term = 0.0
        for p_i, param in enumerate(model.parameters()):
                prox_term += (mu / 2) * torch.norm((param - global_weights[p_i])) ** 2
        loss += prox_term

        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        losses.append(loss.item())  # 将每次训练的损失值添加到列表中
    #     total_loss = 0.0
    #     for step,(x,y) in enumerate(zip(X_train_tensor,y_train_tensor)):
    #         # current_local_model.train()
    #         output = model(x) # current_local_model(x)
    #         loss = criterion(output, y)
    #         total_loss+=loss.item()
    #         optimizer.zero_grad()
    #
    #         # fedprox
    #         prox_term = 0.0
    #         for p_i, param in enumerate(model.parameters()):
    #             prox_term += (mu / 2) * torch.norm((param - global_weights[p_i])) ** 2
    #         loss += prox_term
    #         loss.backward()
    #         optimizer.step()
    #     losses.append(total_loss)
    #     if epoch % 10 == 0:
    #         print(f"Epoch [{epoch+1}/{num_epochs}], Epoch Total Loss: {total_loss:.4f}")
    return losses

def train_model_FedNova_local(input_model,X_train_tensor, y_train_tensor, num_epochs):
    # because last round trained global model replaced local model,
    # that in this round the first local model is last round global model
    losses =[]
    model = input_model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    global_weights = copy.deepcopy(input_model.state_dict())
    tau = 0
    rho = 0.9
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        tau +=len(y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        losses.append(loss.item())
    # for epoch in range(num_epochs):
    #     total_loss = 0.0
    #     for step,(x,y) in enumerate(zip(X_train_tensor,y_train_tensor)):
    #         # current_local_model.train()
    #         # model.train()
    #         output = model(x) # current_local_model(x)
    #         loss = criterion(output, y)
    #         total_loss+=loss.item()
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         tau +=1
    #     losses.append(total_loss)
    #     if epoch % 10 == 0:
    #         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
    coeff = (tau - rho * (1 - pow(rho, tau)) / (1 - rho)) / (1 - rho)
    state_dict = model.state_dict()
    norm_grad = copy.deepcopy(global_weights)
    for key in norm_grad:
        norm_grad[key] = torch.div(global_weights[key] - state_dict[key], coeff)

    return losses, coeff, norm_grad,len(X_train_tensor)
# 定义模型参数共享函数
def share_params(model):
    params = model.state_dict()
    # return {k: v.clone().detach().requires_grad_(True) for k, v in params.items()}
    return {k: v.clone().detach().requires_grad_(False) for k, v in params.items()}

# # 定义模型参数聚合函数
def aggregate_params(params_list):
    aggregated_params = {}
    for key in params_list[0].keys():
        # 将参数转换为张量进行处理
        params_tensors = [params[key].clone().detach().float() for params in params_list]
        # 聚合参数
        aggregated_params[key] = sum(params_tensors) / len(params_tensors)
    return aggregated_params
def test(global_model,X_test,y_test):
    # 在全局模型上进行测试
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        outputs = global_model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        # 计算度量
        predictions = predicted.numpy() # 将张量转换为 NumPy 数组并去除零维数组
        true_labels = y_test_tensor.numpy()  # 将张量转换为 NumPy 数组并去除零维数组
        precision = precision_score(true_labels,predictions,zero_division=0.0,average=None)
        precision_micro = precision_score(true_labels,predictions,zero_division=0.0,average='micro')
        precision_macro = precision_score(true_labels,predictions,zero_division=0.0,average='macro')
        # recall
        recalls = recall_score(true_labels,predictions,zero_division=0.0,average=None)
        recalls_micro =recall_score(true_labels,predictions,zero_division=0.0,average='micro')
        recalls_macro =recall_score(true_labels,predictions,zero_division=0.0,average='macro')
        f1_scores = f1_score(true_labels, predictions, average=None)
        acc = accuracy_score(true_labels, predictions)
        kappa = cohen_kappa_score(true_labels,predictions)
        conf_matrix = confusion_matrix(true_labels,predictions)
        # 计算所有类别乘积的几何平均值作为 G-mean
        g_mean_all= np.power(np.prod(recalls), 1 / len(recalls))
        # AUC
        lb = LabelBinarizer()
        lb.fit(true_labels)
        true_labels_bin = lb.transform(true_labels)
        predictions_bin = lb.transform(predictions)
        auc = roc_auc_score(true_labels_bin, predictions_bin, average='weighted', multi_class='ovr')
        metrics = {
            'recall':recalls,
            'recall_micro':recalls_micro,
            'recall_macro':recalls_macro,
            'precision':precision,
            'precision_micro':precision_micro,
            'precision_macro':precision_macro,
            'f1_score':f1_scores,
            'g_mean':g_mean_all,
            'acc':acc,
            'auc':auc,
            'kappa':kappa,
            'confusion_matrix':conf_matrix
        }
        return metrics

def save_loss(loss_list,client_id,round_id,save_loss_path):
    if not os.path.exists(save_loss_path):
        os.makedirs(save_loss_path)
    # 构建文件路径
    file_path = os.path.join(save_loss_path, f"client_{client_id}.csv")

    if os.path.exists(file_path):
        # 如果文件存在，加载现有的 CSV 文件为 DataFrame
        df = pd.read_csv(file_path)
    else:
        # 如果文件不存在，直接创建新的 DataFrame
        df = pd.DataFrame()

    # 将损失值添加到 DataFrame 中
    column_name = f'round_{round_id}'
    df[column_name] = loss_list

    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(file_path, index=False)
def save_model(global_model,round_id,save_model_path):
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    model_path = os.path.join(save_model_path,f'round_{round_id}_gm.pt')
    torch.save(global_model,model_path)

def save_metrics(title, rounds, metrics, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    file_name = f"{title}.csv"
    file_path = os.path.join(save_folder, file_name)
    # print(file_path)
    recalls = metrics['recall']
    class_nums = len(recalls)
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 如果文件存在，加载现有的 Excel 文件为 DataFrame
        df = pd.read_csv(file_path)
    else:
        # 如果文件不存在，直接创建新的 DataFrame
        columns = [
        'rounds', 'accuracy', 'auc', 'kappa', 'g_mean', 'recall_micro', 'precision_micro',
        'recall_macro', 'precision_macro'
        ]
        df = pd.DataFrame(columns=columns)
        for i in range(class_nums):  # 动态生成 f1-score 相关列名
            columns.append(f'f1_score_{i}')
            columns.append(f'recall_{i}')
            columns.append(f'precession_{i}')

    data = {
        'rounds': rounds,
        'accuracy': metrics['acc'],
        'auc': metrics['auc'],
        'kappa': metrics['kappa'],
        'g_mean':metrics['g_mean'],
        'recall_micro':metrics['recall_micro'],
        'precision_micro':metrics['precision_micro'],
        'recall_macro':metrics['recall_macro'],
        'precision_macro':metrics['precision_macro']
    }
    # 添加每个类别的 F1-score、G-mean 和 Recall 到 data 中
    for i in range(class_nums):  #类别数
        data[f'recall_{i}'] = metrics['recall'][i]
        data[f'precision_{i}'] = metrics['precision'][i]
        data[f'f1_score_{i}'] = metrics['f1_score'][i]
    # 创建新行并追加到 DataFrame
    new_row = pd.DataFrame(data, index=[0])
    df = pd.concat([df, new_row], ignore_index=True)

    # 将 DataFrame 保存为 Excel 文件
    df.to_csv(file_path, index=False)

from imblearn.over_sampling import SMOTE,RandomOverSampler,SMOTENC,SMOTEN,ADASYN,BorderlineSMOTE,KMeansSMOTE,SVMSMOTE
from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler,NearMiss,TomekLinks,EditedNearestNeighbours,RepeatedEditedNearestNeighbours,AllKNN,CondensedNearestNeighbour,OneSidedSelection,NeighbourhoodCleaningRule,InstanceHardnessThreshold
from imblearn.combine import SMOTEENN,SMOTETomek
# SMOTE,ROS,SMOTENC,SMOTEN,ADASYN,BorderlineSMOTE1,BorderlineSMOTE2,KMeansSMOTE,SVMSMOTE
# ClusterCentroids,RUS,NearMiss1,NearMiss2,NearMiss2,TomekLinks,ENN,RENN,AllKNN,CNN,OSS,NC,IHT
# SMOTEENN,SMOTETomek
def data_sampling(raw_X,raw_y,sampling_strategy):
    if sampling_strategy.upper() == 'NO' :
        return raw_X,raw_y
    # overSampling
    elif sampling_strategy.upper() == 'SMOTE': # overSampling
        """
            1.对于样本x ,按照欧氏距离找到离其距离最近的K个近邻样本
            2.确定采样比例，然后从K个近邻中选择x_n
            3.公式 x_new = x + rand(0,1)*(x_n-x)
        """
        smote = SMOTE( random_state=42)
        X_resampled, y_resampled = smote.fit_resample(raw_X,raw_y)
        return X_resampled, y_resampled
    elif sampling_strategy == "RandomOverSampler" or sampling_strategy.upper()=='ROS': # overSampling
        ros = RandomOverSampler(random_state=1)
        ros_data,ros_label = ros.fit_resample(raw_X,raw_y)
        return ros_data,ros_label
    elif sampling_strategy.upper() == 'SMOTENC':    # overSampling
        smotenc = SMOTENC(random_state=1,categorical_features=[0])
        smotenc_data,smotenc_label = smotenc.fit_resample(raw_X,raw_y)
        return smotenc_data,smotenc_label
    elif sampling_strategy.upper() == 'SMOTEN': # overSampling
        smoten = SMOTEN(random_state=1)
        smoten_data,smoten_label = smoten.fit_resample(raw_X,raw_y)
        return smoten_data,smoten_label
    elif sampling_strategy.upper() =='ADASYN':
        adasyn = ADASYN(random_state=1)
        adasyn_data,adasyn_label = adasyn.fit_resample(raw_X,raw_y)
        return adasyn_data,adasyn_label
    elif sampling_strategy == 'BorderlineSMOTE1' or sampling_strategy.upper()=='BSMOTE1':
        bsmote1 = BorderlineSMOTE(kind='borderline-1',random_state=1)
        bsmote1_data,bsmote1_label = bsmote1.fit_resample(raw_X,raw_y)
        return bsmote1_data,bsmote1_label
    elif sampling_strategy == 'BorderlineSMOTE2'or sampling_strategy.upper()=='BSMOTE2':
        bsmote2 = BorderlineSMOTE(kind='borderline-2',random_state=1)
        bsmote2_data,bsmote2_label = bsmote2.fit_resample(raw_X,raw_y)
        return bsmote2_data,bsmote2_label
    elif sampling_strategy == 'KMeansSMOTE' or sampling_strategy.upper() == 'KSMOTE':
        kmeanssmote = KMeansSMOTE(random_state=1)
        kmeanssmote_data,kmeanssmote_label = kmeanssmote.fit_resample(raw_X,raw_y)
        return kmeanssmote_data,kmeanssmote_label
    elif sampling_strategy == 'SVMSMOTE':
        svmsmote = SVMSMOTE(random_state=1)
        svmsmote_data,svmsmote_label = svmsmote.fit_resample(raw_X,raw_y)
        return svmsmote_data,svmsmote_label
    # downSampling
    elif sampling_strategy == 'ClusterCentroids': # down-sampling,generate
        clustercentroids = ClusterCentroids(random_state=1)
        clustercentroids_data,clustercentroids_label = clustercentroids.fit_resample(raw_X,raw_y)
        return clustercentroids_data,clustercentroids_label
    elif sampling_strategy=='RandomUnderSampler' or sampling_strategy.upper()=='RUS':
        rus = RandomUnderSampler(random_state=1)
        rus_data,rus_label = rus.fit_resample(raw_X,raw_y)
        return rus_data,rus_label
    elif sampling_strategy.upper() =='NEARMISS1':
        # 在k个少数类别样本中，选择出与他们-平均距离最近的多数类样本-进行保存
        nearmiss1 = NearMiss(version=1)
        nearmiss1_data,nearmiss1_label = nearmiss1.fit_resample(raw_X,raw_y)
        return nearmiss1_data,nearmiss1_label
    elif sampling_strategy.upper() =='NEARMISS2':
        # 选择K个距离最远的少数类别样本，然后根据这些样本选出的"平均距离最近"的样本进行保存
        nearmiss2 = NearMiss(version=2)
        nearmiss2_data,nearmiss2_label = nearmiss2.fit_resample(raw_X,raw_y)
        return nearmiss2_data,nearmiss2_label
    elif sampling_strategy.upper() =='NEARMISS3':
        # 1、对于每一个少数类别样本，保留其K个最近邻多数类样本；2、把到K个少数样本平均距离最大的多数类样本保存下来。
        nearmiss3 = NearMiss(version=3)
        nearmiss3_data,nearmiss3_label = nearmiss3.fit_resample(raw_X,raw_y)
        return nearmiss3_data,nearmiss3_label
    elif sampling_strategy == 'TomekLinks' or sampling_strategy.upper()=='TOMEK':
        # 它需要计算每个样本之间的距离，然后把互为最近邻且类别不同的一对样本拿出来，根据需求的选择将这一对数据进行剔除 or 把多数类样本剔除
        tomelink = TomekLinks(sampling_strategy='all')#sampling_strategy='all'表示全部删除，'auto'表示只删除多数类
        tomelink_data,tomelink_label = tomelink.fit_resample(raw_X,raw_y)
        return tomelink_data,tomelink_label
    elif sampling_strategy == 'EditedNearestNeighbours' or sampling_strategy.upper() == 'ENN':
        ENN = EditedNearestNeighbours()
        ENN_data,ENN_label = ENN.fit_resample(raw_X,raw_y)
        return ENN_data,ENN_label
    elif sampling_strategy == 'RepeatedEditedNearestNeighbours' or sampling_strategy.upper() == 'RENN':
        RENN = RepeatedEditedNearestNeighbours()
        RENN_data,RENN_label = RENN.fit_resample(raw_X,raw_y)
        return RENN_data,RENN_label
    elif sampling_strategy =='AllKNN':
        ## ENN的改进版本，和RepeatedEditedNearestNeighbours一样，会多次迭代ENN 算法，不同之处在于，他会每次增加KNN的K值
        allknn = AllKNN()
        allknn_data,allknn_label = allknn.fit_resample(raw_X,raw_y)
        return allknn_data,allknn_label
    elif sampling_strategy == 'CondensedNearestNeighbour'or sampling_strategy.upper() == 'CNN':
        ## 如果有样本无法和其他多数类样本聚类到一起，那么说明它极有可能是边界的样本，所以将这些样本加入到集合中
        CNN = CondensedNearestNeighbour(random_state=1)
        CNN_data,CNN_label = CNN.fit_resample(raw_X,raw_y)
        return CNN_data,CNN_label
    elif sampling_strategy == 'OneSidedSelection' or sampling_strategy.upper() == 'OSS':
        # OneSidedSelection = tomekLinks + CondensedNearestNeighbour,先使用自杀式的方式把大类数据中的其他值剔除，然后再使用CondensedNearestNeighbour的下采样
        OSS = OneSidedSelection(random_state=1)
        OSS_data,OSS_label = OSS.fit_resample(raw_X,raw_y)
        return OSS_data,OSS_label
    elif sampling_strategy == 'NeighbourhoodCleaningRule'or sampling_strategy.upper() == 'NC':
        # 若在大类的K-近邻中，少数类占多数，那就剔除这个多数类别的样本
        NC = NeighbourhoodCleaningRule()
        NC_data,NC_label = NC.fit_resample(raw_X,raw_y)
        return NC_data,NC_label
    elif sampling_strategy == 'InstanceHardnessThreshold' or sampling_strategy.upper() == 'IHT':
        # 默认算法是随机森林，通过分类算法给出样本阈值来剔除部分样本，（阈值较低的可以剔除）,慢
        IHT = InstanceHardnessThreshold(random_state=1)
        IHT_data,IHT_label = IHT.fit_resample(raw_X,raw_y)
        return IHT_data,IHT_label
    # hibird
    elif sampling_strategy.upper() =='SMOTEENN':
        se = SMOTEENN(random_state=1)
        se_data,se_label = se.fit_resample(raw_X,raw_y)
        return se_data,se_label
    elif sampling_strategy.upper() =='SMOTETOMEK':
        st = SMOTETomek(random_state=1)
        st_data,st_label = st.fit_resample(raw_X,raw_y)
        return st_data,st_label
    elif sampling_strategy == 'Triplets':
        print(" Triplets sampling")
        tpl = Triplets()
        tpl_data,tpl_label = tpl.fit_resample(raw_X,raw_y)
        return tpl_data,tpl_label
    else :
        print("skipped all the sampling strategy,but return the raw data and label")
        return raw_X,raw_y
# ['kmeans','spectral','kmeans++','kmedoids','OPTICS','meanshift','gmm']
def inremental_sampling(prototype_map,raw_X,raw_y,cluster_strategy):
    class_nums = len(np.unique(raw_y))
    print("class nums" ,class_nums)
    if cluster_strategy == 'ros-p' or cluster_strategy == 'ros-h' or cluster_strategy=='tpl':
        isap = Incremental_sampling2()
        resampling_data,resampling_label,prototype_map = isap.fit(new_data=raw_X,new_data_label=raw_y,last_round_prototype=prototype_map,sampling_strategy=cluster_strategy)
    else :
        isap = IncrementalSampling(cluster_strategy=cluster_strategy)
        resampling_data,resampling_label,prototype_map = isap.fit(incremental_prototype_map=prototype_map,data=raw_X,label=raw_y)
    return resampling_data,resampling_label,prototype_map
# ['kmeans','spectral','kmeans++','kmedoids','OPTICS','meanshift','gmm']
def read_data_return_tensor(dataset_path, round_id, client_id, sampling_strategy='no',prototype_map = {},cluster_strategy='kmeans'):
    folder_path = os.path.join(dataset_path, f'client_{client_id}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    open_file_path = os.path.join(folder_path, f'round_{round_id}.csv')
    data = pd.read_csv(open_file_path, header=0)
    # 排除标题行并打乱数据
    data_shuffled = shuffle(data)  # 排除标题行并打乱数据

    # 提取特征和目标变量
    raw_X = data_shuffled.iloc[:, :-1].values  # 特征
    raw_y = data_shuffled.iloc[:, -1].values   # 目标变量
    raw_y = raw_y.astype(float)
    # print(len(raw_X),type(raw_X))
    # print(len(raw_X),type(raw_X))
    if sampling_strategy == 'IncrementalSampling':
        print("using IncrementalSampling")
        resampling_X,resampling_y,prototype_map = inremental_sampling(prototype_map,raw_X,raw_y,cluster_strategy=cluster_strategy)
    else:
        resampling_X,resampling_y = data_sampling(raw_X,raw_y,sampling_strategy)
        resampling_y.astype(float)
    resampling_X = np.array(resampling_X)  # 将列表转换为单个NumPy数组
    resampling_y = np.array(resampling_y)  # 将列表转换为单个NumPy数组

    X_train_tensor = torch.tensor(resampling_X, dtype=torch.float32)  # 特征张量
    y_train_tensor = torch.tensor(resampling_y, dtype=torch.long)   # 标签张量
    return X_train_tensor,y_train_tensor,prototype_map
def read_test_data(test_data_path):
    data = pd.read_csv(test_data_path, header=0)
    # 排除标题行并打乱数据
    data_shuffled = shuffle(data)  # 排除标题行并打乱数据

    # 提取特征和目标变量
    X = data_shuffled.iloc[:, :-1].values.astype(float)  # 特征
    # print(X)
    y = data_shuffled.iloc[:, -1].values   # 目标变量
    y = y.astype(float)

    X_test_tensor = torch.tensor(X, dtype=torch.float32)  # 特征张量
    y_test_tensor = torch.tensor(y, dtype=torch.long)   # 标签张量
    return X_test_tensor,y_test_tensor

```

`Run`

```python
def aggregate_fednova(local_params_list,gm):
    # (share_params(clients_models[i]),coeff, norm_grad, data_len) as input
    total_data_len = sum(data_len for _, _, _, data_len in local_params_list)
    global_model_state = gm.state_dict()
    nova_model_state = copy.deepcopy(global_model_state)
    # avg_loss = 0
    coeff = 0.0
    for clientID,(client_model,client_coeff,client_norm_grad,client_local_data_len) in enumerate(local_params_list):
        coeff = coeff + client_coeff*client_local_data_len/total_data_len
        for key in client_model.state_dict():
            if clientID == 0:
                nova_model_state[key] = client_norm_grad[key] * client_local_data_len/total_data_len
            else:
                nova_model_state[key] =nova_model_state[key]+ client_norm_grad[key] * client_local_data_len/total_data_len
        # avg_loss = avg_loss + cl
    for key in global_model_state:
        global_model_state[key] -= coeff*nova_model_state[key]

    return global_model_state
pp = []
def runFedNova(samplingName,settingName,cluster_strategy):
    sampling_strategy_name = samplingName
    num_clients = 10
    input_size = 10
    hidden_size = 100
    output_size = 5
    global_model = MLP(input_size, hidden_size, output_size)
    client_prototype_map = [{} for _ in range(num_clients)]
    clients_models = [MLP(input_size, hidden_size, output_size) for _ in range(num_clients)]

    num_epochs = 200
    num_global_updates = 195
    base_path = '/root/FedStream/'
    dataset_name = 'pokerhand_five'
    setting_name = settingName  
    sampling_strategy_name = samplingName
    # /root/FedStream/real_data_set/realdataset0427/covertypeNorm/covertypeNorm_client

    algorithm = 'FedNova_kvalue30_cluster'
    experiment_times = 'epoch200'
    save_loss_path = f'{base_path}/loss/{dataset_name}_Sampling/{algorithm}/{setting_name}/{sampling_strategy_name}/{cluster_strategy}'
    save_model_path = f'{base_path}/models/{dataset_name}_Sampling/{algorithm}/{setting_name}/{sampling_strategy_name}/{cluster_strategy}'
    save_metrics_path  = f'{base_path}/metrics/{dataset_name}_Sampling/{algorithm}/{setting_name}/{sampling_strategy_name}/{cluster_strategy}'
    # /root/FedStream/real_data_set/realdataset0427/covertypeNorm/covertypeNorm_client
    read_data_path = f'/root/FedStream/real_data_set/realdataset0427/{dataset_name}/{setting_name}/'
    # 读取测试集CSV文件并转换为PyTorch张量
    test_path = '/root/FedStream/real_data_set/realdataset0427/pokerhand_five/test.csv'
    X_test_tensor,y_test_tensor = read_test_data(test_path)

    for update in range(num_global_updates):
        print(f"round{update}")
        local_params_list = []

        for i in range(num_clients):
            # 在每个客户端训练本地模型
            print(f"client{i}")
            X_train_local,y_train_local,prototype_map_r = read_data_return_tensor(read_data_path,round_id=update,client_id=i,sampling_strategy=sampling_strategy_name,prototype_map=client_prototype_map[i],cluster_strategy=cluster_strategy)
            client_prototype_map[i] = prototype_map_r
            # 训练本地模型并获取损失值
            losses, coeff, norm_grad, data_len = train_model_FedNova_local(clients_models[i],X_train_local,y_train_local,num_epochs=num_epochs)
            # y_train_tensor = torch.sub(y_train_tensor, 1)
            # losses, coeff, norm_grad, data_len = train_model_FedNova_local(clients_models[i],X_train_local,torch.sub(y_train_local, 1),num_epochs=num_epochs)
            # save_loss(loss_list=losses,client_id=i,round_id=update,save_loss_path=save_loss_path)

            local_metrics = test(copy.deepcopy(clients_models[i]),X_test_tensor,y_test_tensor)
            local_params_list.append((copy.deepcopy(clients_models[i]),coeff, norm_grad, data_len))
            save_metrics(title=f"client_{i}_metrics", rounds=update, metrics=local_metrics,save_folder = save_metrics_path)

        aggregated_params = aggregate_fednova(local_params_list,gm = copy.deepcopy(global_model))
        global_model.load_state_dict(aggregated_params)

        # 在每轮结束后发送全局模型参数给客户端
        gm = copy.deepcopy(global_model)
        save_model(copy.deepcopy(gm),update,save_model_path)
        for client_model in clients_models:
            client_model.load_state_dict(gm.state_dict())

        me = test(gm,X_test_tensor,y_test_tensor) # torch.sub(y_train_tensor, 1)
        # me = test(gm,X_test_tensor,torch.sub(y_test_tensor, 1))
        save_metrics(title="global_back", rounds=update, metrics=me,save_folder = save_metrics_path)
        print("gme acc:" ,me)

def train_model_FedScaFFold_local(input_model, X_train_tensor, y_train_tensor, num_epochs, c_global, c_local, lr=0.01):
    model = input_model
    global_weights = copy.deepcopy(input_model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    # 初始化控制变量差异
    if not c_local:
        c_local = [torch.zeros_like(param) for param in model.parameters()]

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()

        # 计算并加上控制变量差异
        c_diff = [c_g - c_l for c_g, c_l in zip(c_global, c_local)]
        for param, c_d in zip(model.parameters(), c_diff):
            param.grad += c_d.data

        optimizer.step()

    # 计算 y_delta（模型参数的变化量）
    y_delta = [param.data - global_weights[name].data for name, param in model.named_parameters()]

    # 更新本地控制变量
    coef = 1 / (num_epochs * lr)
    c_local = [c_l - c_g - coef * delta for c_l, c_g, delta in zip(c_local, c_global, y_delta)]

    return model.state_dict(), y_delta, c_local
# 定义服务器端聚合函数
def scaffold_aggregator(local_params):
    global_params = local_params[0][0]
    global_c = local_params[0][2]
    num_clients = len(local_params)

    # 初始化全局y_delta和c_delta
    avg_y_delta = [torch.zeros_like(param) for param in global_params.values()]
    avg_c_delta = [torch.zeros_like(c) for c in global_c]

    for params, y_delta, c_local in local_params:
        for i, delta in enumerate(y_delta):
            avg_y_delta[i] += delta / num_clients
        for i, c_delta in enumerate(c_local):
            avg_c_delta[i] += c_delta / num_clients

    # 更新全局模型参数
    for (name, param), delta in zip(global_params.items(), avg_y_delta):
        param.data += delta

    # 更新全局控制变量
    for i, delta in enumerate(avg_c_delta):
        global_c[i] += delta

    return global_params, global_c

# 运行SCAFFOLD算法
def runFedScaFFold(samplingName, settingName,cluster_strategy):
    sampling_strategy_name = samplingName
    num_clients = 10
    input_size = 10
    hidden_size = 100
    output_size = 5
    global_model = MLP(input_size, hidden_size, output_size)
    client_prototype_map = [{} for _ in range(num_clients)]
    clients_models = [MLP(input_size, hidden_size, output_size) for _ in range(num_clients)]

    c_global= [torch.zeros_like(param) for param in global_model.parameters()]
    c_locals = [[] for _ in range(num_clients)]

    
    num_epochs = 200
    num_global_updates = 195
    base_path = '/root/FedStream/'
    dataset_name = 'pokerhand_five'
    setting_name = settingName  
    sampling_strategy_name = samplingName
    # /root/FedStream/real_data_set/realdataset0427/covertypeNorm/covertypeNorm_client

    algorithm = 'FedScofflod_kvalue30'
    experiment_times = 'epoch200'
    save_loss_path = f'{base_path}/loss/{dataset_name}_Sampling/{algorithm}/{setting_name}/{sampling_strategy_name}/{cluster_strategy}'
    save_model_path = f'{base_path}/models/{dataset_name}_Sampling/{algorithm}/{setting_name}/{sampling_strategy_name}/{cluster_strategy}'
    save_metrics_path  = f'{base_path}/metrics/{dataset_name}_Sampling/{algorithm}/{setting_name}/{sampling_strategy_name}/{cluster_strategy}'
    # /root/FedStream/real_data_set/realdataset0427/covertypeNorm/covertypeNorm_client
    read_data_path = f'/root/FedStream/real_data_set/realdataset0427/{dataset_name}/{setting_name}/'
    X_test_tensor,y_test_tensor = read_test_data(test_path)

    for update in range(num_global_updates):
        print(f"round {update}")
        local_params_list = []

        for i in range(num_clients):
            print(f"client {i}")
            X_train_local, y_train_local, prototype_map_r = read_data_return_tensor(
                read_data_path, round_id=update, client_id=i,
                sampling_strategy=sampling_strategy_name,
                prototype_map=client_prototype_map[i]
            )
            client_prototype_map[i] = prototype_map_r

            model_weights, y_delta, updated_c_local = train_model_FedScaFFold_local(
                clients_models[i], X_train_local, y_train_local, num_epochs=num_epochs,
                c_global=c_global, c_local=c_locals[i], lr=0.01
            )
            local_params_list.append((model_weights, y_delta, updated_c_local))
            c_locals[i] = updated_c_local

            local_metrics = test(copy.deepcopy(clients_models[i]), X_test_tensor, y_test_tensor)
            save_metrics(title=f"client_{i}_metrics", rounds=update, metrics=local_metrics, save_folder=save_metrics_path)

        # 聚合本地模型参数到全局模型
        aggregated_params, updated_c_global = scaffold_aggregator(local_params_list)
        global_model.load_state_dict(aggregated_params)
        c_global = updated_c_global

        # 在每轮结束后发送全局模型参数给客户端
        gm = copy.deepcopy(global_model)
        save_model(copy.deepcopy(gm), update, save_model_path)
        for client_model in clients_models:
            client_model.load_state_dict(gm.state_dict())

        me = test(gm, X_test_tensor, y_test_tensor)
        save_metrics(title="global_back", rounds=update, metrics=me, save_folder=save_metrics_path)
        print(me)
import time

dataset_list =['pokerhand_client','pokerhand_client','pokerhand_client','pokerhand_client','pokerhand_client']
cluster_strategy_list =  ['tpl','kmeans','gmm','kmeans++','OPTICS','meanshift','ros-p']#['meanshift','ros-p','kmeans','gmm','kmeans++','OPTICS','ros-h','tpl' ]

total_start_time = time.time()  # 记录整个循环的开始时间
ex_time_list = []
import gc
for j, settingname in enumerate(dataset_list):
    for i, cluster_strategy in enumerate(cluster_strategy_list):
        start_time = time.time()  # 记录当前迭代的开始时间
        # runFedNova(samplingName='IncrementalSampling', settingName=settingname,cluster_strategy=cluster_strategy)
        # end_time = time.time()  # 记录当前迭代的结束时间
        # execution_time = end_time - start_time  # 计算当前迭代的执行时间
        # ex_time_list.append(execution_time)
        # gc.collect()

total_end_time = time.time()  # 记录整个循环的结束时间
total_execution_time = total_end_time - total_start_time  # 计算整个循环的执行时间

print(f"Total execution time: {total_execution_time} seconds")
print(ex_time_list)
# 
```

