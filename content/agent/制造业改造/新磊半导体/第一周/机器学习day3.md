# Day 3：分类问题 —— 外延片 Pass / Fail 质量预测

今天开始进入真正的机器学习任务。

Day 1 讲：

```text
业务场景 → AI 任务类型
```

Day 2 讲：

```text
X 是什么，y 是什么，训练集/测试集是什么，什么是数据泄露
```

Day 3 专门讲：

```text
分类问题 Classification
```

也就是：

> 给模型一批历史样本，让它学习什么样的外延片更容易 Pass，什么样的外延片更容易 Fail。

---

# 1. 什么是分类问题？

分类问题就是：

```text
输入一组特征 X
输出一个类别 y
```

在外延片场景里，最典型的是：

```text
输入：
MBE 生长参数、设备状态、recipe、衬底批次、源材料状态

输出：
Pass / Fail
```

也可以是：

```text
低风险 / 中风险 / 高风险
```

或者：

```text
PL 异常 / XRD 异常 / 厚度异常 / 表面缺陷异常
```

---

# 2. 外延片质量预测里的分类任务

## 任务 1：综合 Pass / Fail 分类

这是第一版最推荐做的。

```text
y = Pass / Fail
```

例子：

|wafer_id|temp_std|pressure_std|alarm_count|qc_result|
|---|--:|--:|--:|---|
|W001|1.2|0.03|0|Pass|
|W002|3.8|0.12|2|Fail|
|W003|1.5|0.04|0|Pass|
|W004|4.1|0.15|3|Fail|

模型学到的可能是：

```text
温度波动越大
压力波动越大
报警次数越多
Fail 风险越高
```

---

## 任务 2：单项异常分类

综合 Pass/Fail 太粗，可以拆细。

例如：

```text
pl_fail：PL 是否异常
xrd_fail：XRD 是否异常
thickness_fail：厚度是否异常
surface_fail：表面缺陷是否异常
```

这比单纯 `qc_result` 更有业务价值。

因为工程师不只想知道：

```text
这批片有风险
```

他还想知道：

```text
风险主要来自 PL？
还是 XRD？
还是厚度？
还是表面缺陷？
```

---

## 任务 3：风险等级分类

有时候不要直接做 Pass / Fail，而是做：

```text
低风险 / 中风险 / 高风险
```

比如：

|fail_probability|risk_level|
|--:|---|
|0.05|低风险|
|0.28|中风险|
|0.72|高风险|

这个更适合生产现场。

因为第一版模型不应该直接替工程师判废，而应该做：

```text
风险提醒系统
```

不是：

```text
自动判废系统
```

---

# 3. 二分类、多分类、多标签，要分清楚

## 3.1 二分类 Binary Classification

只有两个类别：

```text
Pass / Fail
```

这是最简单的分类问题。

外延片质量预测第一版就用它。

---

## 3.2 多分类 Multi-class Classification

多个类别，但每个样本只属于其中一个。

比如 wafer map 缺陷分类：

```text
center defect
edge defect
ring defect
scratch defect
random defect
no defect
```

每张图只能分到一个主要类别。

---

## 3.3 多标签 Multi-label Classification

一个样本可以同时属于多个类别。

比如一片外延片可能同时存在：

```text
PL 异常
XRD 异常
表面缺陷偏多
```

这就是多标签。

对应标签可以长这样：

|wafer_id|pl_fail|xrd_fail|surface_fail|
|---|--:|--:|--:|
|W001|0|0|0|
|W002|1|0|1|
|W003|0|1|0|

外延片质量问题里，多标签其实很常见。

但第一阶段不要太复杂，先做二分类：

```text
Pass / Fail
```

---

# 4. 分类模型输出的不是“真相”，而是概率

这一点非常重要。

模型通常不是直接说：

```text
Fail
```

而是输出：

```text
Fail 概率 = 0.72
```

也就是：

```text
在历史数据规律下，这批片看起来像 Fail 的概率是 72%
```

所以模型结果应该这样用：

```text
Fail 概率 0.00 ~ 0.20：低风险
Fail 概率 0.20 ~ 0.50：中风险，建议复核
Fail 概率 0.50 ~ 1.00：高风险，重点检查
```

注意：  
`0.72` 不是说它一定坏，而是说它**值得工程师重点关注**。

---

# 5. 分类模型怎么学？

你可以把模型理解成：

```text
它在历史数据里找规律：
哪些特征组合更像 Pass？
哪些特征组合更像 Fail？
```

例如：

```text
Pass 样本常见特征：
temp_std 小
pressure_std 小
alarm_count 少
days_since_maintenance 短

Fail 样本常见特征：
temp_std 大
pressure_std 大
alarm_count 多
source_usage_hours 长
```

模型学到这些关系后，未来来一条新数据：

```text
temp_std = 4.2
pressure_std = 0.15
alarm_count = 3
source_usage_hours = 850
```

模型可能输出：

```text
Fail 概率 = 0.76
```

---

# 6. 第一版应该用什么分类模型？

外延片质量预测第一版，我建议按这个顺序学。

## 6.1 Logistic Regression：线性基线

它适合做第一个 baseline。

优点：

```text
简单
快
容易解释
能看出是否有基本信号
```

缺点：

```text
表达能力有限
不适合复杂非线性关系
```

半导体制造数据一般不是简单线性，所以它通常不是最终模型。

但它适合回答：

```text
最简单模型能不能看到一点规律？
```

---

## 6.2 Random Forest：非线性基线

随机森林比逻辑回归强一些。

优点：

```text
能处理非线性
不太需要特征缩放
比较稳
```

缺点：

```text
在高维、复杂表格数据上不一定最强
解释性一般
```

它适合做第二个基线。

---

## 6.3 LightGBM / XGBoost / CatBoost：主力模型

这是第一版真正推荐的主力。

外延片数据通常是：

```text
表格数据
数值特征多
类别字段多
非线性关系多
样本量可能不算特别大
```

所以第一版最合适的是：

```text
CatBoost
LightGBM
XGBoost
```

我的建议：

```text
类别字段很多：CatBoost 优先
数值字段很多：LightGBM 优先
想做稳健对比：三个都跑一遍
```

第一版不要急着上深度学习。

因为：

```text
如果只是表格数据，深度学习未必更强；
而且更难解释、更难上线、更吃数据量。
```

---

# 7. 分类问题最重要的不是 Accuracy

这是今天最重要的内容。

很多人一开始只看：

```text
accuracy = 准确率
```

但在半导体质量预测里，accuracy 很容易骗人。

---

## 举个例子

假设 1000 批外延片里：

```text
950 批 Pass
50 批 Fail
```

一个很蠢的模型永远预测：

```text
全部 Pass
```

它的 accuracy 是：

```text
950 / 1000 = 95%
```

看起来很高。

但它一个坏片都没抓出来。

所以这个模型在业务上是废的。

---

# 8. 混淆矩阵：分类问题必须看懂

我们把 **Fail 当成正类 Positive**。

因为我们真正关心的是：

```text
能不能抓出 Fail
```

混淆矩阵如下：

||实际 Fail|实际 Pass|
|---|--:|--:|
|预测 Fail|TP|FP|
|预测 Pass|FN|TN|

解释：

```text
TP：真实 Fail，模型也预测 Fail，抓对了坏片
FP：真实 Pass，模型预测 Fail，误报
FN：真实 Fail，模型预测 Pass，漏掉坏片
TN：真实 Pass，模型也预测 Pass，正常放行
```

在半导体场景里，最危险的是：

```text
FN = 漏报坏片
```

因为坏片流出去，可能导致：

```text
客户投诉
退货
失去信任
后续芯片加工浪费
```

---

# 9. Precision、Recall、F1 怎么理解？

## 9.1 Recall：坏片抓出来多少？

```text
Recall = TP / (TP + FN)
```

意思是：

```text
所有真实 Fail 里面，模型抓出来多少？
```

比如：

```text
真实 Fail 有 50 批
模型抓出来 40 批
Recall = 40 / 50 = 80%
```

在质量预警里，Recall 很重要。

因为你希望尽量不要漏掉坏片。

---

## 9.2 Precision：报警里面有多少是真的？

```text
Precision = TP / (TP + FP)
```

意思是：

```text
模型报警的批次里面，有多少真的 Fail？
```

比如：

```text
模型报警 100 批
其中真实 Fail 40 批
Precision = 40 / 100 = 40%
```

Precision 太低，工程师会被大量误报烦死。

---

## 9.3 F1：Precision 和 Recall 的折中

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

它综合考虑：

```text
抓得多不多
报警准不准
```

---

# 10. 半导体里应该优先看哪个指标？

我的建议：

```text
第一优先级：Fail Recall
第二优先级：Top-K Recall
第三优先级：Precision
第四优先级：PR-AUC
第五优先级：Accuracy
```

为什么？

因为第一版模型的目标不是自动判废，而是：

```text
把最值得复核的高风险批次挑出来
```

所以你更应该看：

```text
风险最高的前 10% 批次里，覆盖了多少真实 Fail？
```

这就是 Top-K Recall。

---

# 11. Top-K Recall 是什么？

假设有 1000 批外延片，其中 50 批真实 Fail。

模型给每批片一个 Fail 风险分。

我们按风险从高到低排序，取前 10%，也就是 100 批。

如果这 100 批里包含了 35 批真实 Fail，那么：

```text
Top 10% Recall = 35 / 50 = 70%
```

这很有业务意义。

它等价于对老板说：

```text
只让工程师重点复核风险最高的 10% 批次，
就能提前覆盖 70% 的异常批次。
```

这句话比：

```text
模型 accuracy 是 95%
```

专业得多，也更有商业价值。

---

# 12. 阈值为什么很重要？

模型输出概率：

```text
Fail 概率 = 0.31
Fail 概率 = 0.52
Fail 概率 = 0.78
```

你要决定：

```text
超过多少算高风险？
```

很多默认分类阈值是：

```text
0.5
```

但在质量预警里，不一定用 0.5。

比如：

```text
Fail 概率 > 0.2：进入关注列表
Fail 概率 > 0.5：进入重点复核
Fail 概率 > 0.8：暂停出货，工程师强制复查
```

所以分类模型不是简单输出 Pass/Fail，而是配合业务动作做分层。

---

# 13. 阈值和业务动作要绑定

你可以这样设计：

|Fail 概率|风险等级|业务动作|
|--:|---|---|
|0.00 ~ 0.20|低风险|正常流程|
|0.20 ~ 0.50|中风险|增加一项检测或工程师抽查|
|0.50 ~ 0.80|高风险|优先复核 PL / XRD / 表面检测|
|0.80 ~ 1.00|极高风险|暂缓出货，工程师强制复查|

这样模型就不是一个孤立算法，而是嵌入业务流程。

---

# 14. 一个完整的小例子

假设测试集有 100 批外延片：

```text
真实 Fail：20 批
真实 Pass：80 批
```

模型预测结果：

```text
预测 Fail：30 批
其中真实 Fail：15 批
其中真实 Pass：15 批

预测 Pass：70 批
其中真实 Fail：5 批
其中真实 Pass：65 批
```

那么：

```text
TP = 15
FP = 15
FN = 5
TN = 65
```

Recall：

```text
15 / (15 + 5) = 75%
```

Precision：

```text
15 / (15 + 15) = 50%
```

Accuracy：

```text
(15 + 65) / 100 = 80%
```

业务解释：

```text
模型抓出了 75% 的异常批次。
但报警的 30 批里，只有一半是真的异常。
```

工程上这可能是可以接受的。

因为第一版是风险提醒系统，不是自动判废系统。

---

# 15. 分类建模最小代码

今天你不需要背代码，但要看懂流程。

```python
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve
)

from catboost import CatBoostClassifier

# 1. 读取数据
df = pd.read_csv("epi_quality.csv")

# 2. 定义标签
target = "qc_result"
y = df[target].map({
    "Pass": 0,
    "Fail": 1
})

# 3. 根据预测时间点删除不能用的字段
drop_cols = [
    "qc_result",
    "pl_intensity",
    "xrd_fwhm",
    "surface_defect_count",
    "engineer_final_comment",
    "customer_feedback",
    "shipment_decision",
]

X = df.drop(columns=drop_cols)

# 4. 删除纯 ID 字段
id_cols = ["lot_id", "wafer_id", "run_id"]
X = X.drop(columns=id_cols)

# 5. 类别特征
cat_features = [
    "product_type",
    "recipe_id",
    "equipment_id",
    "chamber_id",
    "substrate_batch",
    "source_material_batch",
]

# 6. 按时间切分
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# 7. 训练分类模型
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

# 8. 输出 Fail 概率
fail_proba = model.predict_proba(X_test)[:, 1]

# 9. 设置风险阈值
threshold = 0.3
y_pred = (fail_proba >= threshold).astype(int)

# 10. 评估
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("PR-AUC:", average_precision_score(y_test, fail_proba))
```

这段代码里你重点看 4 个东西：

```text
y = Pass/Fail 标签
drop_cols = 防止数据泄露
fail_proba = Fail 风险概率
threshold = 风险阈值
```

---

# 16. Top-K Recall 代码

这个指标很适合工业质量预警。

```python
import numpy as np

def top_k_recall(y_true, y_score, k_ratio=0.1):
    """
    y_true: 真实标签，Fail=1，Pass=0
    y_score: 模型输出的 Fail 概率
    k_ratio: 取风险最高的前多少比例
    """
    n = len(y_true)
    k = int(n * k_ratio)

    # 按风险分从高到低排序
    order = np.argsort(-y_score)

    # 取风险最高的前 k 个
    top_k_idx = order[:k]

    # 总 Fail 数
    total_fail = np.sum(y_true == 1)

    # Top-K 里面命中的 Fail 数
    hit_fail = np.sum(y_true.iloc[top_k_idx] == 1)

    return hit_fail / total_fail


recall_top_10 = top_k_recall(y_test, fail_proba, k_ratio=0.1)

print("Top 10% Recall:", recall_top_10)
```

业务解释：

```text
风险最高的前 10% 批次，覆盖了多少真实 Fail。
```

这非常适合跟老板讲。

---

# 17. 分类模型最终输出不应该只是 0/1

不要让系统只输出：

```text
Pass
Fail
```

更好的输出是：

```text
批次：L2026-001
Fail 概率：0.68
风险等级：高
建议动作：优先复核
```

再加上解释：

```text
主要风险因子：
1. temp_std 高于同 recipe 历史 95 分位
2. pressure_std 明显偏高
3. alarm_count 高于近期正常水平
4. source_usage_hours 偏长
5. days_since_maintenance 偏长
```

这样工程师才会觉得模型有用。

---

# 18. 分类问题最常见的 5 个坑

## 坑 1：只看 Accuracy

制造业数据通常类别不平衡，只看 Accuracy 很危险。

---

## 坑 2：Fail 样本太少还硬做分类

如果 Fail 样本只有十几个，模型很难学到稳定规律。

这时可以先做：

```text
异常检测
质量分预测
单项指标回归
工程师风险等级标注
```

---

## 坑 3：预测时间点不清楚

不知道模型在什么时候预测，就不知道哪些字段能用。

这会直接导致数据泄露。

---

## 坑 4：把检测后字段放进输入

例如：

```text
pl_intensity
xrd_fwhm
final_qc_result
shipment_decision
```

如果预测时间点是检测前，这些都不能用。

---

## 坑 5：模型没有业务动作

模型说：

```text
Fail 概率 0.63
```

但没有告诉工程师该做什么。

这种模型很难落地。

必须配合：

```text
风险等级
复核建议
优先检测项
相似历史案例
```

---

# 19. 你今天要形成的专业表达

以后你跟客户讲分类模型，不要说：

```text
我们可以训练一个模型判断好坏。
```

你要说：

```text
我们会先把质量预测定义成一个二分类风险评分问题。
模型不是直接替代工程师判废，而是输出每批外延片的 Fail 概率。
然后按照风险分层，把高风险批次推送给工程师优先复核。
评估上我们不会只看准确率，而会重点看 Fail Recall、Top-K Recall 和误报成本。
```

这段话就是专业咨询表达。

---

# 20. Day 3 课堂练习

## 练习 1：判断分类类型

### 问题 A

```text
预测外延片 Pass / Fail
```

答案：

```text
二分类
```

---

### 问题 B

```text
判断 wafer map 是中心缺陷、边缘缺陷、环形缺陷还是划痕缺陷
```

答案：

```text
多分类
```

---

### 问题 C

```text
判断一片外延片是否同时存在 PL 异常、XRD 异常、表面缺陷异常
```

答案：

```text
多标签分类
```

---

## 练习 2：手算混淆矩阵指标

假设：

```text
TP = 30
FP = 20
FN = 10
TN = 140
```

那么：

```text
Recall = TP / (TP + FN) = 30 / 40 = 75%
Precision = TP / (TP + FP) = 30 / 50 = 60%
Accuracy = (TP + TN) / 总数 = 170 / 200 = 85%
```

业务解释：

```text
模型抓出了 75% 的真实异常批次。
报警的批次里，有 60% 真的异常。
整体准确率 85%，但不能只看这个指标。
```

---

# 21. Day 3 作业

今天作业有 3 个。

---

## 作业 1：定义一个分类任务

按照这个格式写：

```text
任务名称：
分类类型：二分类 / 多分类 / 多标签
正类是什么：
输入 X：
输出 y：
预测时间点：
业务动作：
```

示例：

```text
任务名称：
MBE 生长后外延片 Fail 风险预测

分类类型：
二分类

正类：
Fail

输入 X：
recipe_id
equipment_id
chamber_id
substrate_batch
source_material_batch
growth_temp_mean
growth_temp_std
pressure_mean
pressure_std
alarm_count
days_since_maintenance

输出 y：
qc_result = Pass / Fail

预测时间点：
MBE 生长完成后，检测前

业务动作：
Fail 概率超过 0.3 的 run 进入工程师复核列表
```

---

## 作业 2：设计风险阈值

你自己设计一套阈值。

格式：

```text
Fail 概率 0.00 ~ ?
风险等级：
业务动作：

Fail 概率 ? ~ ?
风险等级：
业务动作：

Fail 概率 ? ~ ?
风险等级：
业务动作：
```

参考：

```text
0.00 ~ 0.20：低风险，正常流程
0.20 ~ 0.50：中风险，增加抽检
0.50 ~ 0.80：高风险，工程师优先复核
0.80 ~ 1.00：极高风险，暂缓出货
```

---

## 作业 3：解释分类指标

用自己的话解释：

```text
Accuracy：
Precision：
Recall：
F1：
Top-K Recall：
```

重点解释：

```text
为什么外延片质量预测不能只看 Accuracy？
为什么 Fail Recall 很重要？
为什么 Top-K Recall 适合老板听？
```

---

# 22. Day 3 验收标准

你今天合格的标准是：

```text
1. 能解释什么是分类问题
2. 能区分二分类、多分类、多标签
3. 能说清楚为什么 Fail 是正类
4. 能看懂 TP / FP / FN / TN
5. 能解释 Precision 和 Recall
6. 能说明为什么 Accuracy 不够
7. 能理解阈值不是固定 0.5
8. 能把模型输出变成业务动作
```

---

# 23. Day 3 最核心总结

今天只记住这句话：

> **外延片 Pass / Fail 预测不是为了让模型替工程师判废，而是让模型给每批片打一个 Fail 风险分，再把高风险批次优先交给工程师复核。**

所以分类问题的核心不是：

```text
模型准确率多高
```

而是：

```text
坏片抓出来多少
误报成本能不能接受
风险最高的批次是否值得复核
模型结果能不能变成业务动作
```