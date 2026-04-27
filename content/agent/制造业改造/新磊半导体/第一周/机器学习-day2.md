# Day 2：机器学习最小闭环

今天的目标只有一个：

> 你要知道一份外延片数据拿到手后，怎么变成机器学习可以训练的东西。

Day 1 我们讲的是：

```text
业务场景 → AI 任务类型
```

Day 2 开始讲：

```text
数据 → 特征 X → 标签 y → 训练集 → 测试集 → 模型 → 评估
```

---

# 1. 今天先记住一句话

机器学习不是“让模型自己理解业务”。

机器学习的本质是：

```text
给模型很多历史样本：
每个样本都有输入 X 和答案 y。

模型学习：
X 和 y 之间的关系。

未来来了一个新的 X，
模型预测它可能对应的 y。
```

放到新磊外延片质量预测里就是：

```text
历史外延片数据：
X = 工艺参数 + 设备参数 + 衬底/材料信息
y = 这批外延片最终 Pass / Fail

模型学习：
什么样的工艺状态更容易 Fail

未来新的一批外延片：
输入新的工艺数据
输出质量风险
```

---

# 2. 什么是 X？

`X` 就是模型的输入。

在外延片质量预测里，X 可以是这些东西：

```text
product_type
recipe_id
equipment_id
chamber_id
substrate_batch
source_material_batch

growth_temp_mean
growth_temp_std
pressure_mean
pressure_std
flux_mean
flux_std
growth_time
alarm_count
days_since_maintenance
source_usage_hours
```

你可以把 X 理解成：

> 这批外延片在生产过程中发生过什么。

例如：

|wafer_id|recipe_id|equipment_id|temp_mean|temp_std|pressure_std|alarm_count|
|---|---|---|--:|--:|--:|--:|
|W001|R_HEMT_01|MBE_01|610|1.2|0.04|0|
|W002|R_HEMT_01|MBE_01|613|3.8|0.12|2|
|W003|R_VCSEL_03|MBE_02|585|1.5|0.05|0|

这些都是特征。

---

# 3. 什么是 y？

`y` 是模型要预测的答案。

在场景 1 里，最简单的 y 是：

```text
qc_result = Pass / Fail
```

例如：

|wafer_id|qc_result|
|---|---|
|W001|Pass|
|W002|Fail|
|W003|Pass|

也可以更细：

```text
pl_fail = 是否 PL 异常
xrd_fail = 是否 XRD 异常
thickness_fail = 是否厚度异常
surface_fail = 是否表面缺陷异常
```

第一阶段建议你先做：

```text
y = qc_result
```

也就是：

```text
Pass / Fail 二分类问题
```

---

# 4. 一行数据到底代表什么？

这个非常重要。

在工业 AI 里，你必须先定义：

```text
一行数据 = 什么？
```

外延片质量预测可以有 3 种粒度：

## 粒度 1：一行 = 一片 wafer

适合预测：

```text
每一片外延片是否合格
```

示例：

|wafer_id|run_id|recipe_id|temp_std|pl_intensity|qc_result|
|---|---|---|--:|--:|---|
|W001|RUN001|R01|1.2|95|Pass|
|W002|RUN001|R01|1.3|91|Pass|
|W003|RUN002|R01|3.8|62|Fail|

---

## 粒度 2：一行 = 一次 run

适合预测：

```text
这次 MBE 生长任务整体是否有风险
```

示例：

|run_id|recipe_id|equipment_id|temp_std|pressure_std|fail_rate|
|---|---|---|--:|--:|--:|
|RUN001|R01|MBE01|1.2|0.03|0.02|
|RUN002|R01|MBE01|3.8|0.12|0.35|

---

## 粒度 3：一行 = 一个 lot

适合预测：

```text
这一批出货是否有风险
```

示例：

|lot_id|product_type|recipe_id|avg_temp_std|defect_count|qc_result|
|---|---|---|--:|--:|---|
|L001|HEMT|R01|1.5|3|Pass|
|L002|HEMT|R01|3.2|18|Fail|

---

我的建议：

> 第一版优先用 **一行 = wafer** 或 **一行 = run**。

如果真实数据管理比较粗，就先用 run 级别；如果每片检测数据完整，就用 wafer 级别。

---

# 5. 特征 feature 是什么？

特征就是 X 里的每一列。

例如：

```text
growth_temp_mean
growth_temp_std
pressure_mean
pressure_std
alarm_count
days_since_maintenance
```

每个特征都应该回答一个问题：

> 这个字段有没有可能影响外延片质量？

比如：

```text
growth_temp_mean
```

代表平均生长温度。

```text
growth_temp_std
```

代表温度波动。

这两个都可能影响外延层质量。

再比如：

```text
days_since_maintenance
```

代表距离上次设备维护多少天。

这可能影响设备稳定性。

---

# 6. 标签 label 是什么？

标签就是 y。

在外延片质量预测里，标签可以来自：

```text
QC 判定
检测指标是否超规格
工程师复核结果
客户是否接受
客户投诉记录
```

最简单的标签定义：

```text
如果任一关键指标超规格 → Fail
否则 → Pass
```

例如：

```text
thickness_uniformity > 5% → Fail
PL intensity < 70 → Fail
XRD FWHM > 55 → Fail
surface_defect_count > 12 → Fail
```

这样可以得到：

```text
qc_result = Pass / Fail
```

---

# 7. 今天最重要的坑：数据泄露

数据泄露就是：

> 你把预测时还不知道的信息，提前塞给了模型。

比如你想在“检测前”预测质量风险。

那你不能把这些字段作为输入：

```text
pl_intensity
xrd_fwhm
surface_defect_count
final_qc_result
engineer_final_comment
shipment_decision
customer_acceptance
```

因为这些字段是在检测之后、复核之后、出货之后才知道的。

如果你把它们放进 X，模型效果会非常好，但完全没用。

---

## 举个例子

你要预测：

```text
这批外延片是否 Fail
```

然后你把这个字段放进输入：

```text
final_qc_result
```

这就等于考试时把答案给模型了。

模型当然能预测得很好，但上线后没意义。

---

# 8. 所以必须先定义预测时间点

同一个“质量预测”，时间点不同，能用的数据不同。

## 时间点 A：生长前预测

可用数据：

```text
recipe_id
equipment_id
chamber_id
substrate_batch
source_material_batch
days_since_maintenance
```

不能用：

```text
生长过程曲线
检测结果
QC 结果
```

价值：

```text
提前识别高风险生产任务
```

难度：中

---

## 时间点 B：生长后、检测前预测

可用数据：

```text
recipe_id
equipment_id
chamber_id
生长温度统计
压力统计
束流统计
报警次数
设备状态
衬底批次
源材料批次
```

不能用：

```text
PL
XRD
AFM
最终 QC
客户反馈
```

价值：

```text
决定哪些批次优先检测、重点复核
```

难度：中

这个是我最建议第一版做的。

---

## 时间点 C：初检后、出货前预测

可用数据：

```text
工艺数据
设备数据
PL
XRD
AFM
厚度
表面检测
```

不能用：

```text
最终客户反馈
客户投诉
出货后结果
```

价值：

```text
判断出货风险
```

难度：中高

---

# 9. 训练集和测试集是什么？

机器学习不能只在历史数据上表现好。

你要把数据分成两部分：

```text
训练集：让模型学习规律
测试集：检验模型未来能不能用
```

比如你有 10000 条历史数据：

```text
前 8000 条 → 训练集
后 2000 条 → 测试集
```

模型只看训练集。

训练完之后，用测试集检验。

---

# 10. 为什么制造业不能随便随机切分？

很多教程会这样：

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

但在制造业里，最好不要一开始就随机切分。

因为真实场景是：

```text
用过去的数据预测未来的数据
```

所以更合理的是：

```text
2024 年数据训练
2025 年数据测试
```

或者：

```text
前 80% 时间的数据训练
后 20% 时间的数据测试
```

这叫：

```text
按时间切分
```

因为如果随机切分，可能会出现：

```text
同一个设备、同一个 recipe、同一个批次附近的数据同时出现在训练集和测试集
```

这样测试结果会偏乐观。

---

# 11. 最小机器学习流程

完整流程是：

```text
1. 拿到原始数据
2. 明确一行代表什么
3. 明确预测时间点
4. 定义 X
5. 定义 y
6. 删除泄露字段
7. 划分训练集 / 测试集
8. 训练模型
9. 评估效果
10. 解释结果
```

你现在先记住前 7 步。

模型本身不急。

---

# 12. 用外延片质量预测举一个完整例子

假设我们选择：

```text
预测时间点：生长后，检测前
任务：预测这次 run 是否会 Fail
```

那么：

## 可以用的 X

```text
product_type
recipe_id
equipment_id
chamber_id
substrate_batch
source_material_batch

growth_temp_mean
growth_temp_std
pressure_mean
pressure_std
flux_mean
flux_std
growth_time
alarm_count
days_since_maintenance
source_usage_hours
```

## 不能用的 X

```text
pl_intensity
xrd_fwhm
surface_defect_count
thickness_uniformity
final_qc_result
engineer_comment
customer_feedback
```

因为这些是在检测后或出货后才知道的。

## y

```text
qc_result
```

也就是：

```text
Pass / Fail
```

---

# 13. 最小代码示例

今天你不用真正跑，但你要看懂逻辑。

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier

# 1. 读取数据
df = pd.read_csv("epi_quality.csv")

# 2. 定义标签
target = "qc_result"

# 3. 删除不能作为输入的字段
drop_cols = [
    "qc_result",
    "pl_intensity",
    "xrd_fwhm",
    "surface_defect_count",
    "thickness_uniformity",
    "engineer_comment",
    "customer_feedback",
]

X = df.drop(columns=drop_cols)
y = df[target].map({
    "Pass": 0,
    "Fail": 1
})

# 4. 删除 ID 字段，或者保留部分可作为类别特征的字段
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

# 6. 按时间切分更好
# 这里先简单示意
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# 7. 训练模型
model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    verbose=50
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

# 8. 预测
proba = model.predict_proba(X_test)[:, 1]

# 9. 风险阈值
pred = (proba > 0.3).astype(int)

# 10. 评估
print(classification_report(y_test, pred))
```

你今天只需要看懂这几件事：

```text
X 是输入
y 是答案
drop_cols 是防止数据泄露
cat_features 是类别字段
train 是训练数据
test 是未来模拟数据
proba 是 Fail 风险概率
```

---

# 14. 为什么阈值不是 0.5？

默认分类经常是：

```text
Fail 概率 > 0.5 → 判定 Fail
```

但在半导体质量预警里，我们可能更怕漏掉坏片。

所以可以改成：

```text
Fail 概率 > 0.3 → 进入人工复核
```

这不是说它一定是坏片，而是：

```text
这批片值得工程师多看一眼
```

所以模型第一阶段不要做成：

```text
自动判废系统
```

而应该做成：

```text
风险提醒系统
```

这个认知非常重要。

---

# 15. 机器学习项目里最核心的 5 个判断

## 判断 1：这个字段在预测时间点是否存在？

如果不存在，不能用。

---

## 判断 2：这个字段是不是答案的变体？

比如：

```text
final_qc_result
quality_grade
shipment_decision
```

这些不能用。

---

## 判断 3：标签是否稳定？

如果今天工程师 A 判 Fail，明天工程师 B 判 Pass，标签就不稳定。

标签不稳定，模型也会乱。

---

## 判断 4：样本是否足够？

尤其是 Fail 样本。

如果 10000 条数据里只有 5 条 Fail，那二分类很难做。

---

## 判断 5：测试集是否模拟未来？

如果测试集太容易，模型上线后会翻车。

---

# 16. 今天你要形成的业务表达

你以后跟客户聊时，要这样说：

> 做外延片质量预测，第一步不是直接训练模型，而是先确定预测时间点。如果我们希望在生长后、检测前预警，那么输入只能使用生长过程参数、设备状态、recipe、衬底和源材料信息，不能使用 PL、XRD、最终 QC 等检测后信息。否则模型会发生数据泄露，离线效果很好，但真实上线无效。

这段话非常专业。

---

# 17. Day 2 课堂练习

下面你自己判断：如果我们的预测时间点是 **生长后、检测前**，哪些字段可以用，哪些不能用？

## 字段列表

```text
recipe_id
equipment_id
substrate_batch
growth_temp_mean
growth_temp_std
pressure_std
alarm_count
pl_intensity
xrd_fwhm
final_qc_result
customer_complaint
days_since_maintenance
source_usage_hours
engineer_final_comment
```

## 参考答案

可以用：

```text
recipe_id
equipment_id
substrate_batch
growth_temp_mean
growth_temp_std
pressure_std
alarm_count
days_since_maintenance
source_usage_hours
```

不能用：

```text
pl_intensity
xrd_fwhm
final_qc_result
customer_complaint
engineer_final_comment
```

原因：

```text
这些字段在检测后、出货后或工程师最终判定后才出现。
```

---

# 18. Day 2 作业

今天你做 3 个作业。

## 作业 1：定义一个建模任务

按照这个格式写：

```text
任务名称：
预测时间点：
一行数据代表：
X 包含哪些字段：
y 是什么：
不能使用哪些字段：
业务动作：
```

示例：

```text
任务名称：
MBE 生长后外延片质量风险预测

预测时间点：
MBE 生长完成后，PL/XRD/AFM 检测前

一行数据代表：
一次 MBE run

X 包含：
recipe_id
equipment_id
chamber_id
substrate_batch
source_material_batch
growth_temp_mean
growth_temp_std
pressure_mean
pressure_std
flux_mean
flux_std
alarm_count
days_since_maintenance

y：
qc_result，Pass / Fail

不能使用：
PL、XRD、AFM、最终 QC、客户反馈、工程师最终评语

业务动作：
高风险 run 进入优先检测和工程师复核
```

---

## 作业 2：列出 10 个可用特征

你自己列 10 个外延片质量预测可以用的特征。

格式：

```text
特征名：
含义：
为什么可能影响质量：
预测时间点是否可用：
```

例如：

```text
特征名：
growth_temp_std

含义：
生长过程温度标准差

为什么可能影响质量：
温度波动大可能导致外延层厚度、成分或界面质量不稳定

预测时间点是否可用：
如果预测时间点是生长后、检测前，则可用
```

---

## 作业 3：列出 5 个数据泄露字段

格式：

```text
字段名：
为什么不能用：
```

例如：

```text
字段名：
final_qc_result

为什么不能用：
它本身就是模型要预测的答案。
```

---

# 19. Day 2 验收标准

你今天学完，合格标准是：

```text
1. 能说清楚 X 是什么
2. 能说清楚 y 是什么
3. 能定义一行数据代表什么
4. 能区分特征和标签
5. 能判断一个字段是否数据泄露
6. 能解释为什么制造业要按时间切分训练集和测试集
7. 能说出预测时间点为什么重要
```

---

# 20. Day 2 最核心总结

你今天只需要牢牢记住这句话：

> **机器学习项目的第一步不是选模型，而是定义：在什么时间点，用哪些已知信息 X，预测哪个业务结果 y。**

对于外延片质量预测，最推荐的第一版任务是：

```text
在 MBE 生长完成后、检测前，
用 recipe、设备、衬底、源材料、生长过程统计特征，
预测该 run / wafer 的质量风险。
```

这个任务清晰、可落地、风险低，也最适合你后面继续学习建模。