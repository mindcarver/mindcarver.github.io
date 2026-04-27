# Day 5：异常检测 —— 没有 Fail 标签时，怎么发现 MBE 工艺/设备异常？

前 4 天我们讲了：

```text
Day 1：半导体业务问题怎么变成 AI 问题
Day 2：X、y、训练集、测试集、数据泄露
Day 3：分类问题，Pass / Fail 预测
Day 4：回归问题，预测 PL、XRD、厚度、缺陷数量等指标
```

今天讲第三类非常重要的问题：

```text
异常检测 Anomaly Detection
```

它特别适合半导体，因为很多时候企业没有足够干净的 Fail 标签。

---

# 1. 今天先记住一句话

> **分类模型回答：这批片像不像历史上的 Fail？  
> 异常检测回答：这次工艺/设备状态像不像历史上的正常状态？**

这是两种不同思路。

分类需要：

```text
大量 Pass 样本
大量 Fail 样本
明确标签 y
```

异常检测可以只基于：

```text
大量正常样本
少量甚至没有 Fail 样本
```

所以在真实半导体项目里，如果一开始没有足够 Fail 标签，**异常检测往往比分类更适合作为第一步**。

---

# 2. 为什么新磊这种外延片场景很适合异常检测？

MBE 外延生长是一个高度依赖稳定性的过程。

你不一定马上知道这批片最终会不会 Fail，但你可以先判断：

```text
这次生长过程是否不像正常生长过程？
这台设备最近是否不像历史正常状态？
这个 recipe 最近是否开始漂移？
某个参数组合是否异常？
```

外延质量很容易受到这些因素影响：

```text
温度波动
腔体压力波动
真空状态变化
源炉状态变化
束流漂移
快门时序异常
报警次数增加
维护周期过长
衬底批次变化
源材料使用时长过长
```

很多异常不是一个字段单独超标，而是多个字段组合起来“不对劲”。

例如：

```text
温度没有超规格
压力也没有超规格
束流也没有超规格

但三者组合起来，和历史正常 run 很不一样。
```

这就是异常检测的价值。

---

# 3. 异常检测和分类的区别

|对比项|分类|异常检测|
|---|---|---|
|是否需要 Fail 标签|需要|不一定需要|
|学习目标|区分 Pass / Fail|学习什么是正常|
|输出|Fail 概率|异常分数|
|适合阶段|有足够历史标签后|早期数据不足时|
|典型用途|质量预测|设备漂移、工艺异常、未知风险发现|
|最大价值|预测已知失败类型|发现未知异常模式|

你可以这样理解：

```text
分类模型：学习过去发生过的坏事。
异常检测：发现当前状态不像过去的正常状态。
```

这非常适合半导体。

因为很多真实异常一开始没有标签，工程师只是觉得：

```text
最近这台设备状态不太对。
最近这个 recipe 不太稳。
这批数据看起来怪怪的。
```

异常检测就是把这种“怪怪的”量化成分数。

---

# 4. 异常检测的输出是什么？

异常检测通常不直接输出：

```text
Pass / Fail
```

而是输出：

```text
anomaly_score = 异常分数
```

例如：

|run_id|anomaly_score|风险等级|
|---|--:|---|
|RUN001|0.05|正常|
|RUN002|0.18|轻微异常|
|RUN003|0.63|高异常|
|RUN004|0.91|严重异常|

然后你把异常分数转成业务动作：

|异常分数|风险等级|业务动作|
|--:|---|---|
|0.00 ~ 0.20|正常|正常流程|
|0.20 ~ 0.50|关注|工程师抽查|
|0.50 ~ 0.80|高异常|优先复核检测结果|
|0.80 ~ 1.00|严重异常|暂缓放行，检查设备/工艺|

注意：

```text
异常 ≠ 一定 Fail
```

异常只是说明：

```text
这次状态不像历史正常状态，值得工程师重点看。
```

---

# 5. 外延片场景里有哪些异常检测任务？

## 任务 1：MBE run 级异常检测

一行代表一次 MBE 生长任务。

输入：

```text
recipe_id
equipment_id
growth_temp_mean
growth_temp_std
pressure_mean
pressure_std
flux_mean
flux_std
alarm_count
days_since_maintenance
source_usage_hours
```

输出：

```text
这个 run 是否异常
异常分数是多少
哪些字段导致异常
```

业务价值：

```text
生长后、检测前，先把异常 run 标出来，优先检测和复核。
```

---

## 任务 2：设备状态异常检测

一行代表一台设备在某一天、某一周或某个窗口的状态。

输入：

```text
设备最近 N 次 run 的温度波动
压力波动
报警次数
Fail 比例
维护间隔
源炉状态
真空恢复时间
```

输出：

```text
设备健康分
是否需要维护
哪个设备风险最高
```

业务价值：

```text
从事后维修变成提前预警。
```

---

## 任务 3：recipe 漂移检测

同一个 recipe，理论上应该比较稳定。

如果最近几炉的输出开始变差，就要发现。

输入：

```text
同 recipe 最近 N 次 run 的 PL、XRD、厚度、缺陷数量分布
```

输出：

```text
这个 recipe 最近是否漂移
漂移方向是什么
从什么时候开始漂移
```

业务价值：

```text
发现工艺窗口变窄、设备变差、材料批次变化。
```

---

## 任务 4：检测指标异常检测

即使没有最终 Pass / Fail 标签，也可以判断某个检测指标是否异常。

例如：

```text
PL intensity 是否异常低
XRD FWHM 是否异常高
thickness_uniformity 是否异常差
surface_defect_count 是否异常多
```

业务价值：

```text
建立质量指标的自动预警系统。
```

---

# 6. 异常有几种类型？

## 6.1 单点异常

某个值突然非常离谱。

例如：

```text
某次 run 的 pressure_std 是历史平均的 5 倍。
```

适合方法：

```text
Z-score
IQR
3-sigma
```

---

## 6.2 上下文异常

单独看不异常，但在当前上下文里异常。

例如：

```text
温度 610℃ 对 HEMT recipe 正常，
但对 VCSEL recipe 可能异常。
```

所以不能所有产品混在一起看。

必须考虑：

```text
product_type
recipe_id
equipment_id
chamber_id
```

---

## 6.3 组合异常

每个字段单独看都正常，但组合起来异常。

例如：

```text
温度略高
压力略高
束流略波动
报警次数略多

每个都没超标，
但组合起来就是不像正常 run。
```

适合方法：

```text
Isolation Forest
One-Class SVM
AutoEncoder
```

---

## 6.4 漂移异常

不是突然坏，而是慢慢变差。

例如：

```text
同一台 MBE 设备最近 30 天 pressure_std 慢慢升高。
同一个 recipe 最近 20 炉 PL 均值逐渐下降。
```

适合方法：

```text
移动平均
EWMA
Control Chart
PSI
KS Test
趋势检测
```

---

# 7. 最简单的异常检测：Z-score

Z-score 适合判断一个数值是否偏离历史正常水平。

例如：

```text
growth_temp_std = 4.8
历史平均 temp_std = 1.5
历史标准差 = 0.8
```

Z-score 公式是：

genui{"learning_block_v3":{"content":"z=\frac{x-\mu}{\sigma}"}}

直觉解释：

```text
当前值距离历史平均值有几个标准差。
```

如果：

```text
|z| > 3
```

通常可以认为非常异常。

---

## 用外延片例子解释

假设：

```text
历史正常 run 的 temp_std 平均值 = 1.5
历史正常 run 的 temp_std 标准差 = 0.7
当前 run 的 temp_std = 4.0
```

那么：

```text
z = (4.0 - 1.5) / 0.7 = 3.57
```

这说明：

```text
当前温度波动远高于历史正常水平。
```

业务动作：

```text
标记为温度稳定性异常；
建议复查该 run 的温控曲线和相关检测结果。
```

---

# 8. Z-score 的局限

Z-score 很简单，但有几个问题。

## 问题 1：假设数据大致稳定

如果数据分布本身不稳定，Z-score 会误报。

---

## 问题 2：容易受极端值影响

如果历史数据里已经有异常值，平均值和标准差会被拉偏。

---

## 问题 3：只能看单个变量

它很难识别组合异常。

例如：

```text
temp_std 不算异常
pressure_std 不算异常
flux_std 不算异常

但三者组合起来异常。
```

这时候要用机器学习异常检测。

---

# 9. IQR：更稳健的单变量异常检测

IQR 适合数据有极端值时使用。

核心是看四分位数：

```text
Q1 = 25% 分位数
Q3 = 75% 分位数
IQR = Q3 - Q1
```

通常定义：

```text
低于 Q1 - 1.5 * IQR → 异常低
高于 Q3 + 1.5 * IQR → 异常高
```

适合检测：

```text
pressure_std 是否异常高
alarm_count 是否异常多
surface_defect_count 是否异常多
xrd_fwhm 是否异常高
```

---

# 10. Control Chart：工业现场很常用的思想

控制图的核心思想是：

```text
只要工艺稳定，指标应该在一个正常波动范围内。
一旦指标超出控制限，或者连续偏向一侧，就说明过程可能失控。
```

例如监控：

```text
同一 recipe 的 PL intensity
同一设备的 pressure_std
同一 chamber 的 alarm_count
同一源炉的 flux_std
```

你可以设置：

```text
中心线：历史均值
上控制限：均值 + 3σ
下控制限：均值 - 3σ
```

业务解释：

```text
不是等最终 Fail 才报警，
而是当过程指标开始偏离正常控制范围时就提醒。
```

---

# 11. Isolation Forest：第一版最推荐的机器学习异常检测模型

如果你现在有表格数据，但 Fail 标签不多，我建议第一版用：

```text
Isolation Forest
```

它适合：

```text
多变量异常检测
无标签数据
表格数据
快速建立 baseline
```

它的直觉是：

```text
异常点通常更容易被孤立出来。
```

比如一个 run 的特征组合是：

```text
temp_std 高
pressure_std 高
alarm_count 多
source_usage_hours 长
days_since_maintenance 长
```

即使每个字段不一定单独超标，整体也可能被 Isolation Forest 判为异常。

---

# 12. One-Class SVM：可以用，但第一版不一定优先

One-Class SVM 的思路是：

```text
用正常样本学习一个正常边界，
边界外的点就是异常。
```

适合：

```text
样本量不太大
特征维度不是特别高
数据经过良好标准化
```

缺点：

```text
对参数和特征缩放敏感
数据大时训练可能慢
解释性不如树模型直观
```

所以第一版我会把它作为对照模型，不作为主力。

---

# 13. AutoEncoder：适合更复杂的异常检测

AutoEncoder 的思路是：

```text
用正常数据训练一个模型，让它学会重构正常样本。
如果某个样本重构误差很大，就说明它不像正常样本。
```

适合：

```text
高维数据
时间序列
图像
复杂非线性模式
```

例如：

```text
MBE 生长过程完整曲线
RHEED 图像/视频
PL mapping 图
wafer map 图
```

但第一版不要急着上 AutoEncoder，除非你已经有大量过程曲线或图像数据。

---

# 14. 不同数据形态对应的方法

|数据形态|例子|推荐方法|
|---|---|---|
|单个指标|temp_std、pressure_std|Z-score / IQR|
|多个 run 级特征|温度、压力、束流、报警数|Isolation Forest|
|稳定过程监控|同 recipe 的 PL 趋势|Control Chart / EWMA|
|多变量时间序列|MBE 生长曲线|AutoEncoder / LSTM AutoEncoder|
|图像|表面图、wafer map|CNN AutoEncoder / PatchCore|
|没有标签但有正常样本|历史稳定 run|One-Class SVM / Isolation Forest|

第一版建议：

```text
Z-score + IQR + Isolation Forest
```

这三种足够做一个可解释的早期异常检测系统。

---

# 15. 外延片异常检测第一版怎么做？

我建议你定义这个任务：

```text
任务名称：
MBE 生长后 run 级异常检测

预测时间点：
MBE 生长完成后，PL/XRD/AFM 检测前

一行数据代表：
一次 MBE run

输入 X：
recipe_id
equipment_id
chamber_id
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

输出：
anomaly_score
risk_level
top_anomaly_features

业务动作：
异常分数高的 run 优先进入检测和工程师复核。
```

---

# 16. 但这里有一个大坑：不能把不同 recipe 混在一起直接比

比如：

```text
HEMT 的正常温度范围
VCSEL 的正常温度范围
APD 的正常温度范围
```

可能完全不同。

如果你把所有产品、所有 recipe 混在一起做异常检测，模型可能会把正常差异误判成异常。

例如：

```text
R01 recipe 正常 temp = 610℃
R02 recipe 正常 temp = 580℃

如果不区分 recipe，
R02 的 580℃ 可能被误判为异常低。
```

所以要做两件事：

## 方法 1：分组建模

按下面维度分开：

```text
product_type
recipe_id
equipment_id
```

例如：

```text
每个主要 recipe 单独建立正常分布。
```

优点：

```text
更符合工艺逻辑。
```

缺点：

```text
每组样本可能不够。
```

---

## 方法 2：做相对偏差特征

不要直接用：

```text
actual_temp
```

而是用：

```text
temp_deviation = actual_temp - target_temp
```

不要直接用：

```text
pressure_std
```

而是用：

```text
pressure_std_relative_to_recipe
```

这更适合跨 recipe 建模。

---

# 17. 异常检测特征怎么设计？

异常检测很依赖特征工程。

## 17.1 基础统计特征

```text
growth_temp_mean
growth_temp_std
growth_temp_min
growth_temp_max
growth_temp_range

pressure_mean
pressure_std
pressure_min
pressure_max
pressure_spike_count

flux_mean
flux_std
flux_drift
alarm_count
growth_time
```

---

## 17.2 相对 recipe 偏差特征

```text
temp_mean_dev = actual_temp_mean - target_temp
growth_time_dev = actual_growth_time - target_growth_time
pressure_mean_ratio = actual_pressure_mean / target_pressure
flux_dev = actual_flux - target_flux
```

这类特征非常重要。

因为工艺里更关心：

```text
实际执行和目标 recipe 偏了多少。
```

---

## 17.3 稳定性特征

```text
temp_std
pressure_std
flux_std
temp_spike_count
pressure_spike_count
max_deviation_from_setpoint
time_out_of_control_range
```

稳定性往往比平均值更重要。

---

## 17.4 设备状态特征

```text
days_since_maintenance
recent_alarm_count_7d
recent_fail_rate_30d
equipment_recent_anomaly_rate
source_usage_hours
chamber_usage_hours
```

这些特征反映设备和材料状态。

---

## 17.5 历史窗口特征

```text
同一设备最近 10 次 temp_std 均值
同一设备最近 10 次 pressure_std 均值
同一 recipe 最近 20 次 PL 均值
同一 chamber 最近 30 天报警次数
```

这类特征用于发现慢性漂移。

---

# 18. 异常检测代码示例 1：Z-score

假设你有一个 `epi_runs.csv`：

```python
import pandas as pd
import numpy as np

df = pd.read_csv("epi_runs.csv")

feature = "growth_temp_std"

mu = df[feature].mean()
sigma = df[feature].std()

df[f"{feature}_zscore"] = (df[feature] - mu) / sigma

df[f"{feature}_is_anomaly"] = df[f"{feature}_zscore"].abs() > 3

print(df[["run_id", feature, f"{feature}_zscore", f"{feature}_is_anomaly"]].head())
```

业务解释：

```text
如果 growth_temp_std 的 z-score 绝对值大于 3，
说明这次温度波动显著偏离历史正常水平。
```

---

# 19. 异常检测代码示例 2：按 recipe 做 Z-score

这比全局 Z-score 更合理。

```python
import pandas as pd

df = pd.read_csv("epi_runs.csv")

feature = "growth_temp_std"

df[f"{feature}_recipe_mean"] = df.groupby("recipe_id")[feature].transform("mean")
df[f"{feature}_recipe_std"] = df.groupby("recipe_id")[feature].transform("std")

df[f"{feature}_recipe_zscore"] = (
    df[feature] - df[f"{feature}_recipe_mean"]
) / df[f"{feature}_recipe_std"]

df[f"{feature}_recipe_anomaly"] = df[f"{feature}_recipe_zscore"].abs() > 3

print(
    df[
        [
            "run_id",
            "recipe_id",
            feature,
            f"{feature}_recipe_zscore",
            f"{feature}_recipe_anomaly",
        ]
    ].head()
)
```

这个更符合半导体业务。

因为它问的是：

```text
当前 run 相对于同一个 recipe 的历史正常状态是否异常？
```

而不是相对于所有产品整体是否异常。

---

# 20. 异常检测代码示例 3：Isolation Forest

这是第一版主力。

```python
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("epi_runs.csv")

features = [
    "temp_mean_dev",
    "growth_temp_std",
    "pressure_mean_ratio",
    "pressure_std",
    "flux_std",
    "alarm_count",
    "days_since_maintenance",
    "source_usage_hours",
]

X = df[features].copy()

# 缺失值简单处理
X = X.fillna(X.median(numeric_only=True))

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=300,
    contamination=0.05,  # 假设约 5% 样本异常
    random_state=42
)

model.fit(X_scaled)

# sklearn 输出：-1 表示异常，1 表示正常
df["iforest_label"] = model.predict(X_scaled)

# decision_function 越低越异常，这里转成越高越异常
raw_score = model.decision_function(X_scaled)
df["anomaly_score"] = -raw_score

df["is_anomaly"] = df["iforest_label"] == -1

print(
    df[
        [
            "run_id",
            "recipe_id",
            "equipment_id",
            "anomaly_score",
            "is_anomaly",
        ]
    ].sort_values("anomaly_score", ascending=False).head(20)
)
```

你今天要看懂：

```text
features = 用哪些字段判断异常
contamination = 预期异常比例
anomaly_score = 异常分数
is_anomaly = 是否异常
```

---

# 21. contamination 参数怎么设？

`contamination` 表示你预期有多少比例样本是异常。

比如：

```text
contamination=0.01 → 认为约 1% 样本异常
contamination=0.05 → 认为约 5% 样本异常
contamination=0.10 → 认为约 10% 样本异常
```

第一版可以从：

```text
0.03 ~ 0.10
```

之间试。

但注意：

```text
这个不是物理真理，而是报警阈值。
```

它应该根据工程师能接受的复核量来调。

例如：

```text
每天 100 个 run，
如果 contamination=0.05，
每天大概标记 5 个 run 重点复核。
```

这就很容易跟业务动作结合。

---

# 22. 异常检测怎么解释？

Isolation Forest 不像分类模型那样直接有很自然的概率解释。

第一版可以用两种方式解释。

## 方法 1：看异常样本的字段偏离

例如某个 run 被判异常，你输出：

```text
growth_temp_std：高于同 recipe 历史 98 分位
pressure_std：高于同设备历史 95 分位
alarm_count：高于历史 99 分位
source_usage_hours：高于历史 90 分位
```

这对工程师很好理解。

---

## 方法 2：用“规则 + 模型”混合解释

模型发现异常后，再用规则解释：

```text
异常原因候选：
1. 温度波动异常
2. 压力波动异常
3. 报警次数异常
4. 维护周期过长
5. 源材料使用时长偏长
```

这样比直接说：

```text
Isolation Forest 认为异常
```

更容易落地。

---

# 23. 异常检测输出示例

一个好的输出不是：

```text
异常：True
```

而是：

```text
Run：RUN2026-00427-008
产品：HEMT
Recipe：R-HEMT-017
设备：MBE-03

异常分数：0.87
风险等级：严重异常

主要异常点：
1. growth_temp_std 高于同 recipe 历史 98 分位
2. pressure_std 高于同设备历史 96 分位
3. alarm_count = 4，高于近期正常水平
4. source_usage_hours 接近上限
5. days_since_maintenance = 92 天，偏长

建议动作：
1. 优先复查 PL mapping
2. 复查 XRD FWHM
3. 查看 MBE-03 近期温控和真空日志
4. 检查源炉状态
5. 暂缓直接放行，进入工程师复核
```

这才是能给客户看的东西。

---

# 24. 异常检测和分类/回归怎么组合？

最实用的组合是：

```text
异常检测：发现这次 run 是否不像正常
分类模型：预测这次是否可能 Fail
回归模型：预测具体哪个质量指标可能异常
```

最终输出：

```text
异常分数：0.87，高
Fail 概率：0.68，高
预测 PL：68，低于规格线
预测 XRD：正常
预测表面缺陷：偏高
```

综合判断：

```text
这批片整体风险高，主要风险来自生长过程异常、PL 偏低、表面缺陷偏多。
```

这比单个模型更有价值。

---

# 25. 异常检测最适合项目早期

如果客户说：

```text
我们没有系统整理过 Pass / Fail 标签。
```

你不要卡住。

你可以说：

```text
没关系，我们可以先从正常工艺状态建模开始。
先用历史稳定 run 建立正常基线，做设备和工艺异常检测。
等后续质量标签逐渐积累，再升级为 Pass / Fail 预测和关键指标预测。
```

这就是正确路线。

---

# 26. Day 5 课堂练习

## 练习 1：判断适合分类还是异常检测

### 问题 A

```text
我们有 5000 条历史数据，其中 800 条明确 Fail。
想预测新 run 是否 Fail。
```

答案：

```text
适合分类。
```

---

### 问题 B

```text
我们只有大量正常生产数据，很少记录 Fail 标签。
想知道某次生长过程是否不正常。
```

答案：

```text
适合异常检测。
```

---

### 问题 C

```text
同一个 recipe 最近 20 炉 PL 均值持续下降。
```

答案：

```text
适合漂移检测 / 控制图 / 时间序列趋势检测。
```

---

### 问题 D

```text
模型需要判断 wafer map 是中心缺陷还是边缘缺陷。
```

答案：

```text
这是图像多分类，不是异常检测的第一优先场景。
```

---

# 27. 练习 2：判断哪个字段异常

假设同一 recipe 的历史统计如下：

```text
growth_temp_std 平均 = 1.5，标准差 = 0.6
pressure_std 平均 = 0.04，标准差 = 0.02
alarm_count 平均 = 0.3，标准差 = 0.5
```

当前 run：

```text
growth_temp_std = 3.6
pressure_std = 0.11
alarm_count = 3
```

你大致判断：

```text
growth_temp_std 明显异常
pressure_std 明显异常
alarm_count 明显偏高
```

业务结论：

```text
该 run 的工艺稳定性风险较高，建议优先复核检测结果。
```

---

# 28. Day 5 作业

今天作业有 3 个。

---

## 作业 1：定义一个异常检测任务

按照这个模板写：

```text
任务名称：
检测对象：
一行数据代表：
输入 X：
是否需要 y：
异常输出：
业务动作：
```

示例：

```text
任务名称：
MBE run 级工艺异常检测

检测对象：
一次 MBE 生长任务

一行数据代表：
一个 run

输入 X：
recipe_id
equipment_id
growth_temp_mean
growth_temp_std
pressure_mean
pressure_std
flux_mean
flux_std
alarm_count
days_since_maintenance
source_usage_hours

是否需要 y：
不强依赖 Pass / Fail 标签，可以先基于历史正常 run 建模

异常输出：
anomaly_score
risk_level
top_anomaly_features

业务动作：
异常分数高的 run 优先检测，进入工程师复核列表
```

---

## 作业 2：列出 15 个异常检测特征

至少包括：

```text
5 个工艺稳定性特征
5 个设备状态特征
5 个相对 recipe 偏差特征
```

参考：

```text
工艺稳定性：
growth_temp_std
pressure_std
flux_std
temp_spike_count
pressure_spike_count

设备状态：
days_since_maintenance
recent_alarm_count_7d
equipment_recent_anomaly_rate
source_usage_hours
chamber_usage_hours

相对 recipe 偏差：
temp_mean_dev
growth_time_dev
pressure_mean_ratio
flux_dev
max_deviation_from_setpoint
```

---

## 作业 3：设计异常分数业务动作

按照这个格式写：

```text
anomaly_score 0.00 ~ ?
风险等级：
业务动作：

anomaly_score ? ~ ?
风险等级：
业务动作：

anomaly_score ? ~ ?
风险等级：
业务动作：

anomaly_score ? ~ 1.00
风险等级：
业务动作：
```

参考：

```text
0.00 ~ 0.20：正常，正常流程
0.20 ~ 0.50：关注，工程师抽查
0.50 ~ 0.80：高异常，优先检测和复核
0.80 ~ 1.00：严重异常，暂缓放行并检查设备/工艺日志
```

---

# 29. Day 5 验收标准

你今天合格的标准是：

```text
1. 能解释异常检测和分类的区别
2. 知道异常检测不一定需要 Fail 标签
3. 能说出 MBE 外延中哪些状态适合异常检测
4. 能解释 Z-score 的直觉
5. 能解释为什么要按 recipe / product 分组
6. 知道 Isolation Forest 适合第一版表格异常检测
7. 能把 anomaly_score 转成业务动作
8. 能说出异常检测在没有数据标签时的价值
```

---

# 30. Day 5 最核心总结

今天你只需要记住这句话：

> **当 Fail 标签不足时，不要硬做 Pass / Fail 分类；先用异常检测学习“什么是正常工艺状态”，把不像正常状态的 run、设备、recipe 标出来，让工程师优先复核。**

在外延片场景里，Day 5 最推荐的第一版组合是：

```text
Z-score / IQR：发现单指标异常
Isolation Forest：发现多变量组合异常
Control Chart / EWMA：发现慢性工艺漂移
```

最终系统输出不应该是冷冰冰的：

```text
异常 / 不异常
```

而应该是：

```text
异常分数
风险等级
异常原因
建议检测项
建议工程师动作
```

这才是能落地的半导体 AI 异常检测系统。