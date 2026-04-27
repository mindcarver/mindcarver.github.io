# Day 6：第一个完整项目 —— Mini 外延片质量风险评分系统

前 5 天我们分别学了：

```text
Day 1：业务场景怎么拆成 AI 问题
Day 2：X、y、训练集、测试集、数据泄露
Day 3：分类：Pass / Fail 风险预测
Day 4：回归：预测 PL、XRD、厚度等质量指标
Day 5：异常检测：没有 Fail 标签时怎么发现异常 run
```

Day 6 开始把它们串起来，做第一个小项目。

今天的目标不是做一个完美模型，而是打通完整闭环：

```text
外延片 run 数据
→ 数据检查
→ 异常检测
→ Fail 风险预测
→ 关键指标预测
→ 风险等级
→ 工程师建议动作
```

---

# 1. Day 6 项目名称

```text
Mini Epi Quality Risk Scoring System
迷你外延片质量风险评分系统
```

一句话定义：

> 在 MBE 生长完成后、检测前，根据工艺参数、设备状态、recipe 信息和材料批次信息，给每次 run 输出质量风险等级，并生成工程师复核建议。

---

# 2. 今天要做出的结果长什么样？

最终输出不是单纯模型分数，而是类似下面这样：

```text
Run ID：RUN2026-0008
产品类型：HEMT
Recipe：R-HEMT-017
设备：MBE-03

综合风险等级：高

模型结果：
- Fail 风险概率：0.67
- 异常分数：0.82
- 预测 PL 强度：68.5，低于规格线 70
- 预测 XRD FWHM：52.1，正常
- 预测厚度均匀性：4.7%，接近上限 5%

主要风险原因：
1. 生长温度波动偏高
2. 压力波动偏高
3. 源材料使用时长偏长
4. 距离上次维护时间偏长
5. 同 recipe 下当前 run 异常分数偏高

建议动作：
1. 优先做 PL mapping 复查
2. 复查温控曲线和真空日志
3. 检查源炉状态
4. 暂缓直接放行，进入工程师复核
```

你要注意，这就是工业 AI 和普通机器学习 Demo 的区别。

普通 Demo 输出：

```text
prediction = 1
```

工业 AI 输出：

```text
风险概率
异常分数
具体质量指标预测
主要风险原因
建议动作
```

---

# 3. 项目边界：今天只做 v0.1

今天的 v0.1 不做这些：

```text
不接真实 MES
不接真实设备日志
不做实时流式处理
不做复杂深度学习
不做自动判废
不做自动调参
```

今天只做：

```text
用一张结构化表格
训练一个分类模型
训练几个回归模型
训练一个异常检测模型
最后组合成风险评分结果
```

这就是一个最小闭环。

---

# 4. 业务时间点再次确认

我们选择的预测时间点是：

```text
MBE 生长完成后，PL / XRD / AFM / 表面检测之前
```

所以模型输入只能使用：

```text
recipe 信息
设备信息
衬底/源材料信息
生长过程统计特征
设备状态特征
维护状态特征
```

不能使用：

```text
PL 检测结果
XRD 检测结果
AFM 检测结果
表面缺陷检测结果
最终 QC 结果
客户反馈
工程师最终判定
```

注意：  
在训练阶段，`qc_result`、`pl_intensity`、`xrd_fwhm` 这些字段可以作为标签 y。  
但是它们不能作为输入 X。

---

# 5. 数据表设计

今天我们假设有一张 run 级别数据表。

一行代表：

```text
一次 MBE 生长 run
```

字段分成 5 类。

---

## 5.1 ID 字段

```text
run_id
lot_id
product_type
recipe_id
equipment_id
chamber_id
substrate_batch
source_material_batch
```

这些字段用于追溯。

其中部分字段也可以作为类别特征，例如：

```text
product_type
recipe_id
equipment_id
chamber_id
substrate_batch
source_material_batch
```

---

## 5.2 生长过程特征

```text
target_temp
actual_temp_mean
actual_temp_std
target_pressure
actual_pressure_mean
actual_pressure_std
growth_time
flux_mean
flux_std
alarm_count
```

这些代表这次 MBE 生长过程是否稳定。

---

## 5.3 设备状态特征

```text
days_since_maintenance
source_usage_hours
chamber_usage_hours
recent_alarm_count_7d
equipment_recent_anomaly_rate
```

这些代表设备、腔体、源材料是否处于健康状态。

---

## 5.4 检测结果字段

```text
thickness_uniformity
pl_intensity
xrd_fwhm
surface_defect_count
```

这些字段在今天的任务里有两种用途：

```text
训练回归模型时，它们是 y
训练分类标签时，它们可以帮助定义 qc_result
但在检测前预测时，它们不能作为 X
```

---

## 5.5 质量标签

```text
qc_result
```

取值：

```text
Pass
Fail
```

这是分类模型的 y。

---

# 6. 今天做 4 个模型

## 模型 1：异常检测模型

用途：

```text
判断这次 run 的工艺/设备状态是否不像历史正常状态
```

推荐：

```text
Isolation Forest
```

输出：

```text
anomaly_score
```

---

## 模型 2：Pass / Fail 分类模型

用途：

```text
预测这次 run 最终 Fail 的概率
```

推荐：

```text
CatBoostClassifier
或者
LightGBMClassifier
```

输出：

```text
fail_probability
```

---

## 模型 3：关键指标回归模型

用途：

```text
预测具体质量指标
```

比如：

```text
pl_intensity
xrd_fwhm
thickness_uniformity
surface_defect_count
```

推荐：

```text
CatBoostRegressor
或者
LightGBMRegressor
```

输出：

```text
predicted_pl
predicted_xrd_fwhm
predicted_thickness_uniformity
predicted_surface_defect_count
```

---

## 模型 4：规则评分器

这个不是机器学习模型，而是业务规则。

它把上面的结果组合起来：

```text
fail_probability
anomaly_score
predicted_pl
predicted_xrd
predicted_thickness_uniformity
predicted_defect_count
```

最后输出：

```text
低风险 / 中风险 / 高风险 / 极高风险
```

---

# 7. 整体流程图

```text
原始 run 数据
    ↓
字段检查 / 缺失值处理 / 泄露字段删除
    ↓
构造特征
    ↓
按时间切分训练集和测试集
    ↓
训练异常检测模型
    ↓
训练 Pass / Fail 分类模型
    ↓
训练 PL / XRD / 厚度 / 缺陷数量回归模型
    ↓
对新 run 预测
    ↓
输出风险分、异常分、指标预测值
    ↓
结合规格线生成业务动作
```

---

# 8. 第一步：明确输入 X

在我们的预测时间点下，X 应该是：

```python
feature_cols = [
    "product_type",
    "recipe_id",
    "equipment_id",
    "chamber_id",
    "substrate_batch",
    "source_material_batch",

    "target_temp",
    "actual_temp_mean",
    "actual_temp_std",
    "target_pressure",
    "actual_pressure_mean",
    "actual_pressure_std",
    "growth_time",
    "flux_mean",
    "flux_std",
    "alarm_count",

    "days_since_maintenance",
    "source_usage_hours",
    "chamber_usage_hours",
    "recent_alarm_count_7d",
    "equipment_recent_anomaly_rate",
]
```

不能进入 X 的字段：

```python
leakage_cols = [
    "qc_result",
    "pl_intensity",
    "xrd_fwhm",
    "thickness_uniformity",
    "surface_defect_count",
    "engineer_comment",
    "shipment_decision",
    "customer_feedback",
]
```

这一步非常关键。

因为你今天做的是：

```text
检测前预测
```

所以检测后字段不能作为输入。

---

# 9. 第二步：构造几个关键特征

不要只喂原始值，要构造相对偏差。

## 9.1 温度偏差

```python
df["temp_mean_dev"] = df["actual_temp_mean"] - df["target_temp"]
```

含义：

```text
实际平均温度相对目标温度偏了多少
```

---

## 9.2 压力比例

```python
df["pressure_mean_ratio"] = df["actual_pressure_mean"] / df["target_pressure"]
df["pressure_std_ratio"] = df["actual_pressure_std"] / df["target_pressure"]
```

含义：

```text
当前压力水平和压力波动相对目标压力是否偏大
```

---

## 9.3 稳定性综合分

```python
df["process_instability_score"] = (
    df["actual_temp_std"] * 0.4
    + df["pressure_std_ratio"] * 0.4
    + df["flux_std"] * 0.2
)
```

含义：

```text
温度、压力、束流的综合稳定性风险
```

这只是 v0.1 的简单写法，真实项目里权重应该通过数据和工程师经验调整。

---

## 9.4 设备老化特征

```python
df["maintenance_risk"] = df["days_since_maintenance"] / 120
df["source_age_risk"] = df["source_usage_hours"] / 900
```

含义：

```text
距离维护越久、源材料使用越久，风险可能越高
```

---

# 10. 第三步：训练异常检测模型

异常检测只使用 X，不需要 y。

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

anomaly_features = [
    "temp_mean_dev",
    "actual_temp_std",
    "pressure_mean_ratio",
    "pressure_std_ratio",
    "flux_std",
    "alarm_count",
    "maintenance_risk",
    "source_age_risk",
    "recent_alarm_count_7d",
    "equipment_recent_anomaly_rate",
]

X_anomaly = df[anomaly_features].copy()
X_anomaly = X_anomaly.fillna(X_anomaly.median(numeric_only=True))

scaler = StandardScaler()
X_anomaly_scaled = scaler.fit_transform(X_anomaly)

iforest = IsolationForest(
    n_estimators=300,
    contamination=0.05,
    random_state=42
)

iforest.fit(X_anomaly_scaled)

raw_score = iforest.decision_function(X_anomaly_scaled)

df["anomaly_score"] = -raw_score
df["is_anomaly"] = iforest.predict(X_anomaly_scaled) == -1
```

这里注意：

```text
decision_function 越低越异常
所以我们用 -raw_score 转成 anomaly_score
使它越高越异常
```

---

# 11. 第四步：训练 Pass / Fail 分类模型

分类模型预测：

```text
qc_result = Pass / Fail
```

代码结构：

```python
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, average_precision_score

target = "qc_result"

y_cls = df[target].map({
    "Pass": 0,
    "Fail": 1
})

model_features = [
    "product_type",
    "recipe_id",
    "equipment_id",
    "chamber_id",
    "substrate_batch",
    "source_material_batch",

    "temp_mean_dev",
    "actual_temp_std",
    "pressure_mean_ratio",
    "pressure_std_ratio",
    "growth_time",
    "flux_mean",
    "flux_std",
    "alarm_count",
    "maintenance_risk",
    "source_age_risk",
    "recent_alarm_count_7d",
    "equipment_recent_anomaly_rate",
]

X_cls = df[model_features].copy()

cat_features = [
    "product_type",
    "recipe_id",
    "equipment_id",
    "chamber_id",
    "substrate_batch",
    "source_material_batch",
]

split_idx = int(len(df) * 0.8)

X_train = X_cls.iloc[:split_idx]
X_test = X_cls.iloc[split_idx:]
y_train = y_cls.iloc[:split_idx]
y_test = y_cls.iloc[split_idx:]

clf = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=100
)

clf.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

fail_proba = clf.predict_proba(X_test)[:, 1]
y_pred = (fail_proba >= 0.3).astype(int)

print(classification_report(y_test, y_pred))
print("PR-AUC:", average_precision_score(y_test, fail_proba))
```

重点不是代码，而是这几个判断：

```text
Fail 作为正类
不要只看 Accuracy
阈值不一定是 0.5
0.3 可以作为进入工程师复核的预警线
```

---

# 12. 第五步：训练关键指标回归模型

今天我们先训练 3 个回归模型：

```text
PL 强度预测
XRD FWHM 预测
厚度均匀性预测
```

可以写一个通用函数：

```python
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_regression_model(df, target, model_features, cat_features, split_idx):
    X = df[model_features].copy()
    y = df[target]

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        loss_function="RMSE",
        verbose=100
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features
    )

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Target: {target}")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
    print("-" * 40)

    return model
```

训练 3 个模型：

```python
pl_model = train_regression_model(
    df=df,
    target="pl_intensity",
    model_features=model_features,
    cat_features=cat_features,
    split_idx=split_idx
)

xrd_model = train_regression_model(
    df=df,
    target="xrd_fwhm",
    model_features=model_features,
    cat_features=cat_features,
    split_idx=split_idx
)

thickness_model = train_regression_model(
    df=df,
    target="thickness_uniformity",
    model_features=model_features,
    cat_features=cat_features,
    split_idx=split_idx
)
```

这一步的业务含义是：

```text
不是只预测是否 Fail，
而是进一步预测哪个质量指标可能出问题。
```

---

# 13. 第六步：对一个新 run 做综合预测

假设来了一个新 run。

输入：

```python
new_run = pd.DataFrame([{
    "product_type": "HEMT",
    "recipe_id": "R-HEMT-017",
    "equipment_id": "MBE-03",
    "chamber_id": "CH-02",
    "substrate_batch": "SUB-045",
    "source_material_batch": "SRC-021",

    "target_temp": 610,
    "actual_temp_mean": 614,
    "actual_temp_std": 3.7,
    "target_pressure": 1.2e-7,
    "actual_pressure_mean": 1.35e-7,
    "actual_pressure_std": 0.18e-7,
    "growth_time": 122,
    "flux_mean": 1.02,
    "flux_std": 0.07,
    "alarm_count": 3,

    "days_since_maintenance": 95,
    "source_usage_hours": 820,
    "chamber_usage_hours": 1600,
    "recent_alarm_count_7d": 11,
    "equipment_recent_anomaly_rate": 0.18,
}])
```

构造同样的特征：

```python
new_run["temp_mean_dev"] = new_run["actual_temp_mean"] - new_run["target_temp"]
new_run["pressure_mean_ratio"] = new_run["actual_pressure_mean"] / new_run["target_pressure"]
new_run["pressure_std_ratio"] = new_run["actual_pressure_std"] / new_run["target_pressure"]

new_run["maintenance_risk"] = new_run["days_since_maintenance"] / 120
new_run["source_age_risk"] = new_run["source_usage_hours"] / 900
```

分类预测：

```python
new_X = new_run[model_features]

fail_probability = clf.predict_proba(new_X)[:, 1][0]
```

回归预测：

```python
pred_pl = pl_model.predict(new_X)[0]
pred_xrd = xrd_model.predict(new_X)[0]
pred_thickness = thickness_model.predict(new_X)[0]
```

异常检测：

```python
new_X_anomaly = new_run[anomaly_features].copy()
new_X_anomaly_scaled = scaler.transform(new_X_anomaly)

new_raw_score = iforest.decision_function(new_X_anomaly_scaled)[0]
new_anomaly_score = -new_raw_score
```

---

# 14. 第七步：把模型输出转成业务风险

先定义规格线：

```python
specs = {
    "pl_lower": 70,
    "xrd_upper": 55,
    "thickness_uniformity_upper": 5.0,
}
```

判断单项风险：

```python
pl_risk = pred_pl < specs["pl_lower"]
xrd_risk = pred_xrd > specs["xrd_upper"]
thickness_risk = pred_thickness > specs["thickness_uniformity_upper"]
```

组合综合风险：

```python
risk_points = 0

if fail_probability >= 0.5:
    risk_points += 2
elif fail_probability >= 0.3:
    risk_points += 1

if new_anomaly_score >= 0.8:
    risk_points += 2
elif new_anomaly_score >= 0.5:
    risk_points += 1

if pl_risk:
    risk_points += 2

if xrd_risk:
    risk_points += 1

if thickness_risk:
    risk_points += 1

if risk_points >= 5:
    risk_level = "极高风险"
elif risk_points >= 3:
    risk_level = "高风险"
elif risk_points >= 1:
    risk_level = "中风险"
else:
    risk_level = "低风险"
```

这里你要理解：

```text
风险评分不一定完全由机器学习决定。
真实工业系统里，经常是“模型 + 规则 + 工程师经验”组合。
```

这是非常正常的。

---

# 15. 第八步：生成工程师建议动作

```python
actions = []

if pl_risk:
    actions.append("预测 PL 强度低于规格线，建议优先做 PL mapping 复查。")

if xrd_risk:
    actions.append("预测 XRD FWHM 偏高，建议复查晶体质量相关指标。")

if thickness_risk:
    actions.append("预测厚度均匀性接近或超过上限，建议复查厚度 mapping。")

if new_run["actual_temp_std"].iloc[0] > 3:
    actions.append("生长温度波动偏高，建议检查温控曲线。")

if new_run["pressure_std_ratio"].iloc[0] > 0.1:
    actions.append("压力波动偏高，建议检查真空系统和过程日志。")

if new_run["alarm_count"].iloc[0] >= 2:
    actions.append("本次 run 报警次数偏多，建议查看报警日志。")

if new_run["source_age_risk"].iloc[0] > 0.8:
    actions.append("源材料使用时长偏长，建议检查源炉状态。")

if new_run["maintenance_risk"].iloc[0] > 0.75:
    actions.append("距离上次维护时间较长，建议关注设备维护状态。")

if not actions:
    actions.append("暂无明显风险，按正常流程处理。")
```

这就是把模型结果转成业务语言。

---

# 16. 最终输出函数

你可以把它封装成一个函数：

```python
def generate_quality_report(
    run_id,
    fail_probability,
    anomaly_score,
    pred_pl,
    pred_xrd,
    pred_thickness,
    risk_level,
    actions
):
    report = f"""
Run ID：{run_id}

综合风险等级：{risk_level}

模型结果：
- Fail 风险概率：{fail_probability:.2f}
- 异常分数：{anomaly_score:.2f}
- 预测 PL 强度：{pred_pl:.2f}
- 预测 XRD FWHM：{pred_xrd:.2f}
- 预测厚度均匀性：{pred_thickness:.2f}%

建议动作：
"""
    for i, action in enumerate(actions, start=1):
        report += f"{i}. {action}\n"

    return report
```

调用：

```python
report = generate_quality_report(
    run_id="RUN2026-0008",
    fail_probability=fail_probability,
    anomaly_score=new_anomaly_score,
    pred_pl=pred_pl,
    pred_xrd=pred_xrd,
    pred_thickness=pred_thickness,
    risk_level=risk_level,
    actions=actions
)

print(report)
```

输出类似：

```text
Run ID：RUN2026-0008

综合风险等级：高风险

模型结果：
- Fail 风险概率：0.67
- 异常分数：0.82
- 预测 PL 强度：68.50
- 预测 XRD FWHM：52.10
- 预测厚度均匀性：4.70%

建议动作：
1. 预测 PL 强度低于规格线，建议优先做 PL mapping 复查。
2. 生长温度波动偏高，建议检查温控曲线。
3. 压力波动偏高，建议检查真空系统和过程日志。
4. 本次 run 报警次数偏多，建议查看报警日志。
5. 源材料使用时长偏长，建议检查源炉状态。
```

---

# 17. 今天这个项目真正训练你什么？

不是训练你会调 CatBoost 参数。

而是训练你形成完整产品思维：

```text
数据从哪里来
哪些字段能用
哪些字段不能用
预测时间点是什么
分类模型解决什么
回归模型解决什么
异常检测解决什么
规则系统解决什么
模型结果怎么变成业务动作
```

这就是 AI 应用落地的关键。

---

# 18. 这个 Mini 系统和真实项目的差距

今天做的是 v0.1。

真实项目还要补：

```text
真实数据接入
字段含义确认
工程师审核标签
按产品/recipe 分组建模
SHAP 解释
历史相似 run 检索
模型漂移监控
Dashboard
API 服务
工程师反馈闭环
```

但 v0.1 已经能说明一件事：

```text
你不是只会训练模型，
你已经能把机器学习结果组织成一个半导体质量风险系统。
```

---

# 19. Day 6 作业

## 作业 1：写出你的 Mini 系统模块

按照这个格式写：

```text
系统名称：

输入数据：

模型 1：
作用：

模型 2：
作用：

模型 3：
作用：

最终输出：

业务动作：
```

参考答案：

```text
系统名称：
Mini 外延片质量风险评分系统

输入数据：
MBE run 级工艺参数、设备参数、recipe、衬底批次、源材料状态

模型 1：
Isolation Forest

作用：
判断当前 run 是否不像历史正常状态

模型 2：
CatBoostClassifier

作用：
预测 Pass / Fail 风险概率

模型 3：
CatBoostRegressor

作用：
预测 PL、XRD、厚度均匀性等关键质量指标

最终输出：
Fail 概率、异常分数、关键指标预测值、综合风险等级、建议动作

业务动作：
高风险 run 进入优先检测和工程师复核列表
```

---

## 作业 2：设计你的风险评分规则

按照这个格式写：

```text
Fail 概率超过多少，加几分？
异常分数超过多少，加几分？
PL 低于规格线，加几分？
XRD 高于规格线，加几分？
厚度均匀性超过规格线，加几分？

总分多少是低风险？
总分多少是中风险？
总分多少是高风险？
总分多少是极高风险？
```

参考：

```text
Fail 概率 >= 0.5：+2
Fail 概率 0.3 ~ 0.5：+1

异常分数 >= 0.8：+2
异常分数 0.5 ~ 0.8：+1

PL 低于规格线：+2
XRD 高于规格线：+1
厚度均匀性高于规格线：+1

0 分：低风险
1~2 分：中风险
3~4 分：高风险
5 分以上：极高风险
```

---

## 作业 3：写一个最终报告模板

模板：

```text
Run ID：
产品类型：
Recipe：
设备：

综合风险等级：

模型结果：
- Fail 风险概率：
- 异常分数：
- 预测 PL：
- 预测 XRD：
- 预测厚度均匀性：

主要风险原因：

建议动作：
```

你的目标是写成工程师能看懂的形式，不要写成算法报告。

---

# 20. Day 6 验收标准

今天合格标准是：

```text
1. 能把分类、回归、异常检测组合成一个小系统
2. 能说清楚每个模型负责什么
3. 能设计输入字段和输出结果
4. 能避免检测后字段泄露进模型输入
5. 能把模型分数转成风险等级
6. 能把风险等级转成业务动作
7. 能生成一份工程师可读的质量风险报告
```

---

# 21. Day 6 最核心总结

今天最重要的一句话：

> **机器学习模型本身只是组件，真正能落地的是“模型 + 规则 + 解释 + 业务动作”的质量风险系统。**

对于新磊这种外延片场景，第一个 AI 项目不要做得太重。

最合理的 v0.1 是：

```text
MBE 生长后、检测前，
用工艺参数和设备状态，
输出 run 级风险评分，
并建议工程师优先复核哪些检测项。
```

这就是你第一个可以给客户讲的半导体 AI Demo 原型。