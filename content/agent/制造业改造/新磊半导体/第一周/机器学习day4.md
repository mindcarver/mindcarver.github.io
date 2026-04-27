# Day 4：回归问题 —— 预测外延片关键质量指标

Day 3 我们讲的是分类：

```text
这批外延片是 Pass 还是 Fail？
```

Day 4 我们讲回归：

```text
这批外延片的某个质量指标大概是多少？
```

比如：

```text
厚度偏差是多少？
PL 强度是多少？
XRD FWHM 是多少？
表面缺陷数量是多少？
迁移率是多少？
载流子浓度是多少？
电阻率是多少？
```

---

# 1. 什么是回归问题？

回归问题就是：

```text
输入一组特征 X
输出一个连续数值 y
```

分类输出的是类别：

```text
Pass / Fail
低风险 / 中风险 / 高风险
```

回归输出的是数值：

```text
厚度 = 1002.5 nm
PL 强度 = 83.6
XRD FWHM = 42.1 arcsec
缺陷数量 = 8
迁移率 = 8200 cm²/V·s
```

所以你可以这样理解：

|问题|类型|
|---|---|
|是否合格？|分类|
|Fail 概率多少？|分类模型输出概率|
|厚度偏差多少？|回归|
|PL 强度多少？|回归|
|XRD FWHM 多少？|回归|
|缺陷数量多少？|回归|
|质量分多少？|回归|

---

# 2. 外延片里哪些问题适合做回归？

## 2.1 厚度预测

业务问题：

```text
这次 MBE 生长完成后，外延层厚度大概会是多少？
厚度偏差是否可能超规格？
```

机器学习问题：

```text
输入：recipe、温度、时间、束流、压力、设备状态
输出：measured_thickness 或 thickness_deviation
```

模型输出：

```text
预测厚度：1008.2 nm
目标厚度：1000 nm
预测偏差：+0.82%
```

---

## 2.2 厚度均匀性预测

业务问题：

```text
这片外延片的厚度均匀性会不会变差？
```

机器学习问题：

```text
输入：生长过程稳定性、温度波动、压力波动、设备编号、recipe
输出：thickness_uniformity
```

模型输出：

```text
预测厚度均匀性：4.8%
风险：接近上限
```

---

## 2.3 PL 强度预测

PL 可以先理解成光学质量相关指标。

业务问题：

```text
这批外延片 PL 强度是否可能偏低？
```

机器学习问题：

```text
输入：工艺参数、温度稳定性、材料批次、设备状态
输出：pl_intensity
```

模型输出：

```text
预测 PL 强度：68.5
规格下限：70
风险：偏低
```

---

## 2.4 XRD FWHM 预测

XRD FWHM 可以先理解成晶体质量相关指标之一。

业务问题：

```text
这次外延生长的晶体质量是否可能变差？
```

机器学习问题：

```text
输入：生长参数、温度波动、压力波动、源材料状态
输出：xrd_fwhm
```

模型输出：

```text
预测 XRD FWHM：56.2
规格上限：55
风险：可能超规格
```

---

## 2.5 缺陷数量预测

业务问题：

```text
这片外延片表面缺陷可能有多少？
```

机器学习问题：

```text
输入：设备状态、衬底批次、工艺稳定性、报警次数
输出：surface_defect_count
```

模型输出：

```text
预测表面缺陷数量：13
规格上限：12
风险：偏高
```

---

# 3. 回归和分类的关系

分类和回归不是对立的，很多时候可以组合使用。

比如：

## 方案 A：直接做分类

```text
输入 X → 输出 Pass / Fail
```

优点：

```text
直接对应业务结果
```

缺点：

```text
不知道到底是哪项指标导致风险
```

---

## 方案 B：先做回归，再根据规格线判断风险

```text
输入 X → 预测 PL 强度、XRD FWHM、厚度均匀性
再根据规格线判断 Pass / Fail
```

例如：

```text
预测 PL = 68
规格要求 PL >= 70
所以 PL 风险 = Fail
```

优点：

```text
更容易解释
可以告诉工程师具体哪个指标可能异常
```

缺点：

```text
需要每个关键指标都有稳定数据
```

---

# 4. 回归模型的核心价值

分类模型告诉你：

```text
这批片可能 Fail
```

回归模型告诉你：

```text
为什么可能 Fail，以及哪个指标可能 Fail
```

例如：

```text
综合 Fail 风险：高

回归模型预测：
厚度均匀性：正常
PL 强度：偏低
XRD FWHM：正常
表面缺陷数量：偏高

结论：
主要风险来自 PL 偏低和表面缺陷偏多
```

这就比单纯 Pass/Fail 更有工程价值。

---

# 5. 第一版应该预测哪个指标？

如果是新磊这种外延片场景，第一版我建议按这个优先级：

```text
第一优先级：厚度 / 厚度均匀性
第二优先级：PL 强度 / PL 均匀性
第三优先级：XRD FWHM
第四优先级：表面缺陷数量
第五优先级：电学指标，例如 mobility、carrier concentration、sheet resistance
```

原因是：

```text
厚度、PL、XRD、缺陷这类指标更容易成为外延片质量判定依据；
数据通常也比客户反馈数据更容易拿到；
它们比最终 Pass/Fail 更细，可以帮助解释质量风险。
```

---

# 6. 回归问题的数据形式

假设我们要预测：

```text
thickness_uniformity
```

那数据表可能长这样：

|run_id|recipe_id|equipment_id|temp_mean|temp_std|pressure_std|growth_time|thickness_uniformity|
|---|---|---|--:|--:|--:|--:|--:|
|RUN001|R01|MBE01|610|1.2|0.03|120|2.1|
|RUN002|R01|MBE01|614|3.8|0.12|121|5.6|
|RUN003|R03|MBE02|585|1.5|0.04|118|2.7|

其中：

```text
X = recipe_id、equipment_id、temp_mean、temp_std、pressure_std、growth_time
y = thickness_uniformity
```

---

# 7. 回归任务里的数据泄露

回归也有数据泄露。

假设你的预测时间点是：

```text
MBE 生长完成后，检测前
```

你要预测：

```text
PL 强度
```

那你不能把这些字段放入 X：

```text
pl_intensity
pl_fail
final_qc_result
engineer_comment
shipment_decision
customer_feedback
```

因为这些都是结果或结果的变体。

---

## 典型泄露例子

你要预测：

```text
xrd_fwhm
```

但你把这个字段放进输入：

```text
xrd_peak_width
xrd_quality_grade
xrd_pass_flag
```

如果这些字段来自 XRD 检测之后，那就是泄露。

模型效果会非常好，但上线无效。

---

# 8. 回归模型应该用什么？

第一版还是不要上深度学习。

推荐顺序：

```text
1. Linear Regression：线性基线
2. Ridge / Lasso：带正则化的线性模型
3. Random Forest Regressor：非线性基线
4. LightGBM Regressor：主力模型
5. CatBoost Regressor：类别字段多时很好用
6. XGBoost Regressor：强基线
```

我的建议：

```text
表格数据第一版：
LightGBM Regressor + CatBoost Regressor 双模型对比
```

如果类别字段很多，比如：

```text
recipe_id
equipment_id
chamber_id
substrate_batch
source_material_batch
```

那 CatBoost 很适合。

如果数值统计特征很多，比如：

```text
temp_mean
temp_std
pressure_mean
pressure_std
flux_mean
flux_std
alarm_count
source_usage_hours
```

那 LightGBM 很适合。

---

# 9. 回归问题怎么评估？

分类看：

```text
Precision
Recall
F1
PR-AUC
Top-K Recall
```

回归看：

```text
MAE
RMSE
R²
误差分布
规格线附近错误率
```

---

## 9.1 MAE：平均绝对误差

公式：

```text
MAE = 平均 |真实值 - 预测值|
```

直觉理解：

```text
模型平均差多少。
```

例如预测厚度：

```text
真实厚度：1000 nm
预测厚度：1008 nm
误差：8 nm
```

如果很多样本平均下来，MAE = 6 nm，意思就是：

```text
模型预测厚度平均差 6 nm。
```

MAE 好理解，适合跟业务人员讲。

---

## 9.2 RMSE：均方根误差

公式：

```text
RMSE = sqrt(平均(真实值 - 预测值)^2)
```

直觉理解：

```text
大错误会被惩罚得更重。
```

如果你特别怕某些批次预测误差很大，就看 RMSE。

比如：

```text
大部分厚度误差是 3 nm
但偶尔错 40 nm
```

RMSE 会明显变差。

---

## 9.3 R²：解释方差比例

R² 可以理解成：

```text
模型解释了 y 波动的多少比例。
```

比如：

```text
R² = 0.75
```

可以粗略理解为：

```text
模型解释了 75% 的目标变化。
```

但工业场景里，不要迷信 R²。

因为你真正关心的可能是：

```text
规格线附近有没有预测准
高风险批次有没有提前识别
模型误差是否可接受
```

---

# 10. 工业场景里更重要的回归评估

普通机器学习教程会讲 MAE、RMSE、R²。

但半导体质量预测里，你还要看：

## 10.1 规格线附近误差

例如 PL 规格下限是：

```text
PL >= 70
```

真正危险的是这些样本：

```text
真实 PL = 68，模型预测 PL = 75
```

因为模型把高风险片错判成正常。

所以你要特别分析：

```text
接近规格线的样本，模型预测是否可靠？
```

---

## 10.2 方向错误

比如 XRD FWHM 越大越差。

真实情况：

```text
XRD FWHM 变差了
```

但模型预测：

```text
XRD FWHM 变好了
```

这种方向错误在工程上很危险。

---

## 10.3 超规格识别能力

虽然是回归模型，但最后业务上还是会关心：

```text
预测值是否超过规格线？
```

例如：

```text
预测 PL < 70 → PL 风险
预测 XRD FWHM > 55 → XRD 风险
预测 thickness_uniformity > 5% → 厚度均匀性风险
```

所以回归模型也可以转成风险判断。

---

# 11. 一个完整例子：预测 PL 强度

假设我们要做：

```text
任务名称：MBE 生长后 PL 强度预测
```

## 预测时间点

```text
MBE 生长完成后，PL 检测前
```

## 输入 X

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

## 输出 y

```text
pl_intensity
```

## 不能使用的字段

```text
pl_intensity
pl_fail
final_qc_result
engineer_final_comment
customer_feedback
shipment_decision
```

## 业务动作

```text
如果预测 PL 强度低于规格线，则该 run 优先进入 PL mapping 复查。
```

---

# 12. 回归建模最小代码

下面以 CatBoostRegressor 为例。

```python
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. 读取数据
df = pd.read_csv("epi_quality.csv")

# 2. 选择预测目标
target = "pl_intensity"

# 3. 根据预测时间点删除不能用的字段
drop_cols = [
    "pl_intensity",
    "pl_fail",
    "qc_result",
    "final_qc_result",
    "engineer_final_comment",
    "customer_feedback",
    "shipment_decision",
    "xrd_fwhm",              # 如果 XRD 也是检测后才知道，预测 PL 时也先不使用
    "surface_defect_count",  # 同理
    "thickness_uniformity",  # 同理
]

# 只删除数据里存在的列，防止报错
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)
y = df[target]

# 4. 删除纯 ID 字段
id_cols = ["lot_id", "wafer_id", "run_id"]
id_cols = [c for c in id_cols if c in X.columns]
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

cat_features = [c for c in cat_features if c in X.columns]

# 6. 按时间切分
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# 7. 训练回归模型
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

# 8. 预测
y_pred = model.predict(X_test)

# 9. 评估
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
```

你今天重点看懂：

```text
target = 要预测的连续值
CatBoostRegressor = 回归模型
MAE = 平均差多少
RMSE = 大误差惩罚更重
R2 = 解释了多少波动
```

---

# 13. 把回归结果转成业务风险

模型预测出 PL 强度后，不能只展示一个数字。

你要结合规格线：

```python
spec_lower = 70

result = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred
})

result["predicted_pl_risk"] = result["y_pred"] < spec_lower
result["actual_pl_fail"] = result["y_true"] < spec_lower

print(result.head())
```

业务解释：

```text
如果预测 PL < 70，则标记为 PL 风险。
```

最终输出应该是：

```text
批次：RUN2026-001
预测 PL：68.4
规格下限：70
PL 风险：高
建议：优先做 PL mapping 复查
```

---

# 14. 多个回归模型一起用

真实场景中，你不会只预测一个指标。

可以训练多个模型：

```text
模型 1：预测 thickness_uniformity
模型 2：预测 pl_intensity
模型 3：预测 xrd_fwhm
模型 4：预测 surface_defect_count
```

然后组合成一个质量风险表：

|指标|预测值|规格线|风险|
|---|--:|--:|---|
|thickness_uniformity|4.2%|≤ 5%|正常|
|PL intensity|68.4|≥ 70|风险|
|XRD FWHM|51.3|≤ 55|正常|
|surface_defect_count|14|≤ 12|风险|

最终结论：

```text
综合风险：高
主要风险：PL 偏低、表面缺陷偏多
建议：优先进行 PL mapping 和表面检测复核
```

这比单纯分类更有解释性。

---

# 15. 回归任务中的模型解释

回归模型也要解释。

你要回答：

```text
为什么模型预测 PL 偏低？
为什么模型预测 XRD FWHM 偏高？
为什么模型预测厚度均匀性变差？
```

用 SHAP 可以解释。

输出类似：

```text
预测 PL 偏低的主要原因：
1. growth_temp_std 偏高
2. pressure_std 高于同 recipe 历史分位
3. source_usage_hours 偏长
4. days_since_maintenance 偏长
5. substrate_batch 历史质量偏弱
```

这对工程师很重要。

因为他不是只想知道：

```text
模型说 PL 低
```

他更想知道：

```text
为什么 PL 可能低？
下一炉应该检查哪里？
```

---

# 16. 回归和工艺优化的关系

回归模型是后续 recipe 推荐的基础。

你先有一个模型：

```text
输入工艺参数 → 预测质量指标
```

之后才能做优化：

```text
调整工艺参数 → 找到预测质量更好的参数组合
```

比如：

```text
如果温度降低 2 度，PL 是否上升？
如果压力波动降低，XRD 是否改善？
如果源材料使用时长过长，是否应该更换？
```

所以 Day 4 的回归模型，是后面工艺优化的基础。

---

# 17. 回归问题最常见的 6 个坑

## 坑 1：把结果字段放进输入

比如预测 PL，却把 PL 检测相关字段放进 X。

这是数据泄露。

---

## 坑 2：只看 R²

R² 高不代表业务好。

你还要看：

```text
规格线附近预测准不准
高风险样本有没有抓出来
大误差是否集中在某些 recipe 或设备
```

---

## 坑 3：所有产品混在一起建模

HEMT、HBT、VCSEL、APD 的指标逻辑可能不同。

如果数据量足够，最好：

```text
分产品类型建模
或者把 product_type 作为重要特征
```

---

## 坑 4：不区分 recipe

不同 recipe 的目标厚度、温度、结构都不同。

很多时候要预测的不是绝对值，而是：

```text
相对目标值的偏差
```

例如：

```text
thickness_deviation = measured_thickness - target_thickness
```

这比直接预测 measured_thickness 更合理。

---

## 坑 5：忽略时间变化

设备会漂移，材料会老化，工艺会改版。

所以不要只随机切分，要做时间切分。

---

## 坑 6：只输出预测值，不输出业务动作

模型说：

```text
预测 PL = 68.4
```

还不够。

要输出：

```text
低于规格线
风险等级高
建议优先复查 PL mapping
```

---

# 18. 回归任务的专业表达

以后你跟客户聊，不要说：

```text
我们可以预测质量。
```

你要说：

```text
我们可以先把综合质量拆成若干可量化指标，例如厚度均匀性、PL 强度、XRD FWHM 和表面缺陷数量。
每个指标都可以建立一个回归模型，预测其可能数值。
再结合规格线，把预测结果转化成风险等级和复核建议。
这样不仅能判断是否高风险，还能解释风险来自哪个质量指标。
```

这就是专业表达。

---

# 19. Day 4 课堂练习

## 练习 1：判断是分类还是回归

### 问题 A

```text
预测外延片是否合格
```

答案：

```text
分类
```

---

### 问题 B

```text
预测 PL 强度是多少
```

答案：

```text
回归
```

---

### 问题 C

```text
预测 XRD FWHM 是否超过 55
```

答案：

```text
分类
```

因为输出是是否超过。

---

### 问题 D

```text
预测 XRD FWHM 的具体数值
```

答案：

```text
回归
```

---

### 问题 E

```text
预测表面缺陷数量
```

答案：

```text
回归，或者计数回归
```

第一版可以先按普通回归做。

---

# 20. 练习 2：设计一个回归任务

示例：

```text
任务名称：
MBE 生长后 PL 强度预测

预测时间点：
MBE 生长完成后，PL 检测前

一行数据代表：
一次 MBE run

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
flux_mean
flux_std
growth_time
alarm_count
days_since_maintenance
source_usage_hours

输出 y：
pl_intensity

不能使用：
pl_intensity
pl_fail
final_qc_result
customer_feedback
engineer_final_comment

业务动作：
如果预测 PL 低于规格下限，则该 run 进入优先 PL mapping 复查队列
```

---

# 21. Day 4 作业

今天作业有 3 个。

---

## 作业 1：定义 3 个回归任务

按照这个模板写：

```text
任务名称：
预测时间点：
一行数据代表：
输入 X：
输出 y：
不能使用的字段：
规格线：
业务动作：
```

你至少写 3 个：

```text
1. 厚度均匀性预测
2. PL 强度预测
3. XRD FWHM 预测
```

---

## 作业 2：解释 3 个评估指标

用自己的话解释：

```text
MAE：
RMSE：
R²：
```

重点说明：

```text
哪个指标最容易给老板讲？
哪个指标对大误差更敏感？
为什么不能只看 R²？
```

---

## 作业 3：把回归结果转成业务动作

假设规格线如下：

```text
PL intensity >= 70
XRD FWHM <= 55
thickness_uniformity <= 5%
surface_defect_count <= 12
```

模型预测结果：

```text
PL intensity = 68
XRD FWHM = 52
thickness_uniformity = 4.6%
surface_defect_count = 14
```

你要输出：

```text
哪些指标有风险？
综合风险等级是什么？
建议工程师优先复查什么？
```

参考答案：

```text
有风险：
PL intensity 偏低
surface_defect_count 偏高

正常：
XRD FWHM
thickness_uniformity

综合风险：
高

建议：
优先复查 PL mapping 和表面缺陷检测；
暂缓直接放行，进入工程师复核。
```

---

# 22. Day 4 验收标准

你今天合格的标准是：

```text
1. 能解释什么是回归问题
2. 能区分分类和回归
3. 能说出外延片里哪些指标适合回归预测
4. 能定义 X 和 y
5. 能识别回归任务里的数据泄露
6. 能解释 MAE、RMSE、R²
7. 能把预测数值转成规格风险
8. 能说明回归模型为什么有助于质量解释
```

---

# 23. Day 4 最核心总结

今天你只需要记住这句话：

> **分类模型告诉你这批外延片有没有风险，回归模型告诉你哪些质量指标可能出问题，以及大概会偏到什么程度。**

在外延片质量预测里，最推荐的落地组合是：

```text
先用分类模型做综合 Fail 风险评分；
再用回归模型预测 PL、XRD、厚度均匀性、缺陷数量等关键指标；
最后结合规格线生成风险等级和复核建议。
```

这样模型才不是一个黑盒判断，而是能服务工程师的质量分析工具。s