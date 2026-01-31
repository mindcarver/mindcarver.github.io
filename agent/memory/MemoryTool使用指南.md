# MemoryTool 详细使用指南

## 1. 概述

MemoryTool是记忆系统的统一接口工具，为Agent提供记忆功能的封装。它通过`execute()`方法支持多种记忆操作，遵循"统一入口，分发处理"的架构模式。

## 2. 初始化

### 2.1 基本初始化

```python
from hello_agents.tools import MemoryTool

# 创建记忆工具
memory_tool = MemoryTool(user_id="user123")
```

### 2.2 完整参数初始化

```python
from hello_agents.memory import MemoryConfig
from hello_agents.tools import MemoryTool

# 创建配置
memory_config = MemoryConfig(
    working_memory_capacity=50,
    working_memory_ttl=60,
    database_path="./memory_data/memory.db",
    # ... 其他配置
)

# 创建记忆工具
memory_tool = MemoryTool(
    user_id="user123",
    memory_config=memory_config,
    memory_types=["working", "episodic", "semantic"]
)
```

### 2.3 参数说明

| 参数          | 类型         | 默认值                              | 说明           |
| ------------- | ------------ | ----------------------------------- | -------------- |
| user_id       | str          | "default_user"                      | 用户唯一标识   |
| memory_config | MemoryConfig | None                                | 记忆配置对象   |
| memory_types  | List[str]    | ["working", "episodic", "semantic"] | 启用的记忆类型 |

## 3. 核心操作

### 3.1 add - 添加记忆

#### 功能说明

将信息编码为记忆并存储到记忆系统中。

#### 参数

| 参数         | 类型  | 默认值    | 说明                          |
| ------------ | ----- | --------- | ----------------------------- |
| content      | str   | ""        | 记忆内容                      |
| memory_type  | str   | "working" | 记忆类型                      |
| importance   | float | 0.5       | 重要性（0.0-1.0）             |
| file_path    | str   | None      | 文件路径（感知记忆）          |
| modality     | str   | None      | 模态类型（image/audio/video） |
| \*\*metadata | Any   | -         | 自定义元数据                  |

#### 示例代码

```python
# 1. 工作记忆 - 临时信息
memory_tool.execute("add",
    content="用户刚才问了关于Python函数的问题",
    memory_type="working",
    importance=0.6
)

# 2. 情景记忆 - 具体事件
memory_tool.execute("add",
    content="2024年3月15日，用户张三完成了第一个Python项目",
    memory_type="episodic",
    importance=0.8,
    event_type="milestone",
    location="在线学习平台"
)

# 3. 语义记忆 - 抽象知识
memory_tool.execute("add",
    content="Python是一种解释型、面向对象的编程语言",
    memory_type="semantic",
    importance=0.9,
    knowledge_type="factual"
)

# 4. 感知记忆 - 多模态数据
memory_tool.execute("add",
    content="用户上传了一张Python代码截图，包含函数定义",
    memory_type="perceptual",
    importance=0.7,
    modality="image",
    file_path="./uploads/code_screenshot.png"
)
```

#### 返回值

```
✅ 记忆已添加 (ID: a1b2c3d4...)
```

### 3.2 search - 搜索记忆

#### 功能说明

根据查询从记忆中检索相关内容。

#### 参数

| 参数           | 类型      | 默认值   | 说明           |
| -------------- | --------- | -------- | -------------- |
| query          | str       | required | 搜索查询       |
| limit          | int       | 5        | 返回结果数量   |
| memory_type    | str       | None     | 单一记忆类型   |
| memory_types   | List[str] | None     | 多个记忆类型   |
| min_importance | float     | 0.1      | 最小重要性阈值 |

#### 示例代码

```python
# 1. 基础搜索
result = memory_tool.execute("search", query="Python编程", limit=5)
print(result)

# 2. 指定记忆类型搜索
result = memory_tool.execute("search",
    query="学习进度",
    memory_type="episodic",
    limit=3
)

# 3. 多类型搜索
result = memory_tool.execute("search",
    query="函数定义",
    memory_types=["semantic", "episodic"],
    min_importance=0.5
)

# 4. 搜索特定用户记忆
result = memory_tool.execute("search",
    query="机器学习",
    memory_type="semantic",
    user_id="user123",
    limit=10
)
```

#### 返回值格式

```
🔍 找到 3 条相关记忆:
1. [语义记忆] Python是一种解释型、面向对象的编程语言 (重要性: 0.90)
2. [情景记忆] 2024年3月15日，用户张三完成了第一个Python项目 (重要性: 0.80)
3. [工作记忆] 用户刚才问了关于Python函数的问题 (重要性: 0.60)
```

### 3.3 forget - 遗忘记忆

#### 功能说明

根据不同策略删除记忆，模拟人类的选择性遗忘。

#### 支持的策略

| 策略             | 说明             | 参数         |
| ---------------- | ---------------- | ------------ |
| importance_based | 基于重要性的遗忘 | threshold    |
| time_based       | 基于时间的遗忘   | max_age_days |
| capacity_based   | 基于容量的遗忘   | threshold    |

#### 示例代码

```python
# 1. 基于重要性的遗忘 - 删除重要性低于阈值的记忆
memory_tool.execute("forget",
    strategy="importance_based",
    threshold=0.2
)
# 输出: 🧹 已遗忘 15 条记忆（策略: importance_based）

# 2. 基于时间的遗忘 - 删除超过指定天数的记忆
memory_tool.execute("forget",
    strategy="time_based",
    max_age_days=30
)
# 输出: 🧹 已遗忘 42 条记忆（策略: time_based）

# 3. 基于容量的遗忘 - 当记忆数量超限时删除最不重要的
memory_tool.execute("forget",
    strategy="capacity_based",
    threshold=0.3
)
# 输出: 🧹 已遗忘 8 条记忆（策略: capacity_based）
```

### 3.4 consolidate - 整合记忆

#### 功能说明

将短期记忆转化为长期记忆，模拟记忆固化过程。

#### 参数

| 参数                 | 类型  | 默认值     | 说明         |
| -------------------- | ----- | ---------- | ------------ |
| from_type            | str   | "working"  | 源记忆类型   |
| to_type              | str   | "episodic" | 目标记忆类型 |
| importance_threshold | float | 0.7        | 重要性阈值   |

#### 示例代码

```python
# 1. 将重要的工作记忆转为情景记忆
memory_tool.execute("consolidate",
    from_type="working",
    to_type="episodic",
    importance_threshold=0.7
)
# 输出: 🔄 已整合 12 条记忆为长期记忆（working → episodic，阈值=0.7）

# 2. 将重要的情景记忆转为语义记忆
memory_tool.execute("consolidate",
    from_type="episodic",
    to_type="semantic",
    importance_threshold=0.8
)
# 输出: 🔄 已整合 5 条记忆为长期记忆（episodic → semantic，阈值=0.8）
```

### 3.5 summary - 获取记忆摘要

#### 功能说明

获取当前记忆系统的统计摘要信息。

#### 参数

| 参数        | 类型 | 默认值 | 说明         |
| ----------- | ---- | ------ | ------------ |
| memory_type | str  | None   | 指定记忆类型 |

#### 示例代码

```python
# 获取所有记忆摘要
result = memory_tool.execute("summary")
print(result)

# 获取特定类型记忆摘要
result = memory_tool.execute("summary", memory_type="semantic")
print(result)
```

#### 返回值格式

```
📊 记忆系统摘要
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计: 128 条记忆
├─ 工作记忆: 45 条
├─ 情景记忆: 52 条
├─ 语义记忆: 28 条
└─ 感知记忆: 3 条

平均重要性: 0.65
最早记忆: 2024-01-15
最新记忆: 2024-03-20
```

### 3.6 stats - 获取统计信息

#### 功能说明

获取详细的记忆统计数据。

#### 示例代码

```python
result = memory_tool.execute("stats")
print(result)
```

#### 返回值格式

```
📈 记忆统计详情
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
用户ID: user123
会话数: 12

工作记忆:
  总数: 45
  平均重要性: 0.58
  TTL: 60分钟

情景记忆:
  总数: 52
  平均重要性: 0.72
  会话数: 12

语义记忆:
  总数: 28
  平均重要性: 0.85
  实体数: 156
  关系数: 89

感知记忆:
  总数: 3
  模态: 2 图像, 1 音频
```

### 3.7 update - 更新记忆

#### 功能说明

更新现有记忆的内容或元数据。

#### 参数

| 参数         | 类型  | 默认值   | 说明         |
| ------------ | ----- | -------- | ------------ |
| memory_id    | str   | required | 记忆ID       |
| content      | str   | None     | 新内容       |
| importance   | float | None     | 新重要性     |
| \*\*metadata | Any   | -        | 更新的元数据 |

#### 示例代码

```python
# 更新记忆内容
memory_tool.execute("update",
    memory_id="a1b2c3d4e5f6",
    content="更新后的内容",
    importance=0.9
)

# 更新元数据
memory_tool.execute("update",
    memory_id="a1b2c3d4e5f6",
    status="completed",
    tags=["python", "重要"]
)
```

### 3.8 remove - 删除记忆

#### 功能说明

删除指定的记忆。

#### 参数

| 参数      | 类型 | 默认值   | 说明   |
| --------- | ---- | -------- | ------ |
| memory_id | str  | required | 记忆ID |

#### 示例代码

```python
memory_tool.execute("remove", memory_id="a1b2c3d4e5f6")
# 输出: ✅ 记忆已删除 (ID: a1b2c3d4...)
```

### 3.9 clear_all - 清空所有记忆

#### 功能说明

清空当前用户的所有记忆。

#### 示例代码

```python
memory_tool.execute("clear_all")
# 输出: 🗑️ 已清空所有记忆 (共 128 条)
```

## 4. 完整使用示例

### 4.1 创建学习助手

```python
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import MemoryTool

# 创建LLM和Agent
llm = HelloAgentsLLM()
agent = SimpleAgent(
    name="学习助手",
    llm=llm,
    system_prompt="你是一个有记忆的AI学习助手"
)

# 配置记忆工具
memory_tool = MemoryTool(user_id="student_001")
tool_registry = ToolRegistry()
tool_registry.register_tool(memory_tool)
agent.tool_registry = tool_registry

# 对话并记忆
response = agent.run("我叫张三，正在学习Python机器学习")
print(response)

# 添加学习记录
memory_tool.execute("add",
    content="张三完成了线性回归课程，掌握了基础概念",
    memory_type="episodic",
    importance=0.8
)

# 后续对话时可以调用记忆
response = agent.run("我之前学了什么？")
print(response)
```

### 4.2 个性化推荐系统

```python
# 记录用户偏好
memory_tool.execute("add",
    content="用户喜欢科幻电影，特别是《星际穿越》和《银翼杀手》",
    memory_type="semantic",
    importance=0.9,
    category="preference",
    genre="sci-fi"
)

memory_tool.execute("add",
    content="用户上周观看了《沙丘》，给予了很高评价",
    memory_type="episodic",
    importance=0.8,
    movie_title="Dune",
    rating=9.5
)

# 推荐时检索相关记忆
result = memory_tool.execute("search",
    query="科幻电影推荐",
    memory_type="semantic",
    limit=5
)
print(result)
```

### 4.3 自动记忆管理

```python
# 定期整合工作记忆到长期记忆
def auto_consolidate():
    result = memory_tool.execute("consolidate",
        from_type="working",
        to_type="episodic",
        importance_threshold=0.7
    )
    print(result)

# 定期清理过期记忆
def auto_cleanup():
    # 清理30天前的不重要记忆
    result1 = memory_tool.execute("forget",
        strategy="time_based",
        max_age_days=30
    )

    # 清理重要性低于0.2的记忆
    result2 = memory_tool.execute("forget",
        strategy="importance_based",
        threshold=0.2
    )

    print(f"{result1}\n{result2}")

# 获取记忆摘要
def get_memory_status():
    result = memory_tool.execute("summary")
    print(result)
    return result
```

## 5. 最佳实践

### 5.1 记忆类型选择

| 场景           | 推荐记忆类型 | 理由           |
| -------------- | ------------ | -------------- |
| 当前对话上下文 | working      | 临时、快速访问 |
| 历史事件记录   | episodic     | 需要时间序列   |
| 概念和知识     | semantic     | 需要关系推理   |
| 图像/音频      | perceptual   | 多模态支持     |

### 5.2 重要性设置

| 重要性范围 | 适用场景           |
| ---------- | ------------------ |
| 0.0-0.3    | 临时信息、测试数据 |
| 0.4-0.6    | 一般对话、普通事件 |
| 0.7-0.8    | 重要事件、学习记录 |
| 0.9-1.0    | 核心知识、关键偏好 |

### 5.3 性能优化

- 合理设置`limit`参数，避免返回过多结果
- 使用`memory_type`过滤，减少搜索范围
- 定期执行`forget`清理过期记忆
- 根据场景选择合适的检索策略

### 5.4 错误处理

```python
# 添加记忆时捕获异常
try:
    result = memory_tool.execute("add",
        content="记忆内容",
        memory_type="semantic",
        importance=0.8
    )
    if "✅" in result:
        print("记忆添加成功")
    else:
        print("记忆添加失败")
except Exception as e:
    print(f"发生错误: {e}")
```
