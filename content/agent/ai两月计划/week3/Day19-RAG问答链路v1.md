# Day 19：RAG 问答链路 v1

## 学习目标

经过前三天的学习，我们已经掌握了 RAG 系统的各个组件：文档解析与清洗、文本切分、Embedding 和向量数据库。今天是把这些组件串起来的一天——构建一个端到端的 RAG 问答链路。

这个 v1 版本不追求完美，追求的是"跑通"。从用户提问到获得答案，整个链路能走通、能回答基本问题、能显示引用来源。有了这个基线版本，明天才能在上面做优化。

学完今天，你将拥有一个可以演示的 RAG Demo。它可能还不够好，可能有些问题回答不了，但它是一个真实的、可运行的系统。拿这个 Demo 给别人看，别人能理解 RAG 是什么、能做什么。

---

## 核心概念

### 一、用户问题处理

用户问题处理是 RAG 在线管线的第一个环节。它的任务不只是"拿到问题文本"，还包括一些必要的预处理。

**问题清洗。** 用户的输入可能包含多余的空格、特殊字符、表情符号等。虽然这些问题不影响 Embedding 的效果（模型本身有一定的鲁棒性），但在展示和日志记录时，干净的问题文本更易读。

**问题分类。** 不是所有用户输入都是适合 RAG 回答的问题。有些是闲聊（"你好"），有些是命令（"帮我生成一份报告"），有些是超出知识库范围的提问（"今天天气怎么样"）。在 MVP 阶段可以不做分类，但在生产系统中，需要判断用户输入是否是"知识库问答"类型的请求。如果不是，直接走其他处理路径。

**问题补全。** 在对话场景中，用户的提问可能包含代词或省略。比如用户先问了"退货政策是什么"，接着问"那换货呢？"。第二个问题的"换货"需要结合上下文理解。在 v1 版本中可以不处理多轮对话，但至少要意识到这个问题的存在。

**工程建议。** v1 阶段保持简单：接收问题文本，做基本清洗（去除首尾空白、合并多余空格），然后直接进入向量化环节。对话上下文补全、问题分类等高级功能留到 v2。

### 二、问题向量化

把用户的问题通过 Embedding 模型转换为向量。这和 Day 18 离线阶段对 Chunk 做的向量化是同一个操作，使用同一个模型。

**需要注意的点：**

- **模型一致性。** 必须使用和文档入库时相同的 Embedding 模型。这个在 Day 18 已经强调过了，但它是如此重要，值得再提醒一次。
- **单条 vs 批量。** 在线查询通常一次只处理一个问题，不需要批量。但如果你要支持高并发（多个用户同时提问），需要考虑 Embedding API 的并发能力和速率限制。
- **缓存优化。** 如果多个用户问了相同或非常相似的问题，可以缓存问题的向量，避免重复调用 Embedding API。这是一个可选优化，v1 可以不做。
- **延迟记录。** 记录问题向量化的耗时，作为系统性能监控的基础数据。通常 Embedding API 的响应时间在 50-200ms。

### 三、检索 top-k

用问题向量在向量数据库中检索最相似的 k 个 Chunk。

**参数选择。** v1 阶段建议 k=3 或 k=5。这个值不需要过度优化，先用一个中间值，后面根据测试效果调整。

**相似度阈值。** top-k 检索总是会返回 k 个结果，即使这些结果和问题一点也不相关。你需要设定一个相似度阈值（比如余弦相似度 0.5），低于阈值的结果视为"未检索到相关内容"。这样当知识库中确实没有答案时，系统可以诚实地回答"未找到相关信息"，而不是硬凑一个不相关的答案。

**检索结果的格式。** 每个检索结果应该包含：
- Chunk 的原始文本
- 相似度分数
- 元数据（来源文档、章节、页码等）

**工程建议。** 用向量数据库的 similarity_search_with_score 方法，返回带分数的结果。设定阈值过滤低分结果。

### 四、Prompt 组装

把检索到的 Chunk 和用户问题组合成发送给大模型的 Prompt。

**Prompt 模板设计：**

```
你是一个专业的客服助手。请基于以下知识库内容回答用户的问题。如果知识库中没有相关信息，请诚实地说"未找到相关信息"，不要编造答案。

知识库内容：
{context}

用户问题：
{question}

请给出准确的答案，并引用知识库中的具体内容。
```

**Context 的拼接方式：**

把多个 Chunk 拼接成一个字符串。每个 Chunk 前面可以加一个标识符（如 [文档1]、[文档2]），方便模型在答案中引用。

**Chunk 的排序：**

按相似度从高到低排序，把最相关的 Chunk 放在前面。这样模型更容易关注到真正相关的内容。

**工程建议。** 用 f-string 或模板引擎（Jinja2）组装 Prompt。控制 Context 的总长度，避免超过模型的上下文窗口。

### 五、答案生成

把组装好的 Prompt 发给大模型，生成最终答案。

**模型选择。**

- **闭源模型**（GPT-4、Claude 3）：质量好，但成本高。适合对答案质量要求高的场景。
- **开源模型**（Llama 3、Mistral）：可以自部署，成本低。适合大规模应用或对成本敏感的场景。

**参数设置：**

- **Temperature**。设为 0 或低值（0.1-0.3）。RAG 场景需要的是事实性回答，不需要创造性。Temperature 越低，输出越确定。
- **Max Tokens**。根据需要的答案长度设置。通常 200-500 Token 足够。

**流式输出。** v1 版本可以不支持流式，直接返回完整答案。如果要提升用户体验，可以实现流式输出，让答案逐字显示。

**工程建议。** v1 用 OpenAI API 或 Anthropic API，快速验证。后期考虑成本优化，可以切换到开源模型或自部署。

### 六、引用标注

在答案中标注引用来源，这是 RAG 区别于普通对话的重要特征。

**引用方式：**

- **内联引用。** 在答案中直接说明来源："根据《退货政策》第 3 条，退货时间为 7 天。"
- **脚注引用。** 在答案后列出引用的 Chunk："退货时间为 7 天[1]。[1] 退货政策 v2.0，第 3 条"
- **链接引用。** 如果知识库有在线版本，可以提供链接："更多信息请查看：https://docs.example.com/returns#section3"

**引用粒度：**

- **文档级别。** 只引用文档名称，最简单但不够精确。
- **章节级别。** 引用章节或页码，更精确。
- **片段级别。** 引用具体的 Chunk ID，最精确但可能太技术化。

**工程建议。** v1 版本实现章节级别引用：从元数据中提取文档名和章节名，在答案模板中加入引用信息。

### 七、完整的代码示例

```python
import os
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

class SimpleRAG:
    def __init__(self):
        # 初始化向量数据库
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.openai_client = OpenAI()

        # 获取或创建 collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )
        )

    def query(self, question: str, top_k: int = 3) -> Dict:
        """RAG 查询主流程"""

        # 1. 问题清洗
        question = question.strip()

        # 2. 检索
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k
        )

        # 3. 检查相似度阈值
        if not results["documents"][0]:
            return {
                "answer": "未找到相关信息",
                "sources": []
            }

        # 4. 组装 Context
        context_parts = []
        sources = []

        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            similarity = 1 - distance  # Chroma 用距离，转换为相似度
            if similarity < 0.5:  # 阈值过滤
                continue

            context_parts.append(f"[来源{i+1}] {doc}")
            sources.append({
                "content": doc[:100] + "...",
                "similarity": round(similarity, 3),
                "metadata": metadata
            })

        if not context_parts:
            return {
                "answer": "未找到相关信息",
                "sources": []
            }

        context = "\n\n".join(context_parts)

        # 5. 组装 Prompt
        prompt = f"""你是一个专业的客服助手。请基于以下知识库内容回答用户的问题。如果知识库中没有相关信息，请诚实地说"未找到相关信息"，不要编造答案。

知识库内容：
{context}

用户问题：
{question}

请给出准确的答案，并引用知识库中的具体内容。"""

        # 6. 生成答案
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": sources
        }

# 使用示例
rag = SimpleRAG()
result = rag.query("退货的时限是多少天？")
print(result["answer"])
print("引用来源：")
for source in result["sources"]:
    print(f"- 相似度: {source['similarity']}, 内容: {source['content']}")
```

### 八、v1 版本的局限

这个 v1 版本能跑通，但有很多局限：

- 没有查询改写，用户问题表述不精确时检索效果差
- 没有混合检索（关键词+向量），某些事实性问答效果不好
- 没有重排序，检索到的 Chunk 可能不是最相关的
- 没有上下文压缩，过长的 Context 可能影响模型理解
- 没有多轮对话，无法处理追问

这些局限正是 Day 20 要解决的问题。

## 今日总结

RAG 问答链路包含六个环节：问题处理、向量化、检索、Prompt 组装、答案生成、引用标注。

v1 版本追求"跑通"，不追求完美。

参数选择：top-k 用 3-5，相似度阈值用 0.5，temperature 用 0-0.3。

引用标注是 RAG 的重要特征，让答案可追溯。

完整的代码示例展示了从问题到答案的端到端流程。

v1 版本有很多局限，这些是明天优化的目标。

## 明日预告

今天我们跑通了 RAG 链路 v1。明天是优化的一天——学习 Query Rewrite、Multi-query Retrieval、Hybrid Search、Rerank、Context Compression 等优化手段，提升检索质量和答案准确性。
