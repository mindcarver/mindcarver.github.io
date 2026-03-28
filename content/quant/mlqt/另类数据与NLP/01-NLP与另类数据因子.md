# NLP 与另类数据因子

> 从情绪分析到卫星图像，系统掌握非传统数据源的因子挖掘方法

---

## Part 1：NLP 在量化中的应用

### 1.1 金融情绪分析

情绪分析是 NLP 在量化中最成熟的应用。核心假设：市场情绪影响资产价格，极端情绪往往预示反转。

#### 1.1.1 词典方法

最简单直接的情绪分析方式，基于预定义的金融情绪词典。

```python
"""
金融情绪分析 — 词典方法（Loughran-McMaster）

Loughran-McMaster 金融情绪词典是金融领域最广泛使用的情绪词典，
包含约 4000 个金融相关词汇，分为正面、负面、不确定性、法律等类别。
"""
import numpy as np
import pandas as pd
import re

# 简化的 LM 金融情绪词典（示例）
LM_POSITIVE = {
    "增长", "盈利", "上涨", "突破", "强势", "乐观", "超预期",
    "创新高", "增持", "回购", "景气", "复苏", "繁荣", "利好",
    "稳健", "领先", "优势", "提升", "改善", "升级", "拓展",
    "growth", "profit", "increase", "beat", "strong", "bullish",
    "outperform", "upgrade", "positive", "exceed", "improve",
}

LM_NEGATIVE = {
    "下跌", "亏损", "暴跌", "崩盘", "风险", "警告", "低于预期",
    "减持", "做空", "衰退", "危机", "违约", "暴跌", "利空",
    "疲软", "下滑", "萎缩", "债务", "诉讼", "调查", "处罚",
    "decline", "loss", "drop", "crash", "risk", "warning",
    "miss", "weak", "negative", "cut", "reduce", "default",
}

LM_UNCERTAINTY = {
    "可能", "也许", "不确定", "预计", "估计", "或许", "大概",
    "潜在", "假设", "暂时", "待定", "观望", "难以判断",
    "may", "might", "could", "uncertain", "possibly", "approximately",
    "depend", "estimate", "assume", "tentative", "unclear",
}


def compute_sentiment_scores(text: str) -> dict:
    """
    计算文本的情绪分数

    返回:
        positive_count: 正面词数量
        negative_count: 负面词数量
        uncertainty_count: 不确定词数量
        net_sentiment: 净情绪 = (正面 - 负面) / (正面 + 负面 + 1)
        total_words: 总词数（用于标准化）
    """
    # 中文分词（简化处理，实际应用应使用 jieba）
    tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text.lower())

    pos_count = sum(1 for t in tokens if t in LM_POSITIVE)
    neg_count = sum(1 for t in tokens if t in LM_NEGATIVE)
    unc_count = sum(1 for t in tokens if t in LM_UNCERTAINTY)

    total = max(len(tokens), 1)

    return {
        "positive_count": pos_count,
        "negative_count": neg_count,
        "uncertainty_count": unc_count,
        "positive_ratio": pos_count / total,
        "negative_ratio": neg_count / total,
        "net_sentiment": (pos_count - neg_count) / (pos_count + neg_count + 1),
        "total_words": total,
    }


# 示例：分析财报文本
earnings_text = "公司第三季度营收增长25%，超过市场预期。但管理层警告下半年可能面临供应链不确定性和原材料成本上升风险。"
scores = compute_sentiment_scores(earnings_text)
print(f"情绪分析结果: {scores}")
# 预期输出: net_sentiment > 0, uncertainty_count > 0
```

#### 1.1.2 FinBERT 深度学习方法

FinBERT 是基于 BERT 微调的金融情绪分析模型，在金融文本上的表现远超通用模型。

```python
"""
金融情绪分析 — FinBERT 深度学习方法

FinBERT 是 Prospect.ai 在金融语料上微调的 BERT 模型，
能够理解金融语境中的反讽、否定和条件句。
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd


class FinBERSentimentAnalyzer:
    """基于 FinBERT 的金融情绪分析器"""

    def __init__(self, model_name="ProsusAI/finbert"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # FinBERT 标签映射
        self.label_map = {0: "positive", 1: "negative", 2: "neutral"}

    def analyze(self, text: str) -> dict:
        """
        分析单条文本的情绪

        返回:
            label: 情绪标签（positive/negative/neutral）
            confidence: 置信度
            probabilities: 各情绪的概率分布
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()

        return {
            "label": self.label_map[predicted_class],
            "confidence": probs[0, predicted_class].item(),
            "probabilities": {
                self.label_map[i]: probs[0, i].item()
                for i in range(3)
            },
        }

    def analyze_batch(self, texts: list, batch_size=16) -> pd.DataFrame:
        """批量分析文本情绪"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            for j in range(len(batch)):
                predicted_class = torch.argmax(probs[j]).item()
                results.append({
                    "text": batch[j][:50] + "...",
                    "label": self.label_map[predicted_class],
                    "confidence": probs[j, predicted_class].item(),
                    "prob_positive": probs[j, 0].item(),
                    "prob_negative": probs[j, 1].item(),
                    "prob_neutral": probs[j, 2].item(),
                })

        return pd.DataFrame(results)


# 使用示例
analyzer = FinBERSentimentAnalyzer()

news_headlines = [
    "公司营收超预期增长30%，董事会批准10亿美元回购计划",
    "Q3净亏损扩大至5亿美元，CEO表示将裁员15%以降低成本",
    "分析师维持持有评级，目标价从50美元上调至55美元",
]

df_sentiment = analyzer.analyze_batch(news_headlines)
print(df_sentiment[["text", "label", "confidence"]])
```

### 1.2 事件抽取

从非结构化文本中提取结构化事件信息，是构建事件驱动策略的基础。

```python
"""
金融事件抽取 — 基于规则 + NER 的混合方法

目标：从新闻中提取 (公司, 事件类型, 时间, 影响) 四元组

事件类型定义:
    - earnings: 财报发布
    - guidance: 业绩指引调整
    - m&a: 并购重组
    - regulation: 监管变化
    - product: 产品发布/召回
    - executive: 高管变动
"""
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class FinancialEvent:
    """金融事件结构"""
    company: str           # 涉及公司
    event_type: str        # 事件类型
    timestamp: str         # 事件时间
    sentiment: float       # 事件情绪 [-1, 1]
    importance: float      # 重要性 [0, 1]
    source: str            # 信息来源


# 简化的规则匹配模式
EVENT_PATTERNS = {
    "earnings": [
        r"(?:发布|公布|披露)(?:第[一二三四]季度|Q[1-4])?(?:财报|业绩|经营数据)",
        r"(?:营收|净利|利润)(?:同比)?(?:增长|下降|增长).*?(\d+\.?\d*)%",
    ],
    "guidance": [
        r"(?:上调|下调|维持).*(?:业绩指引|收入预期|利润预期)",
        r"(?:预计|预期|展望).*(?:全年|下半年).*(?:增长|下降)",
    ],
    "m&a": [
        r"(?:收购|并购|合并|重组|入股)(?:.{0,20}?)(?:公司|集团|企业)",
        r"(?:被).{0,10}?(?:收购|并购)",
    ],
    "regulation": [
        r"(?:证监会|银保监|SEC|FED).{0,20}?(?:处罚|罚款|警告|批准|否决)",
        r"(?:监管|政策).{0,10}?(?:收紧|放松|变化|调整)",
    ],
    "executive": [
        r"(?:任命|聘任|选举|罢免|辞职|离职).{0,10}?(?:CEO|CFO|董事长|总经理)",
        r"(?:高管|董事).{0,10}?(?:变动|变更|辞职)",
    ],
}


def extract_events(text: str, company_name: str = "") -> list:
    """
    从文本中抽取金融事件

    参数:
        text: 新闻文本
        company_name: 目标公司名称（可选）

    返回:
        匹配到的金融事件列表
    """
    events = []

    for event_type, patterns in EVENT_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # 简单情绪判断
                positive_keywords = {"增长", "上调", "收购", "批准", "超预期"}
                negative_keywords = {"下降", "下调", "处罚", "罢免", "亏损"}

                matched_text = match.group()
                sentiment = 0.0
                for kw in positive_keywords:
                    if kw in matched_text:
                        sentiment += 0.3
                for kw in negative_keywords:
                    if kw in matched_text:
                        sentiment -= 0.3

                events.append(FinancialEvent(
                    company=company_name,
                    event_type=event_type,
                    timestamp="",
                    sentiment=np.clip(sentiment, -1, 1),
                    importance=0.7,  # 规则匹配置信度
                    source="rule_based",
                ))

    # 去重（同一事件类型只保留一个）
    seen_types = set()
    unique_events = []
    for e in events:
        if e.event_type not in seen_types:
            seen_types.add(e.event_type)
            unique_events.append(e)

    return unique_events
```

### 1.3 LLM 应用 — RAG 系统

大语言模型（LLM）在量化研究中的最实用场景是 RAG（检索增强生成），将研报、新闻等文档与实时查询结合。

```python
"""
LLM + RAG 量化研究助手 — 架构示意

核心流程:
    1. 文档索引：将研报/新闻向量化并存入向量数据库
    2. 检索：根据查询检索相关文档片段
    3. 生成：LLM 基于检索结果回答问题

适用场景:
    - 快速研报摘要
    - 公司基本面问答
    - 行业趋势分析
    - 事件影响评估
"""

# RAG 架构示意图
RAG_ARCHITECTURE = """
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG 量化研究系统                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐ │
│  │   文档层          │    │   检索层          │    │   生成层       │ │
│  │                  │    │                  │    │               │ │
│  │  研报 → 分段     │ →  │  Embedding 模型  │ →  │  LLM (GPT-4  │ │
│  │  新闻 → 清洗     │    │  向量数据库检索   │    │   /Claude)    │ │
│  │  财报 → 结构化   │    │  相关性排序      │    │  增强生成     │ │
│  │                  │    │                  │    │               │ │
│  └──────────────────┘    └──────────────────┘    └───────────────┘ │
│                                                                     │
│  查询示例:                                                          │
│    "宁德时代最近的产能扩张计划如何影响2026年营收预期？"               │
│    → 检索: 研报段落、新闻、公告                                     │
│    → 生成: 综合分析 + 数据引用                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
"""

# 简化的 RAG 实现（概念性）
def build_rag_query(
    question: str,
    embedding_model,
    vector_db,
    llm_client,
    top_k: int = 5,
) -> str:
    """
    RAG 查询流程

    步骤:
        1. 将问题向量化
        2. 在向量数据库中检索 top-k 相关文档
        3. 构造增强 prompt
        4. 调用 LLM 生成回答
    """
    # 步骤 1：问题向量化
    question_embedding = embedding_model.embed_query(question)

    # 步骤 2：检索相关文档
    relevant_docs = vector_db.search(question_embedding, top_k=top_k)

    # 步骤 3：构造增强 prompt
    context = "\n\n".join([
        f"[文档 {i+1}] {doc.text}\n来源: {doc.source}, 日期: {doc.date}"
        for i, doc in enumerate(relevant_docs)
    ])

    prompt = f"""你是一位专业的量化研究分析师。请基于以下参考文档回答问题。
回答要求：数据驱动、引用具体来源、给出量化判断。

参考文档:
{context}

问题: {question}

请给出你的分析:"""

    # 步骤 4：LLM 生成
    response = llm_client.generate(prompt)
    return response
```

### 1.4 文本因子构建

将 NLP 分析结果转化为可量化的因子，是连接 NLP 和量化策略的关键桥梁。

#### 因子类型总览

| 因子类型 | 数据源 | 构建方法 | 典型更新频率 |
|---------|--------|---------|------------|
| **情绪因子** | 新闻、研报、社交媒体 | FinBERT / 词典法 | 日频/分钟频 |
| **关注度因子** | 搜索热度、新闻数量 | 统计计数、异常检测 | 日频 |
| **不确定性因子** | 财报会议、研报 | LM 不确定性词典 + NER | 季度/月频 |
| **事件冲击因子** | 结构化事件 | 事件类型 + 情绪 | 事件驱动 |
| **文本相似度因子** | 公司文档 | 余弦相似度 / embedding | 周频/月频 |

```python
"""
文本因子构建示例

因子定义:
    - sentiment_score: 个股日度情绪综合得分
    - attention_score: 个股关注度异常值
    - uncertainty_index: 不确定性指数
"""
import pandas as pd
import numpy as np


def build_sentiment_factor(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建个股日度情绪因子

    输入:
        news_df: 包含 [date, stock_code, headline, source] 的新闻数据

    输出:
        因子值 DataFrame: [date, stock_code, sentiment_factor]
    """
    # 1. 计算每条新闻的情绪
    analyzer = FinBERSentimentAnalyzer()

    results = []
    for _, row in news_df.iterrows():
        sentiment = analyzer.analyze(row["headline"])
        results.append({
            "date": row["date"],
            "stock_code": row["stock_code"],
            "sentiment_label": sentiment["label"],
            "sentiment_conf": sentiment["confidence"],
            "prob_positive": sentiment["probabilities"]["positive"],
            "prob_negative": sentiment["probabilities"]["negative"],
            "prob_neutral": sentiment["probabilities"]["neutral"],
        })

    df = pd.DataFrame(results)

    # 2. 按天聚合
    daily = df.groupby(["date", "stock_code"]).agg(
        news_count=("stock_code", "count"),
        avg_positive=("prob_positive", "mean"),
        avg_negative=("prob_negative", "mean"),
        weighted_sentiment=(
            "prob_positive", lambda x: np.average(
                df.loc[x.index, "prob_positive"],
                weights=df.loc[x.index, "sentiment_conf"]
            )
        ),
    ).reset_index()

    # 3. 构建情绪因子（加权正面概率 - 加权负面概率）
    daily["sentiment_factor"] = (
        daily["avg_positive"] - daily["avg_negative"]
    ) * np.log1p(daily["news_count"])  # 新闻数量加权

    # 4. 横截面标准化（Z-Score）
    daily["sentiment_factor"] = daily.groupby("date")["sentiment_factor"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    return daily[["date", "stock_code", "sentiment_factor"]]


def build_attention_factor(news_df: pd.DataFrame,
                           lookback_window: int = 60) -> pd.DataFrame:
    """
    构建关注度因子（异常新闻数量）

    核心逻辑: 当天新闻数量偏离过去 N 天均值的程度
    """
    daily_count = news_df.groupby(["date", "stock_code"]).size().reset_index(name="count")

    # 计算滚动均值和标准差
    daily_count = daily_count.sort_values(["stock_code", "date"])
    daily_count["rolling_mean"] = daily_count.groupby("stock_code")["count"].transform(
        lambda x: x.rolling(lookback_window, min_periods=20).mean()
    )
    daily_count["rolling_std"] = daily_count.groupby("stock_code")["count"].transform(
        lambda x: x.rolling(lookback_window, min_periods=20).std()
    )

    # 异常得分（Z-Score）
    daily_count["attention_factor"] = (
        (daily_count["count"] - daily_count["rolling_mean"])
        / (daily_count["rolling_std"] + 1e-8)
    )

    # 横截面标准化
    daily_count["attention_factor"] = daily_count.groupby("date")["attention_factor"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    return daily_count[["date", "stock_code", "attention_factor"]].dropna()
```

---

## Part 2：另类数据因子

### 2.1 卫星数据

卫星图像可以提供传统数据无法获取的实体经济活动信息。

| 应用场景 | 监测指标 | 预测目标 | 领先时间 |
|---------|---------|---------|---------|
| 零售 | 停车场车辆数 | 同店销售额 | 2-4 周 |
| 能源 | 油罐阴影面积 | 原油库存 | 1-2 周 |
| 农业 | NDVI 植被指数 | 农作物产量 | 1-3 月 |
| 工业 | 工厂热信号 | 产能利用率 | 1-4 周 |
| 物流 | 港口集装箱密度 | 进出口贸易 | 2-6 周 |

```python
"""
卫星数据处理流程（概念性）

注意：实际卫星图像处理需要专业的遥感工具和标注数据
这里展示数据处理框架和因子构建逻辑
"""
import numpy as np
import pandas as pd


def build_satellite_factor(parking_counts: pd.DataFrame,
                           revenue_data: pd.DataFrame) -> pd.DataFrame:
    """
    基于停车场车辆数构建零售销售预测因子

    参数:
        parking_counts: [date, store_id, vehicle_count, confidence]
        revenue_data: [date, store_id, revenue]（用于回归校准）

    返回:
        去噪后的车辆数因子
    """
    # 1. 数据清洗：去除低置信度观测
    clean = parking_counts[parking_counts["confidence"] > 0.8].copy()

    # 2. 去除季节性（使用同比变化而非绝对值）
    clean["yoy_change"] = clean.groupby("store_id")["vehicle_count"].pct_change(
        periods=252  # 年化对比
    )

    # 3. 去噪：移动平均平滑
    clean["smoothed_yoy"] = clean.groupby("store_id")["yoy_change"].transform(
        lambda x: x.rolling(30, min_periods=7).mean()
    )

    # 4. 与实际销售数据回归校准（如有）
    if revenue_data is not None:
        merged = clean.merge(revenue_data, on=["date", "store_id"])
        correlation = merged["smoothed_yoy"].corr(merged["revenue"])
        print(f"停车场车辆数 vs 销售额相关系数: {correlation:.3f}")

    return clean[["date", "store_id", "smoothed_yoy"]].rename(
        columns={"smoothed_yoy": "satellite_factor"}
    )
```

### 2.2 信用卡与消费数据

信用卡交易数据提供消费者支出的近实时视图。

```python
"""
信用卡消费数据因子构建

数据特点:
    - 近实时（T+1 ~ T+3）
    - 覆盖面有限（仅合作银行/卡的持卡人）
    - 需要去偏（样本不代表总体）
"""
import pandas as pd
import numpy as np


def build_consumer_spending_factor(
    card_transactions: pd.DataFrame,
    industry_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    构建消费趋势因子

    输入:
        card_transactions: [date, merchant_category, transaction_amount]
        industry_mapping: [merchant_category, sector, company_exposure]
    """
    # 1. 按行业聚合日度消费额
    daily_spend = card_transactions.groupby(["date", "merchant_category"]).agg(
        total_amount=("transaction_amount", "sum"),
        transaction_count=("transaction_amount", "count"),
        avg_ticket=("transaction_amount", "mean"),
    ).reset_index()

    # 2. 与行业映射合并
    daily_spend = daily_spend.merge(industry_mapping, on="merchant_category")

    # 3. 计算消费增速（消除基线效应）
    daily_spend["spend_growth"] = daily_spend.groupby(
        ["sector", "company_exposure"]
    )["total_amount"].pct_change(periods=7)  # 周同比

    # 4. 标准化
    daily_spend["consumer_factor"] = daily_spend.groupby("date")["spend_growth"].transform(
        lambda x: (x - x.median()) / (x.std() + 1e-8)
    )

    return daily_spend[["date", "sector", "company_exposure", "consumer_factor"]]
```

### 2.3 供应链数据

供应链数据可以揭示上下游传导效应，提前预测公司业绩变化。

```python
"""
供应链因子构建

核心思路:
    1. 构建供应链关系图谱（供应商-客户-竞争对手）
    2. 追踪上游业绩变化对下游的影响
    3. 利用信息传导时间差获取 Alpha

示例: 台积电业绩预告 → 苹果/英伟达/AMD 股价反应
"""

# 供应链关系示例
SUPPLY_CHAIN = {
    "台积电": {
        "downstream": ["苹果", "英伟达", "AMD", "高通"],
        "lead_time_days": 30,  # 上游业绩变化传导到下游的领先天数
    },
    "中芯国际": {
        "downstream": ["韦尔股份", "兆易创新", "中微公司"],
        "lead_time_days": 20,
    },
}

def compute_supply_chain_signal(
    upstream_data: pd.DataFrame,  # 上游公司指标变化
    stock_returns: pd.DataFrame,  # 下游公司收益率
) -> pd.DataFrame:
    """
    计算供应链传导信号

    逻辑: 上游指标恶化 → 下游订单减少 → 下游未来业绩承压
    """
    signals = []

    for upstream, config in SUPPLY_CHAIN.items():
        lead_time = config["lead_time_days"]
        for downstream in config["downstream"]:
            # 将上游数据按领先天数对齐到下游
            upstream_shifted = upstream_data[upstream].shift(lead_time)

            # 计算上游变化与下游收益的相关性
            merged = pd.DataFrame({
                "upstream_change": upstream_shifted,
                "downstream_return": stock_returns[downstream],
            }).dropna()

            if len(merged) > 30:
                corr = merged["upstream_change"].corr(merged["downstream_return"])
                signals.append({
                    "upstream": upstream,
                    "downstream": downstream,
                    "correlation": corr,
                    "lead_time": lead_time,
                })

    return pd.DataFrame(signals)
```

### 2.4 Web 爬虫与链上数据

```python
"""
Web 爬虫数据与链上数据概述

Web 爬虫数据:
    - 搜索热度（Google Trends / 百度指数）
    - 产品价格监控（电商比价）
    - APP 活跃度（Sensor Tower / SimilarWeb）
    - 招聘数据（LinkedIn / Boss 直聘）
    - 政府公开数据（专利、招投标、行政处罚）

链上数据（Crypto 量化专用）:
    - 交易所净流入/流出
    - 大户地址持仓变化（鲸鱼追踪）
    - DeFi 协议 TVL 变化
    - Gas 费用网络拥堵度
    - NFT 成交量与地板价
"""

# APP 活跃度因子示例
def build_app_activity_factor(app_rankings: pd.DataFrame) -> pd.DataFrame:
    """
    基于 APP 下载排名变化构建因子

    逻辑: APP 排名突增 → 用户活跃度上升 → 营收预期改善
    """
    # 计算排名变化
    rankings = app_rankings.sort_values(["app_id", "date"])
    rankings["rank_change"] = rankings.groupby("app_id")["rank"].diff()

    # 排名大幅上升 = 正面信号
    rankings["activity_signal"] = -rankings["rank_change"]  # 负号：排名下降是好事

    # 标准化
    rankings["activity_factor"] = rankings.groupby("date")["activity_signal"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    return rankings
```

### 2.5 另类数据评估框架

在采购或使用另类数据前，必须系统性评估其价值。

| 评估维度 | 关键问题 | 评估方法 |
|---------|---------|---------|
| **信号质量** | 能否预测资产收益？ | IC/IR 分析、回归检验 |
| **独占性** | 其他机构也能获取吗？ | 供应商客户数、替代数据源 |
| **时效性** | 数据更新频率如何？ | 延迟测试、频率对比 |
| **衰减速度** | Alpha 能持续多久？ | 滚动 IC 测试、半衰期估算 |
| **覆盖度** | 能覆盖多少标的？ | 缺失率分析、样本偏差 |
| **成本效益** | 数据成本 vs 预期收益 | 成本/收益比、ROI 估算 |
| **合规风险** | 数据使用是否合规？ | 马赛克理论、GDPR 审查 |

```python
"""
另类数据信号质量评估

核心指标:
    - IC (Information Coefficient): 因子值与未来收益的相关系数
    - IR (Information Ratio): IC 的均值 / IC 的标准差
    - 衰减半衰期: IC 降至初始值一半所需时间
"""
import pandas as pd
import numpy as np


def evaluate_alternative_signal(
    factor_values: pd.DataFrame,  # [date, stock, factor_value]
    forward_returns: pd.DataFrame,  # [date, stock, forward_return]
    decay_test_periods: list = None,
) -> dict:
    """
    评估另类数据因子的信号质量

    返回:
        评估报告字典，包含 IC、IR、衰减分析等
    """
    if decay_test_periods is None:
        decay_test_periods = [1, 5, 10, 20, 60]  # 未来 1/5/10/20/60 天收益

    merged = factor_values.merge(forward_returns, on=["date", "stock"])
    results = {}

    for period in decay_test_periods:
        return_col = f"fwd_return_{period}d"
        if return_col not in merged.columns:
            continue

        # 按日期计算 IC（Spearman 相关系数）
        daily_ic = merged.groupby("date").apply(
            lambda x: x["factor_value"].corr(x[return_col], method="spearman")
        )

        results[period] = {
            "mean_ic": daily_ic.mean(),
            "ic_std": daily_ic.std(),
            "ir": daily_ic.mean() / (daily_ic.std() + 1e-8),
            "ic_positive_ratio": (daily_ic > 0).mean(),
        }

    return results


# 评估报告模板
def print_evaluation_report(results: dict):
    """打印评估报告"""
    print("=" * 60)
    print("另类数据因子评估报告")
    print("=" * 60)
    print(f"{'持有期':>8} {'平均IC':>8} {'IR':>8} {'IC>0占比':>10}")
    print("-" * 40)
    for period, metrics in results.items():
        print(
            f"{period:>6}d  "
            f"{metrics['mean_ic']:>8.4f}  "
            f"{metrics['ir']:>8.4f}  "
            f"{metrics['ic_positive_ratio']:>10.1%}"
        )
    print("-" * 40)

    # 判断因子质量
    best_period = max(results, key=lambda k: abs(results[k]["ir"]))
    best_ir = abs(results[best_period]["ir"])
    if best_ir > 0.5:
        print(f"评估结论: 优质因子 (最佳IR={best_ir:.2f} @ {best_period}d)")
    elif best_ir > 0.3:
        print(f"评估结论: 可用因子 (最佳IR={best_ir:.2f} @ {best_period}d)")
    else:
        print(f"评估结论: 信号较弱 (最佳IR={best_ir:.2f} @ {best_period}d)")
```

### 2.6 工程挑战

另类数据在实际使用中面临大量工程挑战。

#### 2.6.1 Point-in-Time 数据

```python
"""
Point-in-Time (PIT) 数据的重要性

问题: 财务数据在发布后可能被修正（重述），如果使用最新版本的历史数据，
      就会产生 look-ahead bias（未来信息泄露）。

解决: 必须使用"当时可用的"数据版本，即 Point-in-Time 数据。

示例:
    公司在 2025-03-30 发布 2024 年报，营收 100 亿
    公司在 2025-06-15 修正 2024 年报，营收 98 亿

    PIT 正确做法:
        - 2025-03-30 ~ 2025-06-14: 使用营收 = 100 亿
        - 2025-06-15 之后: 使用营收 = 98 亿
"""

# PIT 数据重建示例
def build_pit_data(revisions_df: pd.DataFrame) -> pd.DataFrame:
    """
    从修正记录构建 PIT 数据

    输入:
        revisions_df: [report_date, stock, field, original_value,
                       revised_value, revision_date]
    """
    pit_records = []

    for _, row in revisions_df.iterrows():
        # 原始值在报告日到修正日之间有效
        pit_records.append({
            "date": row["report_date"],
            "stock": row["stock"],
            "field": row["field"],
            "value": row["original_value"],
        })
        # 修正值在修正日之后有效
        pit_records.append({
            "date": row["revision_date"],
            "stock": row["stock"],
            "field": row["field"],
            "value": row["revised_value"],
        })

    pit = pd.DataFrame(pit_records)
    pit = pit.sort_values(["stock", "field", "date"])

    # 前向填充：每个日期使用当时可用的最新值
    pit_data = pit.set_index(["stock", "field", "date"])["value"].unstack(level="date")
    pit_data = pit_data.ffill(axis=1).T  # 按日期前向填充

    return pit_data
```

#### 2.6.2 常见工程挑战汇总

| 挑战 | 具体问题 | 解决方案 |
|------|---------|---------|
| **时区对齐** | 全球数据时区不一致 | 统一使用 UTC，按交易日历对齐 |
| **数据缺失** | 另类数据覆盖不全 | 插值、因子中性化、缺失标记 |
| **频率不匹配** | 卫星周频 vs 新闻日频 | 低频向上采样或高频向下采样 |
| **供应商风险** | 数据源中断或质量下降 | 多源冗余、监控告警 |
| **清洗复杂** | 原始数据噪声大 | 自动化清洗管道、人工抽检 |
| **合规审查** | GDPR、MAR 等法规限制 | 法务审核、数据脱敏、马赛克理论 |
| **存储成本** | 卫星图像等数据量大 | 增量存储、冷热分层、Parquet 压缩 |

### 核心总结

```
┌──────────────────────────────────────────────────────────────────┐
│                NLP 与另类数据因子核心要点                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 情绪分析两板斧                                                │
│     词典法（快但粗）+ FinBERT（准但慢），推荐混合使用               │
│                                                                  │
│  2. 因子构建三步骤                                                │
│     原始信号 → 横截面标准化 → IC/IR 评估                          │
│                                                                  │
│  3. 另类数据的核心价值在于独占性和时效性                           │
│     不独占的数据很快被套利消灭，不及时的信号没有预测力               │
│                                                                  │
│  4. Point-in-Time 是生命线                                        │
│     任何使用修正后历史数据的行为都是未来信息泄露                     │
│                                                                  │
│  5. 数据质量 > 数据数量                                           │
│     一个干净的信号比十个噪声数据更有价值                            │
│                                                                  │
│  6. LLM/RAG 是研究效率工具，不是 Alpha 来源                      │
│     用 LLM 加速信息处理，用传统方法验证因子                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```
