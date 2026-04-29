# Day 13：行业分析 API Demo

## 学习目标

昨天你设计了后端的结构框架。今天要把这个框架跑通，做一个完整的、可演示的行业分析 API Demo。

这个 Demo 不只是一个技术验证。它要模拟一个真实的产品场景：用户输入一个行业名称，系统自动完成需求解析、Prompt 模板选择、模型调用、结构化输出、结果保存、报告生成，最终返回一份结构化的行业分析结果和可选的 Markdown 报告。

这是 Week 2 所有学习内容的综合实战。你要用到 Day 8 的 LLM Client、Day 9 的结构化输出、Day 10 的 Function Calling、Day 11 的 Prompt 模板、Day 12 的后端架构。把这些能力串起来，形成一个完整的调用链路。

完成今天的学习后，你应该拥有一个可以给别人演示的 AI 行业分析服务——输入行业名称，输出分析结果。这听起来简单，但把所有环节串通、处理各种异常情况、保证输出质量，需要综合运用你这一周学到的所有技能。

## 核心概念

### 一、输入行业名称

行业分析的第一步是接收用户输入。看似简单，但输入的规范化处理决定了后续流程的稳定性。

用户输入的行业名称可能千奇百怪：有人输入"半导体"，有人输入"芯片行业"，有人输入"IC 制造"，有人输入"semiconductor"。这些名称指向同一个行业，但模型可能把它们当作不同的行业来分析。

输入规范化的策略：

同义词映射。维护一个行业名称的映射表——"芯片行业"映射到"半导体"，"IC 制造"映射到"半导体"。在调用模型之前先做一次映射。这个映射表可以是一个简单的字典，也可以调用模型来判断两个名称是否指向同一行业。

输入校验。检查输入是否为空、是否过短（少于 2 个字可能不是有效的行业名称）、是否过长（超过 50 个字可能包含多余信息）。校验不通过时直接返回错误，不浪费模型调用。

输入清洗。去除多余的空白字符、特殊符号。用户输入"  半导体行业  "应该被清洗为"半导体"。

意图识别。用户可能不是在输入行业名称，而是在描述一个问题或需求。比如输入"我想了解制造业的 AI 机会"。这种情况下需要先提取关键信息（行业名称是"制造业"），再进行分析。

这些预处理步骤虽然不是 AI 的核心能力，但它们决定了系统的健壮性。一个不做输入规范化的系统，用户输入稍微不同就会得到不一致的结果。

### 二、任务解析

任务解析是把用户的输入转化为一组具体的分析任务。

用户说"分析半导体行业"，系统需要理解这个请求包含哪些分析维度。一个好的任务解析应该把一个模糊的请求拆解为具体的子任务：

行业概况分析。半导体行业的定义、范围、主要细分领域。

市场规模分析。全球和国内市场规模、增长趋势、预测数据。

竞争格局分析。主要玩家、市场份额、竞争壁垒。

技术趋势分析。技术发展方向、关键技术节点、创新机会。

AI 应用机会分析。半导体行业中哪些环节适合 AI 赋能、具体的 AI 场景。

风险分析。行业面临的主要风险、不确定性因素。

任务解析可以通过硬编码的规则实现，也可以通过模型来实现。硬编码规则的好处是可控、稳定，但不够灵活。模型实现的好处是灵活、可扩展，但需要更多的验证和测试。

实践中可以结合两者：基础维度用硬编码规则，特殊需求用模型动态生成。

### 三、Prompt 模板选择

Prompt 模板的选择决定了输出的质量和稳定性。

对于不同的行业、不同的分析维度，应该使用不同的 Prompt 模板。半导体行业的分析要关注技术节点、产能分布、设备国产化率。新能源行业的分析要关注政策支持、技术路线（锂电 vs 氢能）、成本变化。用一个通用模板分析所有行业，结果往往不够深入。

Prompt 模板的选择策略：

按行业分类。维护一个行业到模板的映射表——半导体行业用技术密集型模板、消费品行业用市场导向型模板、制造业用成本分析型模板。

按分析深度分类。快速概览（50 字以内）、标准分析（200 字左右）、深度报告（500 字以上）。不同深度使用不同的 Prompt——深度分析需要更多的上下文和更细的引导。

按输出格式分类。JSON 格式用于程序处理、Markdown 格式用于人类阅读、PPT 格式用于演示展示。不同格式需要不同的 Prompt 指令。

Prompt 模板的实现可以用 Day 11 学到的模板库系统——每个模板有编号、版本、参数占位符、示例输入输出。这样可以追踪哪个模板效果好、哪个需要优化。

### 四、结构化输出

结构化输出是让分析结果能被程序可靠处理的关键。

用户要的是一份结构化的行业分析，而不是一段自由文本。结构化意味着固定的字段、明确的数据类型、可验证的格式。这样才能把结果存入数据库、在界面上展示、用于后续的分析和决策。

行业分析的典型输出结构：

行业基础信息。行业名称、所属板块、产业链位置。

市场规模数据。总市场规模、年增长率、未来 3 年预测。

竞争格局。头部企业名单、市场份额分布、进入壁垒。

技术分析。核心技术、技术趋势、创新机会。

AI 机会。可落地的 AI 场景、预期收益、实施难度。

风险提示。主要风险、不确定性因素、应对建议。

用 JSON Schema 定义这个结构，然后用 Day 9 学到的 JSON Mode 确保模型输出符合这个结构。如果模型输出不符合要求，使用带反馈的重试机制。

### 五、报告生成

结构化数据适合程序处理，但不适合人类阅读。人类更习惯阅读有层次、有格式、有重点的报告。

报告生成的作用是把结构化的 JSON 数据转换为人类友好的 Markdown 报告。

好的报告结构：

标题和概述。行业名称、分析时间、核心结论一句话。

分节详细展开。市场规模、竞争格局、技术趋势、AI 机会各占一节，每节有核心观点和支持数据。

关键数据高亮。用表格展示市场规模数据、用列表列出头部企业、用加粗强调关键数字。

数据来源标注。每个数据点都要注明来源——如"来源：IDC 2024 半导体行业报告"。

风险提示。在报告末尾列出主要风险和不确定性。

报告生成是一个确定性的过程，不需要调用模型。可以用模板引擎（如 Jinja2）实现，也可以直接用 Python 字符串操作。

### 六、结果保存

分析结果应该被保存，而不是每次都重新生成。

结果保存的价值：

缓存。同样的行业名称在 24 小时内再次查询时，直接返回缓存结果，节省模型调用成本。

历史记录。用户可以查看之前分析过的行业，对比不同时间的分析结果。

质量追踪。记录每次分析使用的 Prompt 模板、模型参数、输出质量，用于后续优化。

结果保存的存储选择：

结构化数据（JSON）存数据库。可以用 PostgreSQL 的 JSONB 字段，也可以用 MongoDB 这种原生 JSON 数据库。

非结构化数据（Markdown 报告）存文件系统。文件路径记录在数据库中，通过 API 返回给用户。

### 七、完整链路

把以上所有环节串起来，就是完整的行业分析链路：

用户输入"半导体"→ 输入规范化（同义词映射、输入校验、清洗）→ 任务解析（拆解分析维度）→ Prompt 模板选择（根据行业和分析深度）→ 模型调用（使用 JSON Mode）→ 结构化输出校验（Pydantic 验证，失败则重试）→ 报告生成（Markdown 格式）→ 结果保存（数据库 + 文件系统）→ 返回用户（JSON 数据 + 报告链接）。

这个链路中的每个环节都可能出错。输入为空、模型超时、输出格式错误、数据库连接失败。需要在每个环节都做错误处理，确保一个环节的失败不会导致整个流程崩溃。

## 实战示例

### 完整的行业分析服务

```python
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

class IndustryAnalysisService:
    def __init__(self, llm_client, prompt_template_dir, output_dir):
        self.llm = llm_client
        self.template_dir = prompt_template_dir
        self.output_dir = output_dir

        # 行业同义词映射
        self.industry_synonyms = {
            "芯片行业": "半导体",
            "IC 制造": "半导体",
            "集成电路": "半导体",
            # ... 更多映射
        }

    async def analyze(self, industry: str, focus_areas: List[str] = None):
        task_id = str(uuid.uuid4())

        # 1. 输入规范化
        normalized_input = self._normalize_input(industry, focus_areas)

        # 2. 任务解析
        task_config = await self._parse_task(normalized_input)

        # 3. 模板选择
        prompt = self._select_prompt_template(task_config)

        # 4. 模型调用（结构化输出）
        structured_result = await self._call_llm_with_structure(
            prompt=prompt,
            industry=normalized_input["industry"],
            schema=task_config["output_schema"]
        )

        # 5. 报告生成
        report_path = self._generate_markdown_report(
            task_id=task_id,
            industry=industry,
            data=structured_result
        )

        # 6. 结果保存
        self._save_result(task_id, structured_result, report_path)

        return {
            "task_id": task_id,
            "status": "completed",
            "report_url": report_path,
            "data": structured_result
        }

    def _normalize_input(self, industry: str, focus_areas: List[str] = None) -> dict:
        """处理用户输入，确保数据格式一致"""

        # 同义词映射
        normalized_industry = self.industry_synonyms.get(industry, industry)

        # 清洗
        normalized_industry = normalized_industry.strip()

        # 校验
        if len(normalized_industry) < 2:
            raise ValueError("行业名称过短")
        if len(normalized_industry) > 50:
            raise ValueError("行业名称过长")

        return {
            "industry": normalized_industry,
            "focus_areas": focus_areas or [],
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _parse_task(self, input_data: dict) -> dict:
        """解析任务，确定分析维度和输出结构"""
        industry = input_data["industry"]

        # 根据行业类型确定分析框架
        framework_mapping = {
            "半导体": ["技术趋势", "产业链", "竞争格局", "风险挑战"],
            "新能源": ["政策环境", "市场空间", "技术路线", "主要玩家"],
            "default": ["行业概述", "发展趋势", "机会点", "风险因素"]
        }

        dimensions = framework_mapping.get(
            industry,
            framework_mapping["default"]
        )

        return {
            "dimensions": dimensions,
            "output_schema": self._build_output_schema(dimensions)
        }

    def _build_output_schema(self, dimensions: List[str]) -> dict:
        """构建结构化输出的 JSON Schema"""
        properties = {}
        required_fields = []

        for dim in dimensions:
            key = dim.lower().replace(" ", "_")
            properties[key] = {
                "type": "object",
                "description": f"{dim} 分析",
                "properties": {
                    "summary": {"type": "string"},
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "data_sources": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
            required_fields.append(key)

        return {
            "type": "object",
            "properties": properties,
            "required": required_fields
        }

    def _select_prompt_template(self, task_config: dict) -> str:
        """选择或构建 Prompt 模板"""
        dimensions = task_config["dimensions"]

        prompt = f"""你是一位资深的行业分析师。请对指定行业进行深入分析。

## 分析维度
{chr(10).join(f'{i+1}. {dim}' for i, dim in enumerate(dimensions))}

## 输出要求
- 每个维度需要包含：summary（概述）、key_points（3-5 个要点）、data_sources（数据来源建议）
- 分析要基于事实，避免空泛
- 数据来源要具体（如：财报、行业报告、政府数据）

待分析行业：{{industry}}

请严格按照 JSON Schema 格式输出。"""

        return prompt

    async def _call_llm_with_structure(
        self,
        prompt: str,
        industry: str,
        schema: dict
    ) -> dict:
        """使用 JSON Mode 确保结构化输出"""
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "你是专业的行业分析师，输出必须符合 JSON Schema。"
                    },
                    {
                        "role": "user",
                        "content": prompt.format(industry=industry)
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "industry_analysis",
                        "schema": schema
                    }
                },
                temperature=0.7
            )

            result_json = json.loads(
                response.choices[0].message.content
            )
            return result_json

        except Exception as e:
            raise AnalysisError(f"结构化输出失败: {str(e)}")

    def _generate_markdown_report(
        self,
        task_id: str,
        industry: str,
        data: dict
    ) -> str:
        """生成 Markdown 格式报告"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"{industry}_{task_id[:8]}.md"
        filepath = os.path.join(self.output_dir, filename)

        md_content = f"""# {industry} 行业分析报告

**生成时间**：{timestamp}
**任务 ID**：{task_id}

---

"""

        for key, value in data.items():
            if isinstance(value, dict) and "summary" in value:
                title = key.replace("_", " ").title()
                md_content += f"## {title}\n\n"
                md_content += f"{value['summary']}\n\n"

                if "key_points" in value:
                    md_content += "### 要点\n\n"
                    for point in value["key_points"]:
                        md_content += f"- {point}\n"
                    md_content += "\n"

                if "data_sources" in value:
                    md_content += "### 数据来源\n\n"
                    for source in value["data_sources"]:
                        md_content += f"- {source}\n"
                    md_content += "\n"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

        return filepath

    def _save_result(
        self,
        task_id: str,
        data: dict,
        report_path: str
    ):
        """保存结构化结果"""
        result = {
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
            "report_path": report_path
        }

        results_dir = os.path.join(self.output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        result_file = os.path.join(results_dir, f"{task_id}.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
```

### FastAPI 端点实现

```python
from fastapi import APIRouter, HTTPException
from app.services.industry_analysis_service import IndustryAnalysisService
from app.models.schemas import IndustryAnalysisRequest, IndustryAnalysisResponse

router = APIRouter()

analysis_service = IndustryAnalysisService(
    llm_client=llm_client,
    prompt_template_dir="prompts/industry",
    output_dir="outputs/industry_analysis"
)

@router.post("/analyze", response_model=IndustryAnalysisResponse)
async def analyze_industry(request: IndustryAnalysisRequest):
    """行业分析端点"""
    try:
        result = await analysis_service.analyze(
            industry=request.industry,
            focus_areas=request.focus_areas
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    # 实现查询逻辑
    pass
```

## 今日总结

行业分析 API Demo 是 Week 2 所有知识的综合应用。从输入到输出，每个环节都用到了本周学到的技能。

核心要点：

输入规范化是系统健壮性的基础。同义词映射、输入校验、数据清洗能避免很多后续问题。

任务解析把模糊的用户需求转化为具体的分析维度。可以用硬编码规则，也可以用模型动态生成。

Prompt 模板选择决定了输出质量。不同行业、不同深度应该使用不同的模板。

结构化输出让结果能被程序处理。JSON Mode + JSON Schema + Pydantic 确保输出格式可靠。

报告生成让结果对人类友好。Markdown 格式适合阅读，结构清晰。

结果保存节省成本、支持历史查询、便于质量追踪。

完整链路的每个环节都可能出错，需要有完善的错误处理。

## Week 2 综合练习

### 练习：行业分析 API 增强

为今天实现的 API 添加以下功能：

**缓存机制**。相同行业 24 小时内直接返回缓存结果。需要设计缓存 Key、缓存失效策略、缓存命中率统计。

**异步任务**。分析时间较长时，立即返回 task_id，后台执行。需要设计任务状态表、轮询接口、完成通知机制。

**报告格式选择**。支持输出 Markdown、PDF、JSON 三种格式。PDF 生成可以用 weasyprint 或 reportlab。

**批量分析**。一次请求支持多个行业分析。需要设计批量接口、并发控制、部分失败处理。

要求：使用本周所学的所有技术点、代码结构清晰易于维护、包含基本的错误处理和日志、提供 API 文档说明。
