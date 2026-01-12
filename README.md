# 量化投资技术文档

本仓库包含量化投资技术文档，使用 Quartz 4 部署。

## 📚 内容

- **特征工程模块**：系统讲解 Qlib 特征工程的核心概念与实践方法
- **LightGBM 模块**：深入学习 LightGBM 在量化投资中的应用
- **回测引擎模块**：策略回测与绩效评估
- **LSTM 深度学习**：时序数据的深度学习方法

## 🚀 快速开始

### 环境要求

- Node.js 18+
- npm 或 yarn

### 安装依赖

```bash
npm install
```

### 启动本地服务器

```bash
npm run dev
```

访问 http://localhost:8080 查看文档。

### 构建静态站点

```bash
npm run build
```

构建结果将输出到 `public` 目录。

## 📁 项目结构

```
.
├── content/               # 内容目录
│   ├── index.md          # 首页
│   ├── 快速开始.md        # 快速开始指南
│   ├── 站点导航.md        # 完整导航
│   ├── 关于.md          # 关于页面
│   ├── quant/           # 量化投资相关内容
│   │   └── qlib/       # Qlib 相关文档
│   │       ├── week1/  # 特征工程
│   │       ├── week2/  # LightGBM
│   │       ├── week3/  # 回测系统
│   │       └── week5/  # LSTM深度学习
│   ├── blockchain/     # 区块链相关内容
│   └── zlyq/           # 其他内容
├── quartz.config.ts     # Quartz 配置文件
├── package.json        # Node.js 依赖配置
└── README.md          # 本文件
```

## 🎯 特性

- 📱 响应式设计，支持移动端访问
- 🎨 优美的页面设计和排版
- 📐 支持数学公式渲染（KaTeX）
- 💻 代码高亮显示
- 🔍 完整的导航系统
- 🌓 支持深色模式
- ⚡ 快速构建和预览
- 🔗 完美兼容 Obsidian

## 📖 本地开发

### 使用 Obsidian 编辑

1. 在 Obsidian 中打开此仓库
2. 在 `content/` 目录下创建或编辑 Markdown 文件
3. 保存后，Quartz 会自动重新构建

### 文件命名规范

- 使用中文或英文文件名
- 建议使用描述性文件名，如 `01-基础概念.md`
- 文件扩展名必须为 `.md`

### Frontmatter 示例

```yaml
---
title: "页面标题"
description: "页面描述"
tags: [tag1, tag2]
draft: false
---
```

## 🛠️ 自定义配置

编辑 `quartz.config.ts` 文件以自定义：

- 网站标题和描述
- 主题颜色
- 字体设置
- 插件配置

## 📄 许可证

Creative Commons Attribution 4.0 International (CC BY 4.0)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 联系方式

- 📧 Email: [待添加]
- 💻 GitHub: [待添加]
- 💬 微信: [待添加]

## 🔗 相关资源

- [Quartz 官方文档](https://quartz.jzhao.xyz/)
- [Qlib 官方文档](https://qlib.readthedocs.io/)
- [LightGBM 文档](https://lightgbm.readthedocs.io/)
- [Obsidian 下载](https://obsidian.md/)
