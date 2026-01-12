# Deploy Guide

## GitHub Pages 部署配置

完成以下步骤后，网站将自动部署到：https://mindcarver.github.io

### 1. 配置 Actions 权限

访问：https://github.com/mindcarver/mindcarver.github.io/settings/actions

找到 **Workflow permissions**，选择：**Read and write permissions**，然后 Save。

### 2. 配置 GitHub Pages

访问：https://github.com/mindcarver/mindcarver.github.io/settings/pages

- Source: Deploy from a branch
- Branch: gh-pages / (root)
- Save

### 3. 完成！

每次推送代码到 main 分支，GitHub Actions 会自动构建并部署网站。

