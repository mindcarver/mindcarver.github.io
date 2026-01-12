# 域名配置说明 - mindcarver.top

## 当前状态

✅ **DNS 已正确配置：**
```
mindcarver.top → mindcarver.github.io
```

❌ **GitHub Pages 错误：**
```
The custom domain `mindcarver.top` is already taken
```

---

## 🔍 问题分析

域名 `mindcarver.top` 已经被其他 GitHub Pages 仓库使用了。需要通过 **DNS TXT 记录验证**域名所有权。

---

## 🔧 解决步骤

### 第 1 步：获取 GitHub 验证值

1. 访问：https://github.com/mindcarver/mindcarver.github.io/settings/pages
2. 在 "Custom domain" 部分
3. 查看显示的 **DNS TXT 记录验证信息**
4. 复制验证值（格式类似：`xxx-xxx-xxx`）

### 第 2 步：添加 DNS TXT 记录

在你的域名管理后台（阿里云、腾讯云等）添加：

```
主机记录：_github-pages-challenge-mindcarver
记录类型：TXT
记录值：<从 GitHub 复制的验证值>
TTL：600
```

### 第 3 步：保存并等待

1. 保存 DNS 配置
2. 等待 10-30 分钟让 DNS 生效
3. 返回 GitHub Pages 设置页面
4. 点击 **Save** 按钮

### 第 4 步：验证完成

GitHub Pages 页面会显示：
```
✓ Your site is live at mindcarver.top
```

---

## 📋 DNS 配置示例

### 如果是阿里云

1. 登录：https://dc.console.aliyun.com/next/index
2. 找到：域名解析
3. 选择：mindcarver.top
4. 点击：添加记录
5. 填写：
   - **记录类型**：TXT
   - **主机记录**：`_github-pages-challenge-mindcarver`
   - **记录值**：`(从 GitHub 复制的值)`
   - **TTL**：10 分钟或默认

### 如果是腾讯云

1. 登录：https://console.cloud.tencent.com/cns
2. 找到：域名管理
3. 选择：mindcarver.top
4. 点击：添加解析
5. 填写：
   - **主机记录**：`_github-pages-challenge-mindcarver`
   - **记录类型**：TXT
   - **记录值**：`(从 GitHub 复制的值)`
   - **TTL**：600

### 如果是 Cloudflare

1. 登录：https://dash.cloudflare.com/
2. 选择：mindcarver.top
3. 点击：Add record
4. 填写：
   - **Type**：TXT
   - **Name**：`_github-pages-challenge-mindcarver`
   - **Content**：`(从 GitHub 复制的值)`
   - **Proxy status**：DNS only

---

## 🕐 预计时间线

| 操作 | 预计时间 |
|------|---------|
| 添加 TXT 记录 | 5 分钟 |
| DNS 生效 | 10-30 分钟 |
| GitHub 验证 | 1-2 分钟 |
| HTTPS 证书 | 10-30 分钟 |
| 域名全局生效 | 10 分钟 - 24 小时 |

**总计：约 30-60 分钟**

---

## ✅ 验证清单

配置完成后，检查：

- [ ] DNS TXT 记录已添加
- [ ] 在 GitHub Pages 显示：✓ DNS check passed
- [ ] 可以访问：https://mindcarver.top
- [ ] 浏览器地址栏显示：https://mindcarver.top
- [ ] 显示 HTTPS 锁图标

---

## 🚨 常见问题

### Q: 找不到 "_github-pages-challenge-mindcarver" 验证值？
**A:**
1. 清除浏览器缓存
2. 访问：https://github.com/mindcarver/mindcarver.github.io/settings/pages
3. 查看页面的错误提示或验证信息
4. 如果仍然看不到，联系 GitHub Support

### Q: 添加 TXT 记录后还是显示 "already taken"？
**A:**
1. 等待更长时间（最长 24 小时）
2. 检查 TXT 记录是否正确（nslookup）
3. 尝试删除 TXT 记录后重新添加
4. 联系 GitHub Support：https://support.github.com/contact

### Q: 可以使用 GitHub Pages 默认域名吗？
**A:**
可以！https://mindcarver.github.io 一直可以正常访问。

### Q: 需要自定义域名吗？
**A:**
推荐使用自定义域名（mindcarver.top），因为：
- 更专业
- 更利于 SEO
- 更容易记忆

---

## 📞 需要帮助？

如果遇到问题，请提供：
1. GitHub Pages 页面显示的完整错误信息
2. DNS 管理商（阿里云/腾讯云/Cloudflare等）
3. 已添加的 DNS 记录截图

---

## 🎯 备选方案

如果 TXT 验证方式不成功，可以：

### 临时方案 1
暂时使用 GitHub Pages 默认域名：https://mindcarver.github.io

### 临时方案 2
尝试注册新域名（如：mindcarver-docs.top）并配置

### 联系 GitHub
提交工单请求释放域名：https://support.github.com

---

**重要：请先访问 GitHub Pages 设置页面，查看是否有显示 TXT 验证值！**

获取验证值后，按照上述步骤添加 DNS TXT 记录即可。
