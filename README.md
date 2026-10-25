# LightRadiance.github.io

个人网站：分享强化学习方向的研究与探索。个人介绍和文章列表在首页，文章详情单独成页。

网站：<https://lightradiance.github.io>（推送到 GitHub 后 1～2 分钟自动更新）

## 发布文章

在 `_posts/` 中新建 `YYYY-MM-DD-英文标题.md`（例如 `2026-11-02-policy-gradient-notes.md`），开头写：

```yaml
---
layout: post
title: 文章标题
date: 2026-11-02 10:00:00 +0800
description: 一句话摘要（可选）
tags: [强化学习]  # 可选
---
```

下面直接写 Markdown 正文。文章会自动按时间倒序出现在首页；未填 `description` 时，首页会截取正文第一段。

## 修改网站

- 个人信息（邮箱、微信、教育经历）：`_config.yml` 的 `author`
- 个人介绍文案：`index.html` 中 `id="about"` 的部分
- 首页照片：替换 `assets/images/portrait.jpg`（建议 4:5，800 × 1000）
- 网站标题与描述：`_config.yml` 的 `title`、`description`

## 本地预览（可选）

```bash
gem install jekyll
jekyll serve
```

然后打开 `http://localhost:4000`。

## 常见问题

改了没生效？等 1～2 分钟，再强制刷新（Ctrl + Shift + R）；部署状态见仓库 Actions。
