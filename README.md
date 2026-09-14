# LightRadiance.github.io 使用说明

这是一个基于 **Jekyll + GitHub Pages**（主题：pages-themes/minimal）的个人网站。
你**不需要安装任何软件**，只要在 GitHub 上修改文件并提交，网站会在 1~2 分钟内自动更新。

网站地址：https://lightradiance.github.io

---

## 📂 文件结构（你只需要关心这几个）

```
_config.yml   ← 网站设置：标题、描述、个人信息
index.md      ← 首页（自我介绍 + 文章列表，文章列表自动生成）
about.md      ← 关于我 / 个人资料页
_posts/       ← 你的所有文章都放在这里
```

## ✏️ 第一步：填写个人信息

1. 打开 `_config.yml`，把 `title`、`description`、`author` 下的内容改成你自己的。
2. 打开 `index.md`，修改自我介绍部分。
3. 打开 `about.md`，把教育经历、技能、项目等占位内容替换成真实信息。

## 📝 第二步：发布文章

1. 在 `_posts/` 文件夹新建文件，命名格式必须是 `年-月-日-文章名.md`：
   例如 `2026-09-20-hello-world.md`（文件名建议用英文/拼音，不要用中文）
2. 文件开头必须有头信息：

   ```
   ---
   layout: default
   title: 你的文章标题
   date: 2026-09-20
   ---
   ```

3. 头信息下面用 Markdown 写正文（可以参考 `_posts/2026-09-14-my-first-post.md` 里的语法速查）。
4. 提交并推送到 GitHub，文章会自动出现在首页的"最新文章"列表里。

## 🗑 删除示例内容

上手之后，删除 `_posts/2026-09-14-my-first-post.md` 这篇示例文章即可。

## ❓ 常见问题

- **改了没生效？** 等 1~2 分钟，然后强制刷新浏览器（Ctrl+Shift+R）。可以在仓库的 Actions 页签查看部署状态。
- **想换主题？** GitHub Pages 官方支持的主题列表：https://pages.github.com/themes/ ，换掉 `_config.yml` 里的 `remote_theme` 那一行即可。
- **想在本地预览？** 需要安装 Ruby 和 Jekyll（可选，不是必须的）。
