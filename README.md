# LightRadiance.github.io

个人网站，主要用来分享强化学习方向的研究与探索。

网站地址：<https://lightradiance.github.io>

网站基于 **Jekyll + GitHub Pages**，不需要安装任何软件。修改文件并推送到 GitHub 后，1～2 分钟内会自动更新。

## 页面结构

```text
_config.yml          网站标题、描述、个人信息
index.html           首页：自动生成的文章列表
about.md             关于页：照片、教育经历、兴趣、联系方式
_layouts/            页面模板
_posts/              所有文章
assets/css/main.css  样式（一般不需要改）
assets/images/       照片和头像
```

## 发布一篇新文章

1. 在 `_posts/` 目录中新建一个 Markdown 文件，文件名格式为：

   ```text
   YYYY-MM-DD-英文标题.md
   ```

   例如：`2026-11-02-policy-gradient-notes.md`。文件名建议使用英文或拼音，不要使用中文。

2. 在文件最上方写入头信息，最少只需要这三项：

   ```yaml
   ---
   layout: post
   title: 文章标题
   date: 2026-11-02 10:00:00 +0800
   ---
   ```

   还可以加上摘要和标签（都可以省略）：

   ```yaml
   description: 一句话摘要，会显示在首页列表和文章顶部
   tags: [强化学习]
   ```

   如果没有写 `description`，首页会自动截取正文第一段作为摘要。

3. 在头信息下面照常用 Markdown 写正文。

4. 提交并推送到 GitHub。文章会自动按时间倒序出现在首页，不需要手动修改列表。

## 修改个人信息

- **邮箱、微信、学校、兴趣方向**：编辑 `_config.yml` 中的 `author` 部分。
- **关于页文案**：编辑 `about.md`。
- **关于页照片**：替换 `assets/images/portrait.jpg`（建议 4:5 竖图，800 × 1000）。
- **顶部小头像**：替换 `assets/images/avatar.jpg`（建议正方形，320 × 320）。
- **网站标题和描述**：编辑 `_config.yml` 中的 `title` 和 `description`。

## 本地预览（可选）

如果电脑上安装了 Ruby 和 Jekyll：

```bash
gem install jekyll
jekyll serve
```

然后在浏览器打开 `http://localhost:4000`。

## 常见问题

- **改了没生效？** 等待 1～2 分钟，然后强制刷新浏览器（Ctrl + Shift + R）。可以在仓库的 Actions 页签查看部署状态。
- **文章日期不对？** 检查文件名中的日期和头信息里的 `date` 是否一致，并保留 `+0800` 时区。
- **想删掉示例文章？** 删除 `_posts/2026-10-25-start-here.md` 即可。
