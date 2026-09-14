---
layout: default
---

# 👋 你好，我是 {{ site.author.name }}

> {{ site.author.bio }}

（在这里写几段自我介绍：你是谁、在学什么/做什么、这个网站用来记录什么。支持标准 Markdown 语法。）

---

## 📝 最新文章

{% for post in site.posts limit:10 %}
- {{ post.date | date: "%Y-%m-%d" }} — [{{ post.title }}]({{ post.url | relative_url }})
{% endfor %}

---

## 🔗 快速链接

- [关于我]({{ '/about.html' | relative_url }})
- [GitHub](https://github.com/{{ site.author.github }})
- [邮箱](mailto:{{ site.author.email }})
