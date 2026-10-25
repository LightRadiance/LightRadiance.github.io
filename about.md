---
layout: default
title: 关于
description: 关于 LightRadiance。
body_class: about
permalink: /about.html
---

<section class="wrap about">
  <div class="about-grid">
    <figure class="portrait">
      <img src="{{ '/assets/images/portrait.jpg' | relative_url }}" alt="LightRadiance 的照片" width="800" height="1000">
    </figure>

    <div class="about-copy">
      <p class="eyebrow">About</p>
      <h1>LightRadiance</h1>

      <p class="about-lead">
        本科就读于{{ site.author.education.bachelor }}，目前在{{ site.author.education.graduate }}读研究生，对强化学习（Reinforcement Learning）比较感兴趣。
      </p>

      <p class="about-note">
        这个网站主要用来记录我在研究探索过程中的阅读、实验与思考。如果你对相关话题感兴趣，欢迎和我交流。
      </p>

      <dl class="facts">
        <div>
          <dt>本科</dt>
          <dd>{{ site.author.education.bachelor }}</dd>
        </div>
        <div>
          <dt>研究生</dt>
          <dd>{{ site.author.education.graduate }}</dd>
        </div>
        <div>
          <dt>兴趣方向</dt>
          <dd>{{ site.author.interest }}</dd>
        </div>
      </dl>

      <div class="contact-block">
        <p class="eyebrow">联系</p>
        <ul class="contact-list">
          <li>
            <span class="contact-label">邮箱</span>
            <a href="mailto:{{ site.author.email }}">{{ site.author.email }}</a>
          </li>
          <li>
            <span class="contact-label">微信</span>
            <span>{{ site.author.wechat }}</span>
          </li>
        </ul>
      </div>
    </div>
  </div>
</section>
