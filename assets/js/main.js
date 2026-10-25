// 自动根据文章中的 h2 / h3 生成右侧目录。
(function () {
  const content = document.querySelector(".prose");
  const toc = document.getElementById("toc");
  const tocNav = document.getElementById("toc-nav");
  if (!content || !toc || !tocNav) return;

  const headings = Array.from(content.querySelectorAll("h2, h3"));
  if (headings.length < 2) return;

  const links = headings.map(function (heading, index) {
    if (!heading.id) heading.id = "section-" + (index + 1);

    const link = document.createElement("a");
    link.className = "toc-link " + (heading.tagName === "H3" ? "toc-h3" : "toc-h2");
    link.href = "#" + heading.id;
    link.textContent = heading.textContent;
    tocNav.appendChild(link);
    return link;
  });

  toc.hidden = false;

  function setActiveSection() {
    const readingLine = window.scrollY + 140;
    const atBottom =
      window.scrollY > 0 &&
      window.innerHeight + window.scrollY >=
        document.documentElement.scrollHeight - 4;
    let activeIndex = -1;

    if (atBottom) {
      activeIndex = headings.length - 1;
    } else {
      headings.forEach(function (heading, index) {
        if (heading.getBoundingClientRect().top + window.scrollY <= readingLine) {
          activeIndex = index;
        }
      });
    }

    links.forEach(function (link, index) {
      link.classList.toggle("is-active", index === activeIndex);
    });
  }

  let ticking = false;
  window.addEventListener(
    "scroll",
    function () {
      if (!ticking) {
        window.requestAnimationFrame(function () {
          setActiveSection();
          ticking = false;
        });
        ticking = true;
      }
    },
    { passive: true }
  );

  setActiveSection();
})();
