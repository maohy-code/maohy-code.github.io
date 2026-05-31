# 🍁 maohy-code.github.io

> (=^･^=)

个人 GitHub Pages 站点，用于存放各类 Web 小项目和学习笔记。

## 站点结构

```
/
├── index.md                ← 首页（项目导航）
├── notes/                  ← 学习笔记
│   └── phonopy/            ← Phonopy 声子计算笔记
├── tools/                  ← 在线小工具
│   └── seat-arranger/      ← 座位排布工具
└── _layouts/               ← Jekyll 页面布局
```

## 使用方式

本站使用 Jekyll + GitHub Actions 自动构建部署。
向 `main` 分支推送代码即可自动发布到 `https://maohy-code.github.io/`。

### 添加新项目

在根目录下创建子文件夹，放入 `index.md` 或 `index.html` 即可：

```
mkdir my-project
echo "# 新项目" > my-project/index.md
git add .
git commit -m "add: my-project"
git push
```

访问 `https://maohy-code.github.io/my-project/` 即可看到效果。

## 技术栈

- [Jekyll](https://jekyllrb.com/) — 静态站点生成
- [Slate 主题](https://github.com/pages-themes/slate) — 页面样式
- [MathJax](https://www.mathjax.org/) — 数学公式渲染
- [GitHub Pages](https://pages.github.com/) — 托管与自动部署
