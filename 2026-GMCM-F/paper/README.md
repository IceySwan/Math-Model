# 论文

`main.tex` 是当前论文入口，`main.pdf` 由此目录中的源码重新编译生成。章节、表格、图件、模板和参考文献按当前有效引用完整保留。

## 编译

安装完整 TeX Live 或 MiKTeX，在本目录执行：

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Windows 模板使用 Times New Roman、Arial、Courier New、SimSun、SimHei 和 KaiTi。模板在 Linux 上使用 TeX 发行版提供的 Fandol 与 TeX Gyre 字体。

`figures/` 保存论文所需 PDF 图件，`tables/` 保存论文引用的 LaTeX 表格与程序片段。它们构成论文的编译依赖；编译直接读取这些文件。重新生成图表需要在本地准备算法输入与运行结果，详见 `../code/README.md`。

发布副本补齐了原模板封面条件语句的结束标记。论文正文和数值内容沿用当前稿。
