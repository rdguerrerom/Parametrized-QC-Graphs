# Overleaf bundle — submission-ready manuscript

This folder is the self-contained source tree for uploading the manuscript
to [Overleaf](https://overleaf.com) and then to arXiv. Everything compiles
with a single `pdflatex` pass; no BibTeX, no external dependencies beyond
standard TeXLive.

## Contents

```
Overleaf/
├── manuscript.tex              Quantum-template LaTeX source
├── figures/                    Four publication figures (PDF)
│   ├── fig1_framework.pdf
│   ├── fig2_pareto_frontier.pdf
│   ├── fig3_head_to_head.pdf
│   └── fig4_scaling.pdf
└── README.md                   this file
```

The manuscript uses `\documentclass[a4paper,onecolumn,noarxiv]{quantumarticle}`.
This class ships with TeXLive ≥ 2020 and is preinstalled on Overleaf, so
no `.cls` file needs to be bundled.

Bibliography is inline (`\begin{thebibliography}{99}`); no `.bib` file.

## Upload to Overleaf

1. Zip this folder: `zip -r Overleaf.zip Overleaf/`.
2. On Overleaf: **New Project → Upload Project** → select `Overleaf.zip`.
3. Compile with the default `pdfLaTeX` engine.

## Upload to arXiv from Overleaf

1. In the Overleaf project, **Menu → Submit → arXiv**.
2. Overleaf generates a minimal source tarball (excludes the compiled
   PDF and log files).
3. Primary category: `quant-ph`. Secondary: `cs.LG`.
4. License: **arXiv.org perpetual non-exclusive license** (default).
5. Comments field:
   ```
   8 pages, 4 figures. Submitted to Quantum.
   Code and data: https://github.com/rdguerrerom/Parametrized-QC-Graphs
   ```

## Local reproduction

If you prefer local compilation:

```bash
cd Overleaf
pdflatex manuscript.tex
pdflatex manuscript.tex   # twice for cross-references
```
