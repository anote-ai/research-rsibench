# RSI-Bench paper

`rsibench.tex` is a draft of *RSI-Bench: Measuring Recursive Self-Improvement in AI Systems*.

Build with `latexmk -pdf rsibench.tex`. You can also upload `rsibench.tex` and `references.bib` to Overleaf. The paper needs `pgfplots` and `natbib`, both included in TeX Live.

Every number in the tables and figure comes from `python scripts/run_demo.py` (see `../results/sim_results.md`). The results section is harness validation with a deterministic reference learner. Frontier-model results are pre-registered in §8 and will be added once the sweep has run.
