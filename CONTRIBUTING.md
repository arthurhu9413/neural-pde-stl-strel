# Contributing

Thanks for working on the repository.

This codebase is now maintained with paper-quality reproducibility in mind. The
main rule is that **prose follows artifacts, not the other way around**.

---

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[torch,plot,dev]"
```

Add optional extras if you need them:

```bash
pip install -e ".[stl,strel,frameworks]"
```

---

## 2. First commands to know

```bash
python scripts/refresh_committed_summaries.py
python scripts/generate_all_figures.py --dpi 180
python scripts/generate_all_figures.py --dpi 180 --output-root /tmp/neural-pde-stl-strel-figcheck --check
python -m pytest -q
python scripts/check_env.py
```

These should be the default workflow before touching README tables or paper
figures.

---

## 3. Development workflow

1. make or refresh the result artifact you care about,
2. regenerate figures,
3. run tests,
4. update docs and prose,
5. inspect the diff for anything that sounds stronger than the data.

For quick checks:

```bash
make figures
make figures-check
make test
make test-standalone
make doctor
```

---

## 4. Rules for result files

### Do

- regenerate summary JSON/CSV files from scripts whenever possible,
- keep figure generation tied to committed artifacts,
- use scratch-space figure validation for CI and local checks,
- note clearly when a case study is monitoring-only,
- record whether a result is a main checkpoint or an ablation checkpoint.

### Do not

- hand-edit final metric tables,
- mix ablation numbers with main-comparison numbers without saying so,
- commit a plot whose source is unclear,
- leave stale numbers in README or docs after changing configs.

---

## 5. Rules for writing

The repository already got useful feedback on paper writing, so it is worth
recording the basics here.

### Good habits

- write the actual formulas used in the experiments,
- use American English consistently,
- verify every citation,
- explain what the method takes as input and what it outputs,
- state what a figure is supposed to show.

### Avoid

- vague tutorial-style filler in place of technical detail,
- overclaiming ("first", "best", "verified") unless the repo truly supports it,
- references that have not been checked directly,
- copy-pasted prose that introduces formatting artifacts.

---

## 6. Project structure

- `src/neural_pde_stl_strel/` : library code
- `scripts/` : training, evaluation, figure generation, environment checks
- `configs/` : YAML experiment configs
- `results/` : committed summaries, checkpoints, CSV logs
- `figs/` : generated paper-facing figures
- `assets/` : selected committed images and the heat rollout
- `docs/` : notes for specifications, reproducibility, claim discipline, references, and paper plan

---

## 7. Pull request checklist

Before opening a PR or sharing the repo, check:

- [ ] figures regenerate cleanly,
- [ ] tests pass,
- [ ] README tables match committed JSON/CSV artifacts,
- [ ] citations were verified,
- [ ] wording matches what the code actually implements,
- [ ] `make clean` was run before packaging a zip or handoff snapshot.

A small, accurate diff is better than a big diff with shaky claims.
