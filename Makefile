# neural-pde-stl-strel - build targets
.PHONY: help install install-cpu-linux install-extra test test-standalone demo quickstart diffusion1d diffusion1d-demo \
        rtamt-eval ablations heat2d-figs heat2d-stl-safe heat2d-stl-eventually moonlight-eval benchmark doctor lint clean figures figures-check refresh-check status check

PYTHON ?= python3
PIP    ?= pip
FIGURE_CHECK_DIR ?= .figure-check

# ─── Help ─────────────────────────────────────────────────────────────────────

help:
	@echo "neural-pde-stl-strel - available targets"
	@echo ""
	@echo "  Setup:"
	@echo "    make install          Install with PyTorch, plotting, and dev extras"
	@echo "    make install-cpu-linux Install Linux CPU-only PyTorch + plotting + dev extras"
	@echo "    make install-extra    Install all extras declared in pyproject.toml"
	@echo "    make quickstart       Install and print next steps"
	@echo ""
	@echo "  Experiments:"
	@echo "    make diffusion1d-demo Short 1D diffusion demo (200 epochs)"
	@echo "    make diffusion1d      Full 1D diffusion baseline + STL runs"
	@echo "    make rtamt-eval       Post-hoc STL monitoring with RTAMT"
	@echo "    make ablations        Lambda sweep (6 weights) + ablation plots"
	@echo "    make heat2d-figs      Generate 2D heat field snapshots"
	@echo "    make moonlight-eval   STREL evaluation (requires Java 21+ and MoonLight)"
	@echo ""
	@echo "  Figures & Analysis:"
	@echo "    make figures          Generate all publication-quality figures"
	@echo "    make figures-check    Generate figures in scratch space and validate outputs"
	@echo "    make benchmark        Show the committed runtime/cost snapshot"
	@echo "    make status           Print the runnable example inventory"
	@echo ""
	@echo "  Quality:"
	@echo "    make test             Run full pytest suite"
	@echo "    make test-standalone  Run standalone tests (no PyTorch needed)"
	@echo "    make lint             Run ruff linter"
	@echo "    make doctor           Check environment and dependencies"
	@echo ""
	@echo "  Pipeline:"
	@echo "    make demo             Full end-to-end pipeline"
	@echo "    make clean            Remove caches and build artifacts"

# ─── Installation ─────────────────────────────────────────────────────────────

install:
	$(PIP) install -e ".[torch,plot,dev]"

install-cpu-linux:
	$(PIP) install --index-url https://download.pytorch.org/whl/cpu torch
	$(PIP) install -e ".[plot,dev]"

install-extra:
	$(PIP) install -e ".[all]"

quickstart: install
	@echo "Installed. Try: make diffusion1d-demo"

# ─── Testing ──────────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-standalone:
	PYTHONPATH=src $(PYTHON) tests/run_tests.py

doctor:
	$(PYTHON) scripts/check_env.py

# ─── 1D Diffusion ────────────────────────────────────────────────────────────

diffusion1d-demo:
	$(PYTHON) scripts/run_experiment.py -c configs/diffusion1d_baseline.yaml \
	    --set optim.epochs=200 --alias auto
	$(PYTHON) scripts/run_experiment.py -c configs/diffusion1d_stl.yaml \
	    --set optim.epochs=200 --alias auto

diffusion1d:
	$(PYTHON) scripts/run_experiment.py -c configs/diffusion1d_baseline.yaml --alias auto
	$(PYTHON) scripts/run_experiment.py -c configs/diffusion1d_stl.yaml --alias auto

rtamt-eval:
	$(PYTHON) scripts/eval_diffusion_rtamt.py

ablations:
	$(PYTHON) scripts/run_ablations_diffusion.py
	$(PYTHON) scripts/plot_ablations.py

# ─── 2D Heat ─────────────────────────────────────────────────────────────────

heat2d-figs:
	$(PYTHON) scripts/gen_heat2d_frames.py

moonlight-eval:
	$(PYTHON) scripts/eval_heat2d_moonlight.py

heat2d-stl-safe:
	$(PYTHON) scripts/run_experiment.py -c configs/heat2d_stl_safe.yaml

heat2d-stl-eventually:
	$(PYTHON) scripts/run_experiment.py -c configs/heat2d_stl_eventually.yaml

# ─── Figures ─────────────────────────────────────────────────────────────────

figures:
	$(PYTHON) scripts/generate_all_figures.py --dpi 200

figures-check:
	rm -rf $(FIGURE_CHECK_DIR)
	$(PYTHON) scripts/generate_all_figures.py --dpi 200 \
	    --output-root $(FIGURE_CHECK_DIR) \
	    --manifest $(FIGURE_CHECK_DIR)/figure_manifest.json \
	    --check

refresh-check:
	$(PYTHON) scripts/refresh_committed_summaries.py --check

status:
	$(PYTHON) scripts/example_status.py

# ─── Benchmarking ────────────────────────────────────────────────────────────

benchmark:
	$(PYTHON) scripts/show_benchmark_snapshot.py

# ─── Full Demo Pipeline ──────────────────────────────────────────────────────

demo: diffusion1d rtamt-eval heat2d-figs figures
	@echo "Demo complete. See results/ and figs/."

# ─── Cleanup ─────────────────────────────────────────────────────────────────

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/

check: lint test-standalone
	@echo "All checks passed."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .cache .mplconfig .pycache .pytest_cache .ruff_cache .figure-check logs runs .tmp data build dist *.egg-info src/*.egg-info
