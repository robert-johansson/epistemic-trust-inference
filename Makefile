# Reproduce the paper. See README.md.
PY = uv run --frozen python
MEMO_MCP_DIR ?= ../memo-mcp
MEMO_MCP_PYTHON ?= $(MEMO_MCP_DIR)/venv/bin/python
FIGS = fig1_framework fig2_game fig3_coupling fig4_intent fig5_form fig6_bart

.PHONY: setup validate check numbers figures supplement idioms

setup:
	uv sync --frozen

validate:
	$(PY) tools/reproduce.py --check --only validate

check:
	$(PY) tools/reproduce.py --check

numbers:
	$(PY) tools/reproduce.py --write

figures:
	cd figures && mkdir -p build pdf && for f in $(FIGS); do \
	  xelatex -interaction=nonstopmode -halt-on-error -output-directory=build $$f.tex > build/$$f.out && cp build/$$f.pdf pdf/; \
	done

supplement:
	uv run --frozen python -I tools/make_supplement.py

idioms:
	MEMO_MCP_DIR=$(MEMO_MCP_DIR) $(MEMO_MCP_PYTHON) tools/check_memo_idioms.py
