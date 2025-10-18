.PHONY: setup ingest app query

setup:
	uv venv || python -m venv .venv
	. .venv/bin/activate && PYTHONPATH=. pip install -U pip && pip install -e .

ingest:
	. .venv/bin/activate && PYTHONPATH=. python scripts/ingest.py

app:
	. .venv/bin/activate && PYTHONPATH=. streamlit run app/streamlit_app.py

query:
	. .venv/bin/activate && PYTHONPATH=. python scripts/query.py "$(q)"
