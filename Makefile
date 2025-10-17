.PHONY: setup ingest app query

setup:
	uv venv || python -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e .

ingest:
	. .venv/bin/activate && python scripts/ingest.py

app:
	. .venv/bin/activate && streamlit run app/streamlit_app.py

query:
	. .venv/bin/activate && python scripts/query.py "$(q)"
