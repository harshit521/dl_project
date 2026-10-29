.PHONY: setup weights demo eval app test clean

setup:
	python3.12 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

weights:
	.venv/bin/python scripts/download_weights.py --with-colorizer

demo:
	.venv/bin/python scripts/make_synthetic_old.py --input assets --out data/demo --all
	.venv/bin/python scripts/make_eval_set.py

eval:
	.venv/bin/python eval/evaluate.py

app:
	.venv/bin/streamlit run app/streamlit_app.py

test:
	.venv/bin/python -m pytest -q

clean:
	rm -rf outputs/eval outputs/*.png outputs/*.jpg .pytest_cache
