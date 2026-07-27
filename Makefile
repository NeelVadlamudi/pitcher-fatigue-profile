.PHONY: install test notebooks app sample

install:
	python -m pip install -r requirements-dev.txt
	python -m pip install -e .

test:
	python -m pytest -q

notebooks:
	python scripts/build_notebooks.py
	python -m nbconvert --execute --to notebook --inplace \
		notebooks/01_data_quality_and_features.ipynb \
		notebooks/02_model_validation.ipynb \
		--ExecutePreprocessor.timeout=300

app:
	streamlit run app/streamlit_app.py

sample:
	python scripts/generate_sample.py

