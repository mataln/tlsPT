conda env create -f environment.yml
conda activate tlspt
pip install -e .[dev]
pre-commit install
