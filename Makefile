all: clean setup data

setup:
	python3 -m venv ../.DS212_PS2_CarPricePrediction_LinearRegression

## Make Dataset
data: install
	python3 src/data/prepare_data.py data

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt &&\
	pip install -e .

configure:
	make clean
	make setup
	make data
