.PHONY: install test lint train export run demo clean

install:
	pip install -e ".[dev,onnx]"

test:
	pytest --cov=damagescope --cov-report=term-missing

lint:
	ruff check damagescope tests

train:
	python -m damagescope.cli train

export:         ## Export the trained model to ONNX for CPU inference
	python -m damagescope.cli export

demo:
	python -m damagescope.cli demo

run:
	uvicorn damagescope.main:app --reload --port 8000

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov artifacts/* out
	touch artifacts/.gitkeep
