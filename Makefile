install:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest

lint:
	ruff check .
	black --check .

format:
	black .
	ruff check --fix .