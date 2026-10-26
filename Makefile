.PHONY: check demo validate
check:
	python -m unittest discover -s tests -v
	ruff check src tests docker scripts
	ruff format --check src tests docker scripts

demo:
	rl-zero demo

validate:
	rl-zero validate
