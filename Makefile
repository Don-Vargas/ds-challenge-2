.PHONY: all test lint clean

all: lint unittest inttest

lint:
	pylint src tests
