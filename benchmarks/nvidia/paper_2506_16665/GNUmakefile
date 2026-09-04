.PHONY: check mpi-schema-check

check:
	$(PYTHON) -m unittest discover -s . -p 'test_*.py'

mpi-schema-check:
	$(PYTHON) run_mpi_benchmark.py --help >/dev/null
	$(PYTHON) run_self_contained_benchmark.py --help >/dev/null
	$(PYTHON) compare_mpi_benchmarks.py --help >/dev/null
	$(PYTHON) run_transport_acceptance.py --help >/dev/null
	$(PYTHON) validate_mpi_benchmark.py --help >/dev/null
