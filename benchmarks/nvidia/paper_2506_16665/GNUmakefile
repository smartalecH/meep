.PHONY: check mpi-schema-check

check:
	/home/alechammond/meep-env/bin/python -m unittest discover -s . -p 'test_*.py'

mpi-schema-check:
	/home/alechammond/meep-env/bin/python run_mpi_benchmark.py --help >/dev/null
	/home/alechammond/meep-env/bin/python compare_mpi_benchmarks.py --help >/dev/null
	/home/alechammond/meep-env/bin/python run_transport_acceptance.py --help >/dev/null
	/home/alechammond/meep-env/bin/python validate_mpi_benchmark.py --help >/dev/null
