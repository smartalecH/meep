#!/usr/bin/env python3
"""Replay validation for a published multi-rank benchmark artifact."""

import argparse
import json
import pathlib

import benchmark_manifest as bm
import run_mpi_benchmark as runner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        result = bm.load_json_object(args.result, "MPI benchmark result")
        runner.authenticate_result_files(result)
        print(json.dumps({"valid": True, "result": str(args.result.resolve())}))
        return 0
    except (OSError, ValueError, bm.ValidationError, runner.RunnerError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
