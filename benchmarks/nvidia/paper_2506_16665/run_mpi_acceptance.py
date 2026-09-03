#!/usr/bin/env python3
"""Run one CUDA/MPI acceptance binary with an explicit provider configuration."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
import tempfile
from typing import Dict, Mapping, Sequence


UCX_TLS = "self,sm,cuda_copy,cuda_ipc"


class AcceptanceError(RuntimeError):
    pass


def provider_environment(route: str, base: Mapping[str, str]) -> Dict[str, str]:
    """Return the complete pre-MPI environment for one explicitly selected route."""
    if route not in {"ucx", "ob1"}:
        raise AcceptanceError("provider route must be ucx or ob1")
    environment = dict(base)
    environment["OMPI_MCA_opal_cuda_support"] = "true"
    if route == "ucx":
        environment["OMPI_MCA_pml"] = "ucx"
        environment["UCX_TLS"] = UCX_TLS
        environment.pop("OMPI_MCA_btl", None)
    else:
        environment["OMPI_MCA_pml"] = "ob1"
        environment["OMPI_MCA_btl"] = "self,smcuda,tcp"
        environment.pop("UCX_TLS", None)
    return environment


def command_output(command: Sequence[str], environment: Mapping[str, str]) -> str:
    try:
        return subprocess.run(
            list(command),
            env=dict(environment),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def tool_sibling(mpiexec: pathlib.Path, name: str) -> str:
    candidate = mpiexec.resolve().parent / name
    return str(candidate) if candidate.exists() else name


def atomic_write_json(path: pathlib.Path, value: Mapping[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = pathlib.Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpiexec", type=pathlib.Path, required=True)
    parser.add_argument("--ranks", type=int, choices=(2, 4), required=True)
    parser.add_argument("--provider-route", choices=("ucx", "ob1"), default="ucx")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise AcceptanceError("an acceptance command is required after --")

    environment = provider_environment(args.provider_route, os.environ)
    invocation = [str(args.mpiexec), "-n", str(args.ranks), *command]
    completed = subprocess.run(
        invocation,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    provenance = {
        "mpi": command_output([str(args.mpiexec), "--version"], environment),
        "ompi": command_output(
            [tool_sibling(args.mpiexec, "ompi_info"), "--version"], environment
        ),
        "ucx": command_output([tool_sibling(args.mpiexec, "ucx_info"), "-v"], environment),
        "cuda_toolkit": command_output(["nvcc", "--version"], environment),
        "cuda_devices": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            environment,
        ),
    }
    result = {
        "schema_version": 1,
        "kind": "meep_cuda_mpi_provider_acceptance",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "provider_route": args.provider_route,
        "ranks": args.ranks,
        "environment": {
            name: environment[name]
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "OMPI_MCA_opal_cuda_support",
                "OMPI_MCA_pml",
                "OMPI_MCA_btl",
                "UCX_TLS",
                "MEEP_GPU_AWARE_MPI",
                "MEEP_NVIDIA_MPI_OVERLAP",
                "MEEP_NVIDIA_GRAPH_MODE",
            )
            if name in environment
        },
        "command": invocation,
        "exit_code": completed.returncode,
        "output": completed.stdout,
        "provenance": provenance,
    }
    atomic_write_json(args.output, result)
    print(completed.stdout, end="")
    print(f"acceptance artifact: {args.output.resolve()}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
