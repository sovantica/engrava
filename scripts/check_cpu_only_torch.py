"""Fail if the current environment holds anything but a CPU-only torch.

CI and every GPU-less dev environment install torch from the PyTorch CPU
index (``--no-deps torch --index-url https://download.pytorch.org/whl/cpu``)
*before* the project install, so the project install's dependency
resolution finds torch already satisfied and never reaches for the default
PyPI build, which on Linux x86_64 declares the NVIDIA CUDA libraries and
``triton`` as dependencies. This script is the proof that step worked: it
runs right after the project install, in every CI job that installs torch
(directly or through ``sentence-transformers``) and in
``verify_wheel_execution.py``'s ``vector`` lane, and fails closed if either
check below does not hold.

Two independent checks, both must pass:

* torch imports, and reports no CUDA build (``torch.version.cuda is None``
  on a CPU, macOS or Windows wheel; a CUDA build sets it to a version
  string such as ``"13.0"``).
* No installed distribution is CUDA-adjacent: its canonical name (PEP 503 --
  lower-cased, with every run of ``-``, ``_`` and ``.`` collapsed to a
  single ``-``) starts with ``nvidia-`` or ``cuda-``, or equals ``triton``.
  Names are read from ``importlib.metadata.distributions()``, never by
  parsing ``pip list`` output.

Every function below that touches the ambient environment (importing torch,
listing installed distributions) is a module-level, independently
patchable seam, so the unit tests exercise ``check_cpu_only_torch`` against
a stubbed torch module and a stubbed distribution list rather than this
process's own installed packages.
"""

from __future__ import annotations

import importlib.metadata
import re
import sys
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

#: PEP 503 canonicalisation: lower-case, with every run of ``-``, ``_`` and
#: ``.`` collapsed to one ``-``. Equivalent to
#: ``packaging.utils.canonicalize_name`` for a bare distribution name (no
#: extras bracket); implemented directly so this check carries no
#: dependency of its own.
_SEPARATOR_RUN = re.compile(r"[-_.]+")

#: Canonical-name prefixes/exact names that mark a distribution as part of
#: torch's CUDA stack (the NVIDIA wheels, the CUDA toolkit/bindings, and the
#: triton GPU kernel compiler).
_CUDA_ADJACENT_PREFIXES = ("nvidia-", "cuda-")
_CUDA_ADJACENT_EXACT_NAMES = frozenset({"triton"})


class _TorchVersionModule(Protocol):
    """The slice of ``torch.version`` this check reads."""

    cuda: str | None


class _TorchModule(Protocol):
    """The slice of the ``torch`` module this check reads."""

    version: _TorchVersionModule


def canonicalize_name(name: str) -> str:
    """Return the PEP 503 canonical form of a distribution name.

    Args:
        name: A distribution name as installed (for example
            ``"NVIDIA_cuda_runtime"`` or ``"Cuda.Bindings"``).

    Returns:
        The lower-cased name with every run of ``-``, ``_`` and ``.``
        collapsed to a single ``-`` (for example ``"nvidia-cuda-runtime"``).

    """
    return _SEPARATOR_RUN.sub("-", name).lower()


def is_cuda_adjacent(name: str) -> bool:
    """Return whether a distribution name's canonical form is CUDA-adjacent.

    Args:
        name: A distribution name as installed.

    Returns:
        ``True`` if the canonical name starts with ``nvidia-`` or
        ``cuda-``, or equals ``triton``.

    """
    canonical = canonicalize_name(name)
    return canonical.startswith(_CUDA_ADJACENT_PREFIXES) or canonical in _CUDA_ADJACENT_EXACT_NAMES


def _installed_distribution_names() -> list[str]:
    """Return the names of every distribution installed in this environment.

    Reads ``importlib.metadata.distributions()`` rather than parsing
    ``pip list`` output. A distribution with no readable name is skipped
    rather than raising.
    """
    names: list[str] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.name
        if name:
            names.append(name)
    return names


def find_cuda_adjacent_distributions(names: Iterable[str]) -> list[str]:
    """Return the names, in installed form, that are CUDA-adjacent.

    Args:
        names: Distribution names as installed.

    Returns:
        The subset of ``names`` whose canonical form is CUDA-adjacent (see
        ``is_cuda_adjacent``), sorted for a deterministic report.

    """
    return sorted(name for name in names if is_cuda_adjacent(name))


def _import_torch() -> _TorchModule:
    """Import and return the ``torch`` module.

    Raises:
        ImportError: torch is not installed.

    """
    import torch  # noqa: PLC0415 -- the whole point: an importable seam, not a top-level dependency

    return torch  # type: ignore[return-value]


def check_cpu_only_torch(
    *,
    import_torch: Callable[[], _TorchModule] = _import_torch,
    installed_distribution_names: Callable[[], Iterable[str]] = _installed_distribution_names,
) -> list[str]:
    """Return every problem found; an empty list means the environment is CPU-only.

    Args:
        import_torch: Returns the ``torch`` module, or raises
            ``ImportError``. Defaults to a real ``import torch``; tests
            substitute a stub to exercise the CUDA-build branch or the
            absent-torch branch without needing either installed.
        installed_distribution_names: Returns the installed distribution
            names. Defaults to reading the real environment; tests
            substitute a fixed list to exercise the CUDA-adjacent-package
            branch deterministically.

    Returns:
        Human-readable problem descriptions, empty if both checks pass.

    """
    problems: list[str] = []

    try:
        torch = import_torch()
    except ImportError as exc:
        problems.append(f"torch is not importable: {exc}")
    else:
        cuda_version = torch.version.cuda
        if cuda_version is not None:
            problems.append(
                f"torch is a CUDA build (torch.version.cuda = {cuda_version!r}), "
                "not the CPU-only build"
            )

    offenders = find_cuda_adjacent_distributions(installed_distribution_names())
    if offenders:
        problems.append("CUDA-adjacent distributions are installed: " + ", ".join(offenders))

    return problems


def main() -> int:
    """CLI entry point.

    Returns:
        ``0`` if the environment is CPU-only torch with no CUDA-adjacent
        distribution installed, ``1`` otherwise (the offending names, or
        the import failure, are printed to stderr first).

    """
    problems = check_cpu_only_torch()
    if problems:
        sys.stderr.write("check_cpu_only_torch: FAILED\n")
        for problem in problems:
            sys.stderr.write(f"  - {problem}\n")
        return 1
    sys.stdout.write("check_cpu_only_torch: OK -- torch is CPU-only, no CUDA-adjacent packages\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
