"""Tests for the CPU-only-torch check (scripts/check_cpu_only_torch.py).

The real torch build and the real set of installed distributions are both
ambient facts about the environment the suite happens to run in -- on this
project's own dev venv, torch is in fact a CUDA build, which is exactly the
state this check must fail on. So every test below drives
``check_cpu_only_torch`` (and ``main``) through its two injectable seams
(``import_torch``, ``installed_distribution_names``) with controlled stubs,
never the ambient environment, keeping each failure branch independently
reachable and deterministic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_cpu_only_torch.py"


@pytest.fixture
def check_module() -> object:
    """Load ``scripts/check_cpu_only_torch.py`` as a module for direct testing."""
    spec = importlib.util.spec_from_file_location("check_cpu_only_torch", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        msg = f"could not load check_cpu_only_torch module from {SCRIPT_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_torch(cuda: str | None) -> SimpleNamespace:
    """Build a stub object shaped like the ``torch`` module's ``version.cuda``."""
    return SimpleNamespace(version=SimpleNamespace(cuda=cuda))


class TestCanonicalizeName:
    def test_lowercases(self, check_module: object) -> None:
        assert check_module.canonicalize_name("Triton") == "triton"  # type: ignore[attr-defined]

    def test_collapses_separator_runs(self, check_module: object) -> None:
        # Mixed underscores and dots, including a run of more than one
        # separator, all collapse to a single '-'.
        assert check_module.canonicalize_name("NVIDIA_cuda_runtime") == "nvidia-cuda-runtime"  # type: ignore[attr-defined]
        assert check_module.canonicalize_name("Cuda.Bindings") == "cuda-bindings"  # type: ignore[attr-defined]
        assert check_module.canonicalize_name("nvidia--cu13") == "nvidia-cu13"  # type: ignore[attr-defined]
        assert check_module.canonicalize_name("nvidia.cu_13") == "nvidia-cu-13"  # type: ignore[attr-defined]


class TestIsCudaAdjacent:
    @pytest.mark.parametrize(
        "name",
        [
            "nvidia-cudnn-cu13",
            "NVIDIA_cuda_runtime",
            "Cuda.Bindings",
            "cuda-toolkit",
            "triton",
            "Triton",
        ],
    )
    def test_flags_cuda_adjacent_names(self, check_module: object, name: str) -> None:
        assert check_module.is_cuda_adjacent(name) is True  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "name",
        [
            "numpy",
            "torch",
            "pytorch-lightning",
            # Must not match on substring containment: this name contains
            # "cuda" but does not start with "cuda-" or "nvidia-".
            "pycuda",
            "scikit-learn",
        ],
    )
    def test_does_not_flag_unrelated_names(self, check_module: object, name: str) -> None:
        assert check_module.is_cuda_adjacent(name) is False  # type: ignore[attr-defined]


class TestFindCudaAdjacentDistributions:
    def test_filters_and_sorts(self, check_module: object) -> None:
        names = ["numpy", "nvidia-nccl-cu13", "click", "Cuda.Bindings", "triton"]
        assert check_module.find_cuda_adjacent_distributions(names) == [  # type: ignore[attr-defined]
            "Cuda.Bindings",
            "nvidia-nccl-cu13",
            "triton",
        ]

    def test_empty_when_nothing_matches(self, check_module: object) -> None:
        assert check_module.find_cuda_adjacent_distributions(["numpy", "click"]) == []  # type: ignore[attr-defined]


class TestCheckCpuOnlyTorch:
    def test_passes_on_cpu_torch_with_no_cuda_distributions(self, check_module: object) -> None:
        problems = check_module.check_cpu_only_torch(  # type: ignore[attr-defined]
            import_torch=lambda: _fake_torch(cuda=None),
            installed_distribution_names=lambda: ["numpy", "click", "torch"],
        )
        assert problems == []

    def test_fails_when_torch_is_a_cuda_build(self, check_module: object) -> None:
        problems = check_module.check_cpu_only_torch(  # type: ignore[attr-defined]
            import_torch=lambda: _fake_torch(cuda="13.0"),
            installed_distribution_names=lambda: ["numpy", "click", "torch"],
        )
        assert len(problems) == 1
        assert "CUDA build" in problems[0]
        assert "13.0" in problems[0]

    def test_fails_when_torch_is_absent(self, check_module: object) -> None:
        def _raise_import_error() -> object:
            message = "No module named 'torch'"
            raise ImportError(message)

        problems = check_module.check_cpu_only_torch(  # type: ignore[attr-defined]
            import_torch=_raise_import_error,
            installed_distribution_names=lambda: ["numpy", "click"],
        )
        assert len(problems) == 1
        assert "not importable" in problems[0]

    def test_fails_when_a_cuda_adjacent_distribution_is_installed(
        self, check_module: object
    ) -> None:
        problems = check_module.check_cpu_only_torch(  # type: ignore[attr-defined]
            import_torch=lambda: _fake_torch(cuda=None),
            installed_distribution_names=lambda: ["numpy", "nvidia-cudnn-cu13", "click"],
        )
        assert len(problems) == 1
        assert "nvidia-cudnn-cu13" in problems[0]

    def test_canonicalisation_catches_a_mixed_case_distribution_name(
        self, check_module: object
    ) -> None:
        # The installed name is not in canonical form; the check must still
        # catch it (PEP 503 canonicalisation, not a literal string match).
        problems = check_module.check_cpu_only_torch(  # type: ignore[attr-defined]
            import_torch=lambda: _fake_torch(cuda=None),
            installed_distribution_names=lambda: ["numpy", "NVIDIA_cuda_runtime"],
        )
        assert len(problems) == 1
        assert "NVIDIA_cuda_runtime" in problems[0]

    def test_triton_is_flagged_by_exact_canonical_match(self, check_module: object) -> None:
        problems = check_module.check_cpu_only_torch(  # type: ignore[attr-defined]
            import_torch=lambda: _fake_torch(cuda=None),
            installed_distribution_names=lambda: ["numpy", "Triton"],
        )
        assert len(problems) == 1
        assert "Triton" in problems[0]

    def test_both_branches_fail_independently_and_both_are_reported(
        self, check_module: object
    ) -> None:
        problems = check_module.check_cpu_only_torch(  # type: ignore[attr-defined]
            import_torch=lambda: _fake_torch(cuda="13.0"),
            installed_distribution_names=lambda: ["nvidia-nccl-cu13"],
        )
        assert len(problems) == 2


class TestMain:
    def test_returns_zero_and_prints_ok_when_clean(
        self,
        check_module: object,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(check_module, "check_cpu_only_torch", list)  # type: ignore[attr-defined]
        assert check_module.main() == 0  # type: ignore[attr-defined]
        assert "OK" in capsys.readouterr().out

    def test_returns_one_and_prints_offending_names_when_dirty(
        self,
        check_module: object,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            check_module,
            "check_cpu_only_torch",
            lambda: ["CUDA-adjacent distributions are installed: nvidia-cudnn-cu13"],
        )
        assert check_module.main() == 1  # type: ignore[attr-defined]
        err = capsys.readouterr().err
        assert "FAILED" in err
        assert "nvidia-cudnn-cu13" in err
