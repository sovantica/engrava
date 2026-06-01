"""Synthetic benchmark suite — scenarios, generator, evaluator, runner.

The package exposes four public sub-modules:

* :mod:`engrava.benchmarks.synthetic.scenarios` — pre-registered
  scenario library (anti-cherry-pick neutrals included).
* :mod:`engrava.benchmarks.synthetic.generate` — deterministic
  conversation + question generator on top of the scenario library.
* :mod:`engrava.benchmarks.synthetic.evaluate` — OFF / ON dreaming
  evaluator backed by the real ``SqliteEngravaCore`` +
  ``DreamingExtension`` API.
* :mod:`engrava.benchmarks.synthetic.runner` — CLI entry point
  reachable as ``python -m engrava.benchmarks.synthetic``.

The frozen ``synthetic-v1.json`` dataset under
``datasets/synthetic-v1.json`` lands in a later commit on the same
feature branch, together with the ``[tool.setuptools.package-data]``
addition that ships it via the wheel.
"""
