"""Frozen benchmark datasets directory.

Empty at C1 — the frozen ``synthetic-v1.json`` dataset is generated
and committed in a later commit on the same feature branch.  The
corresponding ``[tool.setuptools.package-data]`` entry lands together
with the JSON file so the wheel always matches what is on disk.
"""
