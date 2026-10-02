"""engrava extensions package.

Extensions provide optional capabilities to engrava:
- ``dreaming``: Periodic memory consolidation

KNN vector search via sqlite-vec lives in
:mod:`engrava.infrastructure.sqlite.vector_sqlite_vec` — it is a SQLite
adapter, not an extension. ``engrava.extensions.vector_sqlite_vec`` remains
importable as a backward-compatible alias to that module.
"""
