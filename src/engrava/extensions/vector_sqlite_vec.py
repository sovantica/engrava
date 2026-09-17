"""Backward-compatible alias for the relocated SQLite vector-search backend.

``engrava.extensions.vector_sqlite_vec`` was never an extension in the
architectural sense used by this package: it is a SQLite adapter, not an
optional consolidation capability, and it now lives at
:mod:`engrava.infrastructure.sqlite.vector_sqlite_vec`.

This module is an **alias**, not a re-export. It installs the *same* module
object under this name in :data:`sys.modules`, so
``sys.modules[__name__] is sys.modules["engrava.infrastructure.sqlite.vector_sqlite_vec"]``
holds for the lifetime of the process. A name-based re-export (``from
engrava.infrastructure.sqlite.vector_sqlite_vec import *``, for instance)
would instead leave two distinct module objects that merely look alike, and
a consumer who monkeypatches an attribute through one of them — a test, or a
downstream package built on this one — would silently fail to affect code
that imports through the other. The alias makes both names resolve to the
one object, so a patch applied through either path is observed through
both.
"""

from __future__ import annotations

import sys

from engrava.infrastructure.sqlite import vector_sqlite_vec as _vector_sqlite_vec

sys.modules[__name__] = _vector_sqlite_vec
