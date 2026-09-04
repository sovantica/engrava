"""``ReferentialIntegrityError``, ``DuplicateEdgeError``, and ``CoreMigrationError``
are importable from the package root.

All three are raised by public methods (``create_edge`` for the first two,
``ensure_schema`` for the third) but, until now, could only be caught by
reaching into ``engrava.domain.exceptions`` — a path no other exception in
the top-level surface required. This module pins the fix mechanically: each
name must resolve from ``import engrava`` to the *exact* class the raising
code raises, not a same-named duplicate reachable by some other route.
"""

from __future__ import annotations

import engrava
from engrava.domain import exceptions as domain_exceptions


class TestTopLevelExceptionExports:
    """``from engrava import <name>`` must yield the raising code's own class."""

    def test_referential_integrity_error_is_importable_and_identical(self) -> None:
        from engrava import ReferentialIntegrityError

        assert ReferentialIntegrityError is domain_exceptions.ReferentialIntegrityError
        assert "ReferentialIntegrityError" in engrava.__all__

    def test_duplicate_edge_error_is_importable_and_identical(self) -> None:
        from engrava import DuplicateEdgeError

        assert DuplicateEdgeError is domain_exceptions.DuplicateEdgeError
        assert "DuplicateEdgeError" in engrava.__all__

    def test_core_migration_error_is_importable_and_identical(self) -> None:
        from engrava import CoreMigrationError

        assert CoreMigrationError is domain_exceptions.CoreMigrationError
        assert "CoreMigrationError" in engrava.__all__
