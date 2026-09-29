"""Shared, hostile-exception-safe describers for CLI failure and cleanup logging.

Three call sites across two modules need the same two things from an
arbitrary exception -- one they did not control the raising of, and which may
itself be adversarial:

* :func:`~engrava.cli.memory_commands._error_boundary`'s fallback message and
  its ``DEBUG``-level stack log, for the exception a memory-verb's own body
  raised.
* :func:`~engrava.cli.memory_commands._opened_full_store`'s cleanup-failure
  warning, for a ``--config``-tier store's ``close()`` failing while that
  original exception is still propagating.
* :func:`~engrava.cli.main._close_quietly`'s cleanup-failure warning, for a
  bare/default-tier connection's ``close()`` failing the same way -- reached
  from :func:`~engrava.cli.memory_commands._opened_full_store`'s bare branch
  via :func:`~engrava.cli.main._opened_db`, and also from two cleanup sites in
  ``main.py`` itself that belong to other built-in commands.

``memory_commands`` imports from ``main`` (for ``_opened_db``, ``_run`` and
``cli`` itself) and ``main`` imports ``memory_commands`` at the bottom of the
module, to register the memory verbs as commands -- so neither module can
define these helpers and have the other import them at module level without
a cycle. Both import them from here instead: one implementation, not a copy
per module. A duplicated copy is exactly how this kind of fix drifts: a fix
applied to one module's copy leaves the other module's copy behind.

Both functions below read attributes off an exception they did not raise and
cannot trust, so both are written to survive one that fights back: a
``__str__`` or ``__name__`` that raises, one that raises
``KeyboardInterrupt``/``SystemExit`` instead of an ordinary exception, one
whose result is a hostile ``str`` subclass, and one with an overridden
``__getattribute__``. See each function's own docstring for the specific
hazards that shape its current form.
"""

from __future__ import annotations

import traceback

from engrava.config_validation import own_str


def _describe_exception(exc: BaseException) -> str:
    r"""Build a complete, safe ``"TypeName: text"`` description of any exception.

    Both halves of that description are arbitrary code, not just the text
    half. ``str(exc)`` calls an exception's own ``__str__``, which a
    library-defined exception can override to raise -- exactly the kind of
    thing the callers below exist to convert cleanly rather than let escape
    as a second, unrelated failure. ``type(exc).__name__`` looks equally safe
    but is not: it is a class-attribute lookup, which for a class whose
    *metaclass* overrides attribute access (or defines ``__name__`` as a
    property that raises) runs that metaclass's own code instead of
    returning a guaranteed string.

    Obtaining each half safely is not the whole job: a value that comes back
    without raising can still be unsafe to use. An exception whose
    ``__str__`` returns a ``str`` subclass instance whose own ``__format__``
    raises would defeat a guard that reads the value safely but then
    interpolates it into ``f"{type_name}: {text}"`` unguarded -- exactly as
    escapable as not guarding the read at all, because the interpolation
    would be where the hostile code actually runs. Narrowing either guard to
    ``except Exception`` would have the same effect: a metaclass raising
    ``KeyboardInterrupt`` on ``__name__``, or a ``__str__`` raising
    ``SystemExit``, would walk straight through -- which is why neither
    guard below is narrowed that way.

    The fix is structural, not two more named cases: **no value leaves
    either guarded block except an exact, already-safe ``str``.** Each
    block still catches ``BaseException`` broadly, but re-raises
    ``KeyboardInterrupt`` and ``SystemExit`` immediately instead of
    converting them (see the guards below) -- converting a real Ctrl-C or
    ``sys.exit()`` into an ordinary-looking error object is worse than an
    endless hostile ``__str__`` staying endless. Everything else --
    including a control-flow exception raised by code that had no business
    raising one in the first place -- still converts cleanly, and, still
    inside that same guard, normalizes whatever it obtained with
    :func:`~engrava.config_validation.own_str`, the same primitive the
    config layer uses to close exactly this gap for a validated value.
    It uses ``own_str`` rather than ``"".join(...)`` because ``str.join``
    walks its argument with a plain ``for`` loop, which calls the argument's
    own ``__iter__``, and a ``str`` subclass can override ``__iter__`` to
    yield arbitrarily many characters from a short underlying buffer.
    :func:`~engrava.config_validation.own_str` avoids that: it is
    ``str.__str__`` resolved on the built-in type rather than on the
    instance, so no subclass method -- ``__iter__``, ``__format__``,
    ``__repr__``, or anything else -- ever runs; it reads the real
    underlying buffer directly and hands back an exact ``str``, with no
    attacker code executing inside the guard at all. That does not bound
    the ``KeyboardInterrupt`` window, though: obtaining each half in the
    first place still calls ``str(exc)`` and ``type(exc).__name__`` --
    both attacker code, and both running *before* ``own_str`` ever sees a
    result -- so that read can still block indefinitely, exactly as an
    endless hostile ``__str__`` can. Normalization cannot bound a read
    that already happened; what protects the user is that a
    ``KeyboardInterrupt`` or ``SystemExit`` raised anywhere in that read
    now re-raises immediately instead of being converted, so a deliberate
    Ctrl-C still escapes even out of a call already in flight.
    For a value that already *is* an exact ``str`` this is the identity and
    costs nothing. If the raw value is not string-like at all,
    ``str.__str__`` itself raises ``TypeError``, and the *same*
    ``except BaseException`` clause substitutes the fixed placeholder. By
    the time either half reaches the final f-string, it is a plain ``str``
    with a plain ``str.__format__``, so that interpolation can no longer be
    the thing that raises.

    This function is the place in the failure path that deliberately reads
    either attribute off an arbitrary exception under a guard. The
    boundary's fallback message and its ``DEBUG``-level stack log both use
    this function's result for the same description. An earlier shape's
    debug log line also passed ``exc_info=True``, which made Python's own
    traceback machinery render the original exception a second time, calling
    its ``__str__`` again outside anything defined here; a ``KeyboardInterrupt``
    or ``SystemExit`` raised during that second rendering was caught by the
    standard library's own formatting code rather than propagating, so a
    real Ctrl-C arriving during it could still end in that command emitting
    an ``unexpected_error`` object at exit ``1`` instead of aborting.
    :func:`~engrava.cli.memory_commands._error_boundary` no longer does
    this: its ``DEBUG`` log, :func:`~engrava.cli.memory_commands._opened_full_store`'s
    cleanup-failure warning (``--config`` tier), and
    :func:`~engrava.cli.main._close_quietly`'s cleanup-failure warning
    (bare/default tier) all build their stack text through
    :func:`_frame_only_stack` instead, which reads only frame metadata --
    through the built-in traceback descriptor, not a plain attribute read --
    so the second read this paragraph used to describe no longer happens at
    any of the three sites.

    **Dropping the description entirely at the two close-failure sites would
    be a regression.** Frame metadata alone says *where* closing failed,
    never *why* -- an ordinary ``PermissionError``, a full disk, or a locked
    file all look identical in a stack of file names and line numbers, which
    is a real loss for an ordinary, non-hostile user. Both close-failure
    warnings also call this function once, for the close exception itself,
    and log its result
    alongside the frame-only stack. That is not a second render of
    anything: it is the *same* single, guarded, non-absorbing attempt this
    function always made, applied to the close exception the way it was
    already applied to the body's original exception -- so a real
    ``PermissionError`` comes back into the log as
    ``"PermissionError: [Errno 13] ..."`` while a hostile close exception's
    ``KeyboardInterrupt``/``SystemExit`` still escapes instead of being
    absorbed, and the forbidden second render (``exc_info=True`` walking the
    close exception, the original exception a second time, and any
    exception-group children, all through their own overridable formatters)
    stays gone.

    **Deliberate limit.** This function converts every ordinary
    ``Exception`` raised while obtaining or normalizing either half, but
    re-raises ``KeyboardInterrupt`` and ``SystemExit`` rather than
    converting them -- see above for why a safe *read* is not the same as
    a bounded one. :func:`~engrava.cli.memory_commands._error_boundary`
    itself still catches only ``Exception``: a ``SystemExit`` or
    ``KeyboardInterrupt`` raised directly by a command's own body also
    passes through the boundary untouched. Both paths now agree for the
    same reason -- those are control-flow signals a caller or the
    interpreter itself raises on purpose, not failures this CLI should
    repackage as JSON -- whether the signal originates in the command body
    or in code this function is merely *describing*.

    Args:
        exc: The exception to describe.

    Returns:
        ``"{type name}: {text}"``, substituting a fixed placeholder for
        whichever half could not be obtained and safely normalized. Both
        halves are guaranteed to be exact ``str`` instances before they are
        interpolated.

    Raises:
        KeyboardInterrupt: If obtaining or normalizing either half raises
            one instead of returning normally.
        SystemExit: If obtaining or normalizing either half raises one
            instead of returning normally.

    """
    try:
        type_name = own_str(type(exc).__name__)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # noqa: BLE001 -- deliberate; converts everything else; see above
        type_name = "<type name unavailable>"
    try:
        text = own_str(str(exc))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # noqa: BLE001 -- deliberate; converts everything else; see above
        text = "<str() raised>"
    return f"{type_name}: {text}"


def _frame_only_stack(exc: BaseException) -> str:
    r"""Render ``exc``'s call stack as ``file:line in function`` text, nothing else.

    Shared by :func:`~engrava.cli.memory_commands._error_boundary`'s
    ``DEBUG`` stack log, :func:`~engrava.cli.memory_commands._opened_full_store`'s
    cleanup-failure warning (``--config`` tier), and
    :func:`~engrava.cli.main._close_quietly`'s cleanup-failure warning
    (bare/default tier) -- all three want "where did this happen", never
    "what does this exception say about itself". ``exc_info=True`` would
    supply the location, but it also makes Python's own traceback formatter
    call the exception's ``__str__``, outside every guard this module builds
    elsewhere.

    Each frame contributes only its filename, line number and function
    name -- no source line, no local values, no chained exception, no
    exception-group children -- so nothing this function does can invoke
    exception-controlled formatting at all, let alone a second time.

    The traceback itself is obtained through the built-in descriptor,
    ``BaseException.__traceback__.__get__(exc)``, not the plain attribute
    read ``exc.__traceback__``. The plain form is an ordinary attribute
    lookup, which for an instance of a class (or subclass) that overrides
    ``__getattribute__`` runs that override's own code -- exactly the kind
    of attacker-reachable read this whole boundary exists to avoid, and not
    a hypothetical one: an exception instrumented with an overridden
    ``__getattribute__`` records one call to it through ``exc.__traceback__``,
    zero through the descriptor form used here. Resolving ``__traceback__``
    on ``BaseException`` itself and calling ``__get__`` on it goes straight
    to the built-in getset without ever consulting the instance's (or a
    subclass's) ``__getattribute__``, the same way
    :func:`~engrava.config_validation.own_str` resolves ``str.__str__`` on
    the built-in type rather than the instance to sidestep an overridden
    ``__iter__``.

    Args:
        exc: The exception whose stack to render.

    Returns:
        One ``"  {filename}:{lineno} in {function}"`` line per frame, joined
        with newlines, or ``"  <no frames>"`` if the traceback carries no
        frames at all.

    """
    # typeshed types `__traceback__` by its instance-attribute annotation
    # even when read off the class itself, so mypy sees `TracebackType |
    # None` here instead of the getset_descriptor this actually is at
    # runtime (confirmed: `BaseException.__traceback__.__get__` resolves and
    # calling it returns the real traceback, bypassing `exc`'s own
    # `__getattribute__` -- see this function's docstring).
    tb = BaseException.__traceback__.__get__(exc)  # type: ignore[union-attr]
    frame_lines = [
        f"  {frame.f_code.co_filename}:{lineno} in {frame.f_code.co_name}"
        for frame, lineno in traceback.walk_tb(tb)
    ]
    return "\n".join(frame_lines) if frame_lines else "  <no frames>"
