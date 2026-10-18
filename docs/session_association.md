# Sprint 2 (Team 2): Associate monitoring status with the correct exam session

Scope: this item only. Not built: start/stop monitoring functions, webcam-unavailable and permission-denied handling,
backend session lookup, API endpoints.

## What was built
`exam_monitor/session_binding.py`

| Piece | Role |
|---|---|
| `validate_session_id` | Allow-list check (8-64 chars of `[A-Za-z0-9_-]`, `fullmatch`). The ID becomes part of a log file name, so this also blocks path traversal. |
| `MonitoringStatus` | Frozen (immutable) snapshot: `session_id`, `active`, confirmed `state`, `state_since_utc`, `last_event_type`, `last_observation_utc`, `event_count`. Safe to return from an API handler. |
| `_BoundSink` | Wraps the event sink. Raises `SessionMismatch` if an event's `session_id` differs from the bound one. Association is enforced on every write. |
| `SessionMonitor` | One monitor, one session. Owns a state machine and a lock. `observe()` feeds frames; `status()` reads the snapshot. |
| `MonitoringRegistry` | `bind(id)`, `get_status(id)`, `list_active()`, `release(id)`. Rejects invalid IDs, IDs the validator refuses, and a second live monitor for the same ID. |

`bind()` emits `SESSION_STARTED` and `release()` emits `SESSION_ENDED` because those events must carry the session ID.

## Assumptions to confirm with Team 1
1. Session IDs are strings in `[A-Za-z0-9_-]{8,64}` (UUID with hyphens fits). Change `_SESSION_ID_RE` if the backend differs.
2. "Correct session" ultimately means: exists, status ACTIVE, belongs to the logged-in candidate. That is a backend lookup; pass it as `validator=` to `MonitoringRegistry`. Until then only the format is checked.
3. Re-binding an ID after `release()` is allowed (needed to resume after a crash). Change this if a finished session must never be monitored again.
4. One process, one registry. Across processes or machines the duplicate-monitor guarantee does not hold; the backend would need to own that.

## Sources
Retrieved and read while building this:
- Python `dataclasses`, `frozen=True`: https://docs.python.org/3.8/library/dataclasses.html
- Path traversal; validate against known-good input rather than sanitising: https://owasp.org/www-community/attacks/Path_Traversal
- Lock vs RLock re-entrancy (CPython docs commit): https://github.com/python/cpython/commit/7eb59cd95b09ede8c7524e8f71fbea5623715f7f
- PEP 544, structural subtyping (`typing.Protocol`, used for `EventSink`), mirror copy: https://peps-dir-aaturner.readthedocs.io/en/latest/_sources/pep-0544.txt

Standard references NOT retrieved this session (I could not open them; verify before citing in a report):
- https://docs.python.org/3/library/threading.html
- https://docs.python.org/3/library/re.html (`fullmatch`; the `$` trailing-newline behaviour was confirmed by running it, see tests)
- https://peps.python.org/pep-0544/

The code is original to this project; the sources document the language features and security practice it uses.
Nothing was copied from a repository.
