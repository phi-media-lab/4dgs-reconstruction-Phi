from __future__ import annotations


class P2GError(Exception):
    """Base error for expected pipeline failures."""


class ContractError(P2GError):
    """A typed data contract was violated."""


class IdentityError(P2GError):
    """Runtime identity could not satisfy a requested gate."""


class OutputExistsError(P2GError):
    """An evidence output would overwrite an existing path."""
