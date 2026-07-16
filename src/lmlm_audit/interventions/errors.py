class AuditIntegrationError(RuntimeError):
    """A model's search index or generator does not satisfy the audit's
    expected interface (e.g. an unexpected result shape)."""


class ExclusionSearchExhaustedError(AuditIntegrationError):
    """The exclusion filter ran out of over-retrieval budget before finding
    enough retained candidates."""
