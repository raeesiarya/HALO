"""Audited model backends.

Importing this package registers every bundled backend with
``halo.registry``. Drop a new subpackage here that calls
``register_backend`` in its ``__init__`` and it becomes available to the CLI
with no changes to the audit core.
"""

from models import co_lmlm as co_lmlm  # noqa: F401  (registration side effect)
