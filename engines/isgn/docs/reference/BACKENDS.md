# Backend status

Core discovery is solver-agnostic.

- `native`: historical hierarchical ISGN research solver; retained for regression/research.
- `reference-sindy`: small offline STLSQ-SINDYc-compatible reference backend. **Not official PySINDy.**
- `sklearn-lasso`: scale-normalized sparse linear backend.
- `pysindy`: optional adapter to official PySINDy; requires optional dependency.
- `symmatika`: optional external symbolic bridge; requires external executable.

The practical release should never rename a fallback/reference implementation as an unavailable external method.
