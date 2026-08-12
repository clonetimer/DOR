# Guide: Partial Mechanism Graph completion

Represent prior knowledge as two mappings:

```json
{"known_present": {"i": ["V"], "w": ["TL"]},
 "known_absent": {"i": ["TL"], "w": ["V"]}}
```

`known_present` is audited against the recovered coefficients. `known_absent` constrains supported sparse backends. Keep uncertain relations unknown rather than marking them absent.

A completion is accepted only after structure validation, source identifiability, and coverage certification.
