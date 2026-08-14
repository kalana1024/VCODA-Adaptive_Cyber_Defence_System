# Contributing

1. Create a separate branch.
2. Add or update tests for behavioural changes.
3. Run `python -m compileall -q src scripts dashboard`.
4. Run `pytest`.
5. Do not commit datasets, credentials, PCAPs containing private traffic, or trained models containing sensitive metadata.
6. New response actions must be predefined, reversible where possible, and disabled by default.
7. New model loaders must verify checksums before deserialisation.
