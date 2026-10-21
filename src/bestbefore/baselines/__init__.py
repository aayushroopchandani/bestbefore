"""The readers every model has to beat (docs/m0-data-and-test-set.md, Part C).

A reader takes photo paths and yields, per photo, a dict holding at least
`expiry`: the date it reads as the expiry (an `Expiry`), or None.
"""
