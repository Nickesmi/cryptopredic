"""Alpha Research Engine (Phase 5 of the timing audit).

This package is a *research* component, explicitly separate from the
production pipeline:

    Forecasting Engine  (src/models/)       -- Phase 3 verdict: NOT RELIABLE
    Alpha Research Engine  (src/research/)  -- this package, Phase 5
    Opportunity Scanner  (src/ranking/)     -- Phase 4 verdict: WEAK
    Risk Engine                             -- does not exist as a component
    Execution Engine                        -- does not exist as a component

Nothing in this package is imported by ``src/api/``, ``src/ranking/``, or
``src/models/`` — a signal only reaches production if a human, after
reading a Phase 5-style report, explicitly wires it into
``src/ranking/score_coins.py`` and re-runs the Phase 4 audit process on
the changed scanner. Running an experiment in this package can never
change what the live system recommends.
"""
