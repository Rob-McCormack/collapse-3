"""Reproducible Collapse3 experiments.

Each module is runnable as ``python -m experiments.<name>`` from the repo root.
Every run records full provenance (git commit, config, seed, environment) next
to its results via :mod:`experiments._provenance`, so any number in the write-up
can be traced back to the exact code and configuration that produced it.
"""
