"""End-to-end pipelines that chain the layers of the fidelity ladder.

Each module here is a thin, tested composition of models, the data layer and
the ROM / twin layers. The scripts in ``examples/`` call these functions and
add plotting.
"""
