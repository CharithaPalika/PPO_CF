"""Thin driver layer between the Slurm job scripts and the experiment code.

    stages.py    the ONLY project-specific file: what E1 is, and how to run
                 one unit of it
    manifest.py  builds the agreed run list for one stage
    chunk.py     runs one contiguous slice of that list (one array task)
    analyse.py   merges the chunks, then requeues or summarises and chains on
    smoke.py     the smallest real training step, for slurm/00_verify.sbatch

Nothing here knows a hyperparameter. Everything it needs about the experiment
comes from `config/envs/*.yaml` through `stages.py`.
"""
