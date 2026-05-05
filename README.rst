=======
ffrprep
=======


.. image:: https://github.com/spark-csd/ffrprep/actions/workflows/docs.yml/badge.svg
        :target: https://github.com/spark-csd/ffrprep/actions/workflows/docs.yml

.. image:: https://img.shields.io/pypi/v/ffrprep.svg
        :target: https://pypi.python.org/pypi/ffrprep

.. image:: https://img.shields.io/docker/pulls/sparkcsd/ffrprep
    :alt: Dockerpulls
    :target: https://cloud.docker.com/u/sparkcsd/repository/docker/sparkcsd/ffrprep

.. image:: https://img.shields.io/github/repo-size/spark-csd/ffrprep.svg
        :target: https://img.shields.io/github/repo-size/spark-csd/ffrprep.zip

.. image:: https://img.shields.io/github/issues/spark-csd/ffrprep.svg
        :target: https://img.shields.io/github/issues/spark-csd/ffrprep/issues

.. image:: https://img.shields.io/github/issues-pr/spark-csd/ffrprep.svg
        :target: https://img.shields.io/github/issues-pr/spark-csd/ffrprep/pulls

.. image:: https://img.shields.io/github/license/spark-csd/ffrprep.svg
        :target: https://github.com/spark-csd/ffrprep



``ffrprep`` is a preprocessing and analysis pipeline for frequency-following
response (FFR) EEG data, packaged as a
`BIDS-App <https://bids-apps.neuroimaging.io>`_.


Overview
========

A typical run loads each subject's raw recordings from a
`BIDS <https://bids-specification.readthedocs.io>`_ dataset, applies
referencing and band-pass filtering, epochs around stimulus events,
and (optionally) averages epochs into evoked responses with
FFR-specific metrics — RMS SNR, autocorrelation, pitch tracking.
Outputs land in a BIDS-derivatives layout with JSON sidecars and a
single-file HTML report per subject and stage. The CLI parallelizes
per-(task, run) iterations within a subject via ``--n_procs N``;
multi-node / cluster scaling is one ffrprep invocation per subject
under your scheduler of choice.


Installation
============

See the `installation guide
<https://SPARK-CSD.github.io/ffrprep/installation.html>`_ for the
Docker image, Singularity image, and local Python install via
``uv``.


Quick start
===========

Process two subjects, both preprocessing and analysis stages, with
four parallel workers per subject:

.. code-block:: bash

    docker run --rm \
      -v /path/to/your/bids_dataset:/data:rw \
      sparkcsd/ffrprep:latest \
      /data /data/derivatives participant \
      --participant_label 01 02 \
      --stage both \
      --n_procs 4

Outputs land in ``/path/to/your/bids_dataset/derivatives/`` under
``ffrprep-preprocessing/`` and ``ffrprep-analysis/``, each containing
a ``sub-XX_<stage>_report.html`` per subject.

For cluster runs: one ffrprep invocation per subject (slurm job
array, GNU parallel, etc.) composes with ``--n_procs`` for
intra-subject parallelism. The
`usage docs <https://SPARK-CSD.github.io/ffrprep/usage.html>`_
include slurm and GNU parallel examples.


Documentation
=============

Full documentation — pipeline walkthrough, CLI reference, API
reference — at https://SPARK-CSD.github.io/ffrprep.
