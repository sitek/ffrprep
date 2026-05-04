.. _usage:

======
Usage
======

The general usage of ``ffrprep`` is to preprocess frequency-following response (FFR) EEG data in BIDS format. 
``ffrprep`` is a BIDS-compatible preprocessing workflow designed specifically for FFR experiments, providing 
automated preprocessing pipelines that include data loading, filtering, artifact removal, and report generation 
for neurophysiological data analysis.
The exact command to run ``ffrprep`` depends on the Installation method and user. Regarding the latter, ``ffrprep`` 
can either be used as a ``command line tool`` or directly within ``python``. Please refer to the `Tutorial <https://SPARK-CSD.github.io/ffrprep/walkthrough>`_ for a more detailed walkthrough.

Here's a very conceptual example of running ``ffrprep`` via ``CLI``: ::

    ffrprep 
    ffrprep optional_arguments

and here from within ``python``: ::

    from ffrprep import ffrprep_function
    from ffrprep import ffrprep_function

    result = ffrprep_function(input)

    result = ffrprep_function(input, optional_arguments)

Below, we will focus on the ``CLI`` version. Thus, if you are interested in using ``ffrprep`` directly within ``python``,
please check the `Examples <https://SPARK-CSD.github.io/ffrprep/auto_examples/index>`_.

ffrprep through the CLI
===========================================

As ``ffrprep`` is a `BIDS-App <https://bids-apps.neuroimaging.io>`_ , it is primarily designed as a command-line tool, that you can directly from your terminal or command prompt.  
Ideally, using the provided `Docker <https://spark-csd.github.io/ffrprep/installation.html#docker>`_ or `Singularity <https://spark-csd.github.io/ffrprep/installation.html#singularity>`_ images as they encapsulate all dependencies and ensure a consistent environment across different systems, ensuring ease-of-use and
reproducibility.

Command-Line Arguments
======================
.. argparse::
  :ref: ffrprep.ffrprep_cli.get_parser
  :prog: ffrprep
  :nodefault:
  :nodefaultconst:

Example Call(s)
---------------

Below you'll find two examples calls that hopefully help you to familiarize yourself with ``ffrprep`` and its options.
We will start with the general structure of an ``ffrprep`` call and subsequently explain how to utilize it via the ``docker``
or ``singularity`` images.

Example 1 - Basic preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    ffrprep \
    /data/bids_dataset \
    /data/bids_dataset/derivatives \
    participant \
    --participant_label 01 02 \
    --stage preprocessing

Here's what's in this call:

- The 1st positional argument ``/data/bids_dataset`` is the input BIDS dataset directory
- The 2nd positional argument ``/data/bids_dataset/derivatives`` indicates the output directory for results
- The 3rd positional argument ``participant`` specifies participant-level analysis
- ``--participant_label 01 02`` processes only subjects sub-01 and sub-02
- ``--stage preprocessing`` runs only the preprocessing stage


Example 2 - Full analysis with custom parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    ffrprep \
    /data/bids_dataset \
    /data/bids_dataset/derivatives \
    participant \
    --stage both \
    --high_pass 0.5 \
    --low_pass 50.0 \
    --ref_channels average \
    --baseline "-0.1,0" \
    --by_event_type

Here's what's in this call:

- The 1st positional argument ``/data/bids_dataset`` is the input BIDS dataset directory
- The 2nd positional argument ``/data/bids_dataset/derivatives`` indicates the output directory for results
- The 3rd positional argument ``participant`` specifies participant-level analysis
- ``--stage both`` runs both preprocessing and analysis stages
- ``--high_pass 0.5`` sets high-pass filter to 0.5 Hz
- ``--low_pass 50.0`` sets low-pass filter to 50 Hz
- ``--ref_channels average`` uses average reference
- ``--baseline "-0.1,0"`` sets baseline from -100ms to 0ms
- ``--by_event_type`` creates separate evoked responses for each event type 


Example 3 - Usage through Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    docker run -ti --rm \
    -v /local/bids_dataset:/data:ro \
    -v /local/bids_dataset/derivatives:/outputs \
    ffrprep/ffrprep:latest \
    /data \
    /outputs \
    participant \
    --participant_label 01 \
    --stage both \
    --high_pass 1.0 \
    --low_pass 40.0 \
    --n_procs 4

Here's what's in this call:

- ``docker run -ti --rm`` runs the Docker container interactively and removes it after completion
- ``-v /local/bids_dataset:/data:ro`` mounts local BIDS data directory as read-only
- ``-v /local/bids_dataset/derivatives:/outputs`` mounts local output directory
- ``ffrprep/ffrprep:latest`` specifies the Docker image to use
- ``/data`` is the BIDS dataset directory (inside container)
- ``/outputs`` is the output directory (inside container)
- ``participant`` specifies participant-level analysis
- ``--participant_label 01`` processes only subject sub-01
- ``--n_procs 4`` runs 4 (task, run) iterations in parallel per subject (see
  :ref:`Parallelization <parallelization>` below)


Example 4 - Usage through Singularity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    singularity run --cleanenv \
    -B /local/bids_dataset:/data:ro \
    -B /local/bids_dataset/derivatives:/outputs \
    ffrprep_latest.sif \
    /data \
    /outputs \
    participant \
    --participant_label 01 02 03 \
    --stage preprocessing \
    --high_pass 2.0 \
    --ref_channels "Cz,Fz" \
    --tmin -0.1 \
    --tmax 0.5

Here's what's in this call:

- ``singularity run --cleanenv`` runs the Singularity container with a clean environment
- ``-B /local/bids_dataset:/data:ro`` binds local BIDS data directory as read-only
- ``-B /local/bids_dataset/derivatives:/outputs`` binds local output directory
- ``ffrprep_latest.sif`` specifies the Singularity image file to use
- ``/data`` is the BIDS dataset directory (inside container)
- ``/outputs`` is the output directory (inside container) 
- ``participant`` specifies participant-level analysis
- ``--participant_label 01 02 03`` processes subjects sub-01, sub-02, and sub-03
- ``--stage preprocessing`` runs only the preprocessing stage
- ``--high_pass 2.0`` sets high-pass filter to 2.0 Hz
- ``--ref_channels "Cz,Fz"`` uses Cz and Fz channels as reference
- ``--tmin -0.1`` sets epoch start time to -100ms
- ``--tmax 0.5`` sets epoch end time to 500ms


.. _parallelization:

Parallelization and cluster usage
=================================

``ffrprep`` follows the standard BIDS-App parallelism model:

* **Inside one invocation** — ``--n_procs N`` runs N per-(task, run) iterations
  concurrently for the subject(s) being processed. Each worker runs its own
  preprocessing or analysis workflow with the Nipype ``Linear`` plugin.
  Default: ``--n_procs 1`` (sequential). Memory footprint scales linearly
  with N — each worker loads its own raw + epochs into memory, so dial it
  down on small machines.

* **Across invocations** — for multi-node / cluster scaling, run one
  ``ffrprep`` invocation per subject under your scheduler (slurm job array,
  GNU parallel, HTCondor, etc.). Both layers compose: 8 parallel slurm
  tasks each with ``--n_procs 4`` gives 32-way effective parallelism.

Single workstation
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    ffrprep \
    /data/bids_dataset \
    /data/bids_dataset/derivatives \
    participant \
    --participant_label 01 02 03 \
    --n_procs 4

The 4 workers chew through each subject's (task, run) iterations in parallel,
then move to the next subject.

Cluster (slurm job array)
~~~~~~~~~~~~~~~~~~~~~~~~~

Run one invocation per subject via the scheduler; each job uses
``--n_procs`` for intra-subject parallelism. Example wrapper script:

.. code-block:: bash

    # ffrprep_one_subject.sh
    #!/bin/bash
    #SBATCH --array=0-99
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=16G
    SUBJECTS=(sub-01 sub-02 sub-03 ...)
    SUB=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
    ffrprep /data /data/derivatives participant \
        --participant_label ${SUB#sub-} \
        --n_procs $SLURM_CPUS_PER_TASK

Cluster (GNU parallel on a single beefy box)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    parallel -j 8 \
      "ffrprep /data /data/derivatives participant \
         --participant_label {} --n_procs 4" \
      ::: 01 02 03 04 05 06 07 08

Failure handling
~~~~~~~~~~~~~~~~

A failure in any (task, run) iteration aborts the run (fail-fast). The
exception propagates up from the worker to the CLI entry point, so the
underlying error is visible in the terminal output. Per-iteration logs
land in ``<work_dir>/<task>-<run>.log`` so you can drill into the
failing iteration without scanning the whole subject log.


Support and communication
=========================

The documentation of this project is found here: https://YourGitHubHandle.github.io/ffrprep.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/YourGitHubHandle/ffrprep/issues.

If you have a problem or would like to ask a question about how to use ``ffrprep``,
please submit a question to `NeuroStars.org <http://neurostars.org/tags/ffrprep>`_ with an ``ffrprep`` tag.
NeuroStars.org is a platform similar to StackOverflow but dedicated to neuroinformatics.

All previous ``ffrprep`` questions are available here:
http://neurostars.org/tags/ffrprep/

Not running on a local machine? - Data transfer
===============================================

Please contact you local system administrator regarding
possible and favourable transfer options (e.g., `rsync <https://rsync.samba.org/>`_
or `FileZilla <https://filezilla-project.org/>`_).

A very comprehensive approach would be `Datalad
<http://www.datalad.org/>`_, which will handle data transfers with the
appropriate settings and commands.
Datalad also performs version control over your data.