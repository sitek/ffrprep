.. _usage:

======
Usage
======

The general usage of ``ffrprep`` is to preprocess frequency-following response (FFR) EEG data in BIDS format. 
``ffrprep`` is a BIDS-compatible preprocessing workflow designed specifically for FFR experiments, providing 
automated preprocessing pipelines that include data loading, filtering, artifact removal, and report generation 
for neurophysiological data analysis.
The exact command to run ``ffrprep`` depends on the Installation method and user. Regarding the latter, ``ffrprep`` 
can either be used as a ``command line tool`` or directly within ``python``. Please refer to the `Tutorial <https://sitek.github.io/ffrprep/walkthrough>`_ for a more detailed walkthrough.

Here's a very conceptual example of running ``ffrprep`` via ``CLI``: ::

    ffrprep 
    ffrprep optional_arguments

and here from within ``python``: ::

    from ffrprep import ffrprep_function
    from ffrprep import ffrprep_function

    result = ffrprep_function(input)

    result = ffrprep_function(input, optional_arguments)

Below, we will focus on the ``CLI`` version. For programmatic use,
the :ref:`api_ref` documents every public function with its
signature and parameters.

ffrprep through the CLI
===========================================

As ``ffrprep`` is a `BIDS-App <https://bids-apps.neuroimaging.io>`_ , it is primarily designed as a command-line tool that you can run directly from your terminal or command prompt.
Ideally, using the provided `Docker <https://sitek.github.io/ffrprep/installation.html#docker>`_ or `Singularity <https://sitek.github.io/ffrprep/installation.html#singularity>`_ images as they encapsulate all dependencies and ensure a consistent environment across different systems, ensuring ease-of-use and
reproducibility.

Downloading the example dataset
===============================

The ``download`` subcommand of the container fetches the example
OSF dataset used in the
`Tutorial <https://sitek.github.io/ffrprep/walkthrough>`_. The
container's entrypoint dispatches ``download …`` to the
``ffrprep-download`` console script bundled inside the image.

.. code-block:: bash

    # Just the EEG data (default)
    docker run --rm \
      -v /path/to/data:/out:rw \
      ksitek/ffrprep:latest \
      download example --out /out

    # Also fetch the stimulus audio + augment events.tsv with the
    # BIDS stim_file column (needed for the stim-vs-response
    # correlation in the analysis report)
    docker run --rm \
      -v /path/to/data:/out:rw \
      ksitek/ffrprep:latest \
      download example --with-stimuli --out /out

``--with-stimuli`` is ``BooleanOptionalAction`` (default ``False``);
``--no-with-stimuli`` is its inverse. The download lands the EEG
data under ``<out>/ffrprep_raw_data/`` and the stimulus files
under ``<out>/ffrprep_raw_data/stimuli/`` per the BIDS spec, then
augments every ``sub-*/eeg/*_events.tsv`` with the matching
``stim_file`` column based on a ``trial_type → filename`` lookup.

Command-Line Arguments
======================
.. argparse::
  :ref: ffrprep.ffrprep_cli.get_parser
  :prog: ffrprep
  :nodefault:
  :nodefaultconst:

Example Call(s)
---------------

The examples below all use Docker — the recommended path. They
build up from the simplest call (preprocessing only, one subject)
to per-trial-type analysis outputs with explicit difference pairs.
Singularity users can swap the ``docker run …`` invocation for
``singularity run --cleanenv -B <host>:<container> <image.sif>``
keeping every flag below the image name identical.

All examples assume your BIDS dataset is at
``/local/bids_dataset`` on the host and the derivatives should
land in ``/local/bids_dataset/derivatives``.

Example 1 - Basic preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    docker run --rm \
      -v /local/bids_dataset:/data:rw \
      ksitek/ffrprep:latest \
      /data \
      /data/derivatives \
      participant \
      --participant_label 01 02 \
      --stage preprocessing

What's in this call:

- ``docker run --rm`` runs the container and removes it after completion.
- ``-v /local/bids_dataset:/data:rw`` mounts the local BIDS dataset
  read-write so derivatives can be written back to disk.
- ``ksitek/ffrprep:latest`` is the published image (see
  :ref:`installation` for tag pinning).
- ``/data`` (1st positional) is the BIDS dataset inside the
  container; ``/data/derivatives`` (2nd) is the output directory;
  ``participant`` (3rd) is the BIDS-App analysis level.
- ``--participant_label 01 02`` processes only ``sub-01`` and
  ``sub-02``.
- ``--stage preprocessing`` runs only the preprocessing stage.


Example 2 - Full analysis with custom parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    docker run --rm \
      -v /local/bids_dataset:/data:rw \
      ksitek/ffrprep:latest \
      /data \
      /data/derivatives \
      participant \
      --stage both \
      --high_pass 0.5 \
      --low_pass 50.0 \
      --ref_channels average \
      --baseline "-0.1,0" \
      --n_procs 4

What's in this call:

- ``--stage both`` runs both preprocessing and analysis stages.
- ``--high_pass 0.5`` / ``--low_pass 50.0`` set the band-pass filter
  in Hz.
- ``--ref_channels average`` uses an average reference.
- ``--baseline "-0.1,0"`` sets the baseline window to −100 ms → 0 ms.
- ``--n_procs 4`` runs 4 per-(task, run) iterations in parallel per
  subject (see :ref:`Parallelization <parallelization>` below).

By default the analysis stage emits one evoked response per
``trial_type`` value in ``events.tsv``
(``_desc-evoked{Cond}.fif``), a combined evoked across all events
(``_desc-evoked.fif``), and — for 2-trial-type datasets — an
auto-paired difference evoked (``_desc-evokedDiff{A}Vs{B}.fif``).
This per-trial-type split is governed by ``--split-by-trial-type``
(default on); pass ``--no-split-by-trial-type`` to emit only the
combined evoked. Examples 3 and 4 below show how to pick explicit
difference pairs or restrict the output set.


Example 3 - Per-trial-type analysis with explicit difference pairs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For datasets with more than two ``trial_type`` values, the
automatic difference is ambiguous; use
``--difference-pairs A:B [C:D …]`` to specify which pairs to
subtract. Each pair becomes one
``_desc-evokedDiff{A}Vs{B}.fif`` file. The per-trial-type
``_desc-evoked{Cond}.fif`` and combined ``_desc-evoked.fif``
files are always emitted.

.. code-block:: bash

    docker run --rm \
      -v /local/bids_dataset:/data:rw \
      ksitek/ffrprep:latest \
      /data \
      /data/derivatives \
      participant \
      --stage both \
      --difference-pairs positive:negative tone1:tone2 \
      --n_procs 4

What's in this call:

- ``--difference-pairs positive:negative tone1:tone2`` emits two
  difference evokeds:
  ``_desc-evokedDiffPositiveVsNegative.fif`` (positive − negative)
  and ``_desc-evokedDiffTone1VsTone2.fif`` (tone1 − tone2).
- All per-trial-type and combined evokeds for this (task, run) group
  are also written — ``--split-by-trial-type`` is on by default, and
  ``--difference-pairs`` requires it (the pairs are subtracted from
  the per-trial-type evokeds, so combining it with
  ``--no-split-by-trial-type`` would leave nothing to subtract).


Example 4 - Restrict outputs to a subset, no split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The trial-type split is controlled by a single boolean flag, exposed
in both forms via ``argparse.BooleanOptionalAction``:

- ``--split-by-trial-type`` (the **default**) emits the combined
  evoked **plus** one ``_desc-evoked{Cond}.fif`` per trial type
  **plus** the auto-paired / explicit difference evokeds. Passing
  this flag explicitly is equivalent to omitting it.
- ``--no-split-by-trial-type`` strips back to the combined evoked
  only — no per-trial-type files, no differences.

To keep the split on but restrict it to a subset of trial types
(while still emitting the combined evoked), pass
``--trial-types A B``.

.. code-block:: bash

    # Combined evoked only — strips per-trial-type files and
    # any (auto-paired or explicit) difference evokeds
    docker run --rm \
      -v /local/bids_dataset:/data:rw \
      ksitek/ffrprep:latest \
      /data \
      /data/derivatives \
      participant \
      --stage both \
      --no-split-by-trial-type \
      --n_procs 4

    # Only emit per-trial-type files for "positive" + the combined
    # evoked; drop the "negative" output even though events.tsv
    # contains both
    docker run --rm \
      -v /local/bids_dataset:/data:rw \
      ksitek/ffrprep:latest \
      /data \
      /data/derivatives \
      participant \
      --stage both \
      --trial-types positive \
      --n_procs 4


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
------------------

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
-------------------------

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
--------------------------------------------

.. code-block:: bash

    parallel -j 8 \
      "ffrprep /data /data/derivatives participant \
         --participant_label {} --n_procs 4" \
      ::: 01 02 03 04 05 06 07 08

Failure handling
----------------

A failure in any (task, run) iteration aborts the run (fail-fast). The
exception propagates up from the worker to the CLI entry point, so the
underlying error is visible in the terminal output. Per-iteration logs
land in ``<work_dir>/<task>-<run>.log`` so you can drill into the
failing iteration without scanning the whole subject log.


Support and communication
=========================

The documentation of this project is found here: https://sitek.github.io/ffrprep.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/sitek/ffrprep/issues.

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