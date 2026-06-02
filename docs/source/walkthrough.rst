.. _walkthrough:

====================
Tutorial walkthrough
====================

This tutorial provides a complete walkthrough of using ``ffrprep`` for frequency-following response (FFR) EEG data preprocessing and analysis. We'll cover everything from downloading example data to inspecting the final outputs.

Prerequisites
=============

Before starting, ensure you have:

- Docker or Singularity installed on your system
- Python environment with ``ffrprep`` installed (for data download)
- Sufficient disk space (~2GB for example data and outputs)

Step 1: Download Example Data
=============================

First, let's download some example FFR data to work with. The
``download`` subcommand of the ``ffrprep`` container fetches the
example OSF dataset; the container's entrypoint dispatches it to
the ``ffrprep-download`` console script bundled inside the image.

.. code-block:: bash

    mkdir -p ~/ffrprep_tutorial

    # Fetch the EEG data AND the stimulus files referenced by
    # events.tsv. --with-stimuli augments every
    # sub-*/eeg/*_events.tsv with the BIDS stim_file column so the
    # analysis report can compute the stim ↔ response
    # cross-correlation. Drop the flag (or pass --no-with-stimuli)
    # to skip the stimulus download.
    docker run --rm \
      -v ~/ffrprep_tutorial:/out:rw \
      ksitek/ffrprep:latest \
      download example --with-stimuli --out /out

The same fetch is available from Python via
``ffrprep.datasets.download_example_data``:

.. code-block:: python

    from ffrprep.datasets import download_example_data
    import os

    work_dir = os.path.expanduser("~/ffrprep_tutorial")
    os.makedirs(work_dir, exist_ok=True)
    dataset_path = download_example_data(
        dataset_path=work_dir, with_stimuli=True,
    )
    print(f"Example data downloaded to: {dataset_path}")

Either form produces a complete BIDS dataset with:

- Raw EEG data from one subject
- Proper BIDS directory structure
- Required metadata files (``dataset_description.json``, etc.)
- Event files and channel information (with the ``stim_file``
  column populated when ``with_stimuli=True``)
- ``stimuli/<filename>.wav`` audio files (when
  ``with_stimuli=True``)

Step 2: Get the ffrprep Container
=================================

Next, obtain the ``ffrprep`` container image. Choose either Docker or Singularity:

**Option A: Docker**

.. code-block:: bash

    # Pull the latest ffrprep Docker image
    docker pull ksitek/ffrprep:latest
    
    # Verify the image was downloaded
    docker images | grep ffrprep

**Option B: Singularity**

.. code-block:: bash

    # Build Singularity image from Docker Hub
    singularity build ffrprep_latest.sif docker://ksitek/ffrprep:latest
    
    # Verify the image was created
    ls -lh ffrprep_latest.sif

Step 3: Run Preprocessing
=========================

Now let's run the preprocessing stage on our example data:

**Using Docker:**

.. code-block:: bash

    # Navigate to your working directory
    cd ~/ffrprep_tutorial
    
    # Run preprocessing with Docker
    docker run -ti --rm \
        -v $(pwd):/data \
        ksitek/ffrprep:latest \
        /data/bids_dataset \
        /data/bids_dataset/derivatives \
        participant \
        --stage preprocessing \
        --high_pass 1.0 \
        --low_pass 40.0 \
        --ref_channels average \
        --baseline "-0.2,0" \
        --tmin -0.2 \
        --tmax 0.6

**Using Singularity:**

.. code-block:: bash

    # Navigate to your working directory  
    cd ~/ffrprep_tutorial
    
    # Run preprocessing with Singularity
    singularity run --cleanenv \
        -B $(pwd):/data \
        ffrprep_latest.sif \
        /data/bids_dataset \
        /data/bids_dataset/derivatives \
        participant \
        --stage preprocessing \
        --high_pass 1.0 \
        --low_pass 40.0 \
        --ref_channels average \
        --baseline "-0.2,0" \
        --tmin -0.2 \
        --tmax 0.6

The preprocessing stage will:

- Load raw EEG data from the BIDS dataset
- Apply high-pass (1.0 Hz) and low-pass (40.0 Hz) filters
- Re-reference to average reference
- Extract epochs from -200ms to 600ms around events
- Apply baseline correction from -200ms to 0ms
- Save preprocessed data in MNE format

Step 4: Inspect Preprocessing Reports
=====================================

After preprocessing completes, inspect the generated reports:

.. code-block:: bash

    # Navigate to the preprocessing outputs
    cd ~/ffrprep_tutorial/bids_dataset/derivatives/ffrprep-preprocessing

    # List the generated files
    find . -name "*.fif" -o -name "*.json" -o -name "*.html"

    # Open the preprocessing report in your browser
    open sub-*/eeg/sub-*_preprocessing_report.html  # macOS
    # or
    xdg-open sub-*/eeg/sub-*_preprocessing_report.html  # Linux

The preprocessing report will show:

- Raw data quality metrics
- Filter responses and effects
- Epoch rejection statistics  
- Channel-wise signal quality
- Event-related potential previews

**Key things to check:**

- **Data quality**: Look for excessive noise or artifacts
- **Epoch rejection**: Ensure reasonable rejection rates (<30%)
- **Filter effects**: Verify filters didn't distort your signal of interest
- **Event timing**: Confirm events are properly aligned

Step 5: Run Analysis
====================

Once preprocessing is complete and looks good, run the analysis stage:

**Using Docker:**

.. code-block:: bash

    cd ~/ffrprep_tutorial

    docker run -ti --rm \
        -v $(pwd):/data \
        ksitek/ffrprep:latest \
        /data/bids_dataset \
        /data/bids_dataset/derivatives \
        participant \
        --stage analysis

**Using Singularity:**

.. code-block:: bash

    cd ~/ffrprep_tutorial

    singularity run --cleanenv \
        -B $(pwd):/data \
        ffrprep_latest.sif \
        /data/bids_dataset \
        /data/bids_dataset/derivatives \
        participant \
        --stage analysis

The analysis stage will:

- Load preprocessed epoched data
- Compute one evoked response per ``trial_type`` value
  (``_desc-evoked{Cond}.fif``), a combined evoked across all events
  (``_desc-evoked.fif``), and — for 2-trial-type datasets — an
  auto-paired difference evoked
  (``_desc-evokedDiff{A}Vs{B}.fif``). The trial-type split is on by
  default (``--split-by-trial-type``); pass
  ``--no-split-by-trial-type`` to fall back to a single combined
  evoked. For datasets with 3+ trial types, pass
  ``--difference-pairs A:B [C:D …]`` to opt in to explicit
  difference evokeds, or restrict the per-type outputs to a subset
  with ``--trial-types A B``.
- Generate time-frequency representations, autocorrelation, and
  pitch-tracking plots for each evoked.
- Compute FFR scalar metrics (RMS SNR, mean band-power) for each
  evoked.
- Compute the stim ↔ response cross-correlation (peak r + lag,
  plus an envelope correlation on the combined evoked) when the
  BIDS ``stim_file`` column is populated in ``events.tsv``. See
  Step 1 above for ``ffrprep-download example --with-stimuli``
  which fetches the stimulus files and augments ``events.tsv``.
- Compute per (task, run) phase consistency across polarities
  (two-trial-type datasets only) and a trial-to-trial response-
  consistency row on each per-condition Epoched section.
- Save results in standard formats.

Step 6: Inspect Analysis Outputs
================================

Examine the analysis results:

.. code-block:: bash

    # Navigate to analysis outputs
    cd ~/ffrprep_tutorial/bids_dataset/derivatives/ffrprep-analysis
    
    # List generated files
    find . -name "*.fif" -o -name "*.html" -o -name "*.json"
    
    # Open the analysis report
    open sub-*/sub-*_analysis_report.html

The analysis outputs include:

**Data files (per subject, per task / run):**

- ``*_desc-evoked{Cond}.fif``: per-trial-type evoked response data
  (e.g. ``_desc-evokedPositive.fif``, ``_desc-evokedNegative.fif``).
  Sidecar carries ``Condition`` and the standard ``AverageCount`` /
  ``Baseline`` / ``SamplingFrequency`` fields.
- ``*_desc-evoked.fif``: combined evoked across all events. Sidecar
  omits ``Condition`` (it isn't tied to a single trial type).
- ``*_desc-evokedDiff{A}Vs{B}.fif``: difference evoked for the
  ``A`` − ``B`` polarity pair (auto-emitted for 2-trial-type
  datasets; opt in via ``--difference-pairs`` for 3+). Sidecar
  carries ``DifferenceOf: [A, B]``.

**Report:**

- ``sub-XX_analysis_report.html``: single-file HTML report per subject.
  One section per (task, run) group, containing:

  - per-trial-type Evoked sections with waveform / PSD / TFR /
    autocorrelation / pitch-track figures, the FFR scalar metrics
    (RMS SNR, mean band-power), and a stim ↔ response
    cross-correlation row + lag plot (when ``stim_file`` is
    populated in ``events.tsv``);
  - a combined Evoked section with the same plots plus a second
    stim correlation row + plot for the **envelope** (``|hilbert
    (stim)|``);
  - a difference Evoked section with the same plots plus a single
    raw-waveform stim correlation row + plot;
  - a Phase Consistency section (two-trial-type datasets only):
    masked phase-coherence time–frequency heatmap across both
    polarities plus their sum and difference, using seaborn's
    ``flare_r`` colormap.

  The matched preprocessing report
  (``sub-XX_preprocessing_report.html``) additionally shows a
  ``Mean trial-to-trial r`` row on each Epoched section (when at
  least 10 epochs are present).

**Key analysis features to examine:**

- **Evoked waveform**: Look for the clear FFR response in the
  post-stimulus window.
- **Power spectral density**: Check for spectral peaks at the stimulus
  fundamental frequency and its harmonics.
- **Autocorrelation**: Periodic peaks at the stimulus period indicate
  good phase-locking.
- **RMS SNR**: Response RMS / baseline RMS over the 100–200 ms
  response window. Higher is better.
- **Stim ↔ response cross-correlation**: peak r near zero lag (or
  within typical FFR lag of ~7–14 ms after onset) indicates good
  stimulus tracking. The envelope correlation on the combined
  evoked is the natural metric for ENV-following responses.
- **Phase consistency**: bright cells in the masked plot mark
  (frequency, time) regions where the response phase is
  reproducible across trials.

Step 7: Working with Outputs in Python
======================================

After running ``ffrprep``, you can load and analyze the outputs directly in Python using MNE-Python. This allows for custom analyses, visualization, and integration with your existing analysis pipelines.

**Import required libraries and set up paths:**

First, import the necessary libraries and define the paths to your processed data.

.. code-block:: python

    import mne
    import json
    from pathlib import Path

**Define paths to the processed data:**

Set up the paths to access the derivatives from both preprocessing and analysis stages.

.. code-block:: python

    derivatives_path = Path.home() / "ffrprep_tutorial" / "bids_dataset" / "derivatives"
    analysis_path = derivatives_path / "ffrprep-analysis" / "sub-01"

**Load and visualize evoked responses:**

The evoked responses contain the averaged EEG data across trials, which is the core of FFR analysis.

.. code-block:: python

    evoked_files = list(analysis_path.glob("*_desc-evoked.fif"))
    evoked = mne.read_evokeds(evoked_files[0])

    # Plot the evoked response
    evoked[0].plot()

**Load and examine the BIDS sidecar:**

Each evoked ``.fif`` has a sibling JSON sidecar carrying provenance
and computed metadata (``AverageCount``, ``SamplingFrequency``,
``Tmin`` / ``Tmax``, ``Baseline``, and run/condition identifiers).

.. code-block:: python

    sidecar_files = list(analysis_path.glob("*_desc-evoked.json"))
    with open(sidecar_files[0], "r") as f:
        sidecar = json.load(f)

    print("Sidecar:", sidecar)

**Compute time-frequency yourself:**

The HTML report embeds a time-frequency representation across the
FFR band but does not save the TFR object to a separate file.
Recompute it from the loaded Evoked when you need it for further
analysis:

.. code-block:: python

    import numpy as np

    freqs = np.arange(70.0, 300.0, 2.0)
    tfr = mne.time_frequency.tfr_multitaper(
        evoked[0], freqs=freqs, n_cycles=freqs / 4.0,
        time_bandwidth=4.0, return_itc=False, verbose=False,
    )
    tfr.plot()

Complete Pipeline Example
=========================

For convenience, here's how to run both preprocessing and analysis in one command:

**Docker:**

.. code-block:: bash

    docker run -ti --rm \
        -v $(pwd):/data \
        ksitek/ffrprep:latest \
        /data/bids_dataset \
        /data/bids_dataset/derivatives \
        participant \
        --stage both \
        --high_pass 1.0 \
        --low_pass 40.0 \
        --ref_channels average \
        --n_procs 2

**Singularity:**

.. code-block:: bash

    singularity run --cleanenv \
        -B $(pwd):/data \
        ffrprep_latest.sif \
        /data/bids_dataset \
        /data/bids_dataset/derivatives \
        participant \
        --stage both \
        --high_pass 1.0 \
        --low_pass 40.0 \
        --ref_channels average \
        --n_procs 2

Troubleshooting
===============

**Common issues and solutions:**

1. **Permission errors with containers:**

   - Ensure your data directory has proper permissions
   - On Linux, you may need to add ``--user $(id -u):$(id -g)`` to Docker commands

2. **Memory issues:**

   - Reduce the number of parallel workers with ``--n_procs 1`` (each worker
     loads its own raw + epochs into memory; footprint scales linearly with N)
   - Process fewer subjects at once

3. **BIDS validation errors:**

   - Check that your dataset follows BIDS conventions
   - Use ``--skip_bids_validation`` if necessary (not recommended)

4. **No FFR found in data:**

   - Verify your stimulus timing and event codes
   - Check that the frequency range matches your stimulus
   - Ensure sufficient trial counts

Next Steps
==========

After completing this tutorial, you can:

- Process your own FFR datasets using the same workflow
- Modify preprocessing parameters for your specific experimental setup
- Use the generated outputs for further statistical analysis
- Integrate ``ffrprep`` into automated processing pipelines

For more advanced usage, see the :ref:`usage` documentation and `API reference <api_ref.html>`_.

