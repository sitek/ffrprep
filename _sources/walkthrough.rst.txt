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

First, let's download some example FFR data to work with. ``ffrprep`` provides convenient functions to download BIDS-formatted example datasets:

.. code-block:: python

    from ffrprep.datasets import download_example_data
    import os
    
    # Create a working directory
    work_dir = os.path.expanduser("~/ffrprep_tutorial")
    os.makedirs(work_dir, exist_ok=True)
    
    # Download example data (1 subject)
    dataset_path = download_example_data(dataset_path=work_dir)
    print(f"Example data downloaded to: {dataset_path}")

This will download a complete BIDS dataset with:

- Raw EEG data from one subject
- Proper BIDS directory structure
- Required metadata files (``dataset_description.json``, etc.)
- Event files and channel information

Step 2: Get the ffrprep Container
=================================

Next, obtain the ``ffrprep`` container image. Choose either Docker or Singularity:

**Option A: Docker**

.. code-block:: bash

    # Pull the latest ffrprep Docker image
    docker pull ffrprep/ffrprep:latest
    
    # Verify the image was downloaded
    docker images | grep ffrprep

**Option B: Singularity**

.. code-block:: bash

    # Build Singularity image from Docker Hub
    singularity build ffrprep_latest.sif docker://ffrprep/ffrprep:latest
    
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
        ffrprep/ffrprep:latest \
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
    find . -name "*.html" -o -name "*.h5"
    
    # Open the preprocessing report in your browser
    open sub-*/sub-*_preprocessing_report.html  # macOS
    # or
    xdg-open sub-*/sub-*_preprocessing_report.html  # Linux

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
        ffrprep/ffrprep:latest \
        /data/bids_dataset \
        /data/bids_dataset/derivatives \
        participant \
        --stage analysis \
        --by_event_type

**Using Singularity:**

.. code-block:: bash

    cd ~/ffrprep_tutorial
    
    singularity run --cleanenv \
        -B $(pwd):/data \
        ffrprep_latest.sif \
        /data/bids_dataset \
        /data/bids_dataset/derivatives \
        participant \
        --stage analysis \
        --by_event_type

The analysis stage will:

- Load preprocessed epoched data
- Compute evoked responses (average across trials)
- Create separate evoked responses for each event type (``--by_event_type``)
- Generate time-frequency representations  
- Compute FFR-specific metrics
- Save results in standard formats

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

**Data files:**

- ``*_evoked.fif``: Evoked response data (can be loaded with MNE-Python)   
- ``*_tfr.h5``: Time-frequency representations  
- ``*_metrics.json``: Quantitative FFR metrics  

**Reports:**

Interactive HTML reports with:

- Evoked response waveforms
- Time-frequency plots
- Scalp topographies
- FFR metrics summary

**Key analysis features to examine:**

- **Evoked waveforms**: Look for clear FFR responses
- **Time-frequency**: Check for sustained oscillatory activity
- **Topographies**: Verify expected scalp distributions
- **Metrics**: Review quantitative measures (amplitude, phase-locking, etc.)

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

    derivatives_path = Path("~/ffrprep_tutorial/bids_dataset/derivatives")
    analysis_path = derivatives_path / "ffrprep-analysis" / "sub-01"

**Load and visualize evoked responses:**

The evoked responses contain the averaged EEG data across trials, which is the core of FFR analysis.

.. code-block:: python

    evoked_files = list(analysis_path.glob("*_evoked.fif"))
    evoked = mne.read_evokeds(evoked_files[0])
    
    # Plot the evoked response
    evoked[0].plot()

**Load and examine quantitative FFR metrics:**

``ffrprep`` computes various quantitative metrics that characterize the FFR response, such as amplitude, phase-locking values, and spectral properties.

.. code-block:: python

    metrics_file = list(analysis_path.glob("*_metrics.json"))[0]
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    print("FFR Metrics:", metrics)

**Load and visualize time-frequency representations:**

Time-frequency analysis shows how spectral power changes over time, which is particularly important for understanding FFR dynamics.

.. code-block:: python

    tfr_files = list(analysis_path.glob("*_tfr.h5"))
    if tfr_files:
        tfr = mne.time_frequency.read_tfrs(tfr_files[0])
        tfr[0].plot()

Complete Pipeline Example
=========================

For convenience, here's how to run both preprocessing and analysis in one command:

**Docker:**

.. code-block:: bash

    docker run -ti --rm \
        -v $(pwd):/data \
        ffrprep/ffrprep:latest \
        /data/bids_dataset \
        /data/bids_dataset/derivatives \
        participant \
        --stage both \
        --high_pass 1.0 \
        --low_pass 40.0 \
        --ref_channels average \
        --by_event_type \
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
        --by_event_type \
        --n_procs 2

Troubleshooting
===============

**Common issues and solutions:**

1. **Permission errors with containers:**
   - Ensure your data directory has proper permissions
   - On Linux, you may need to add ``--user $(id -u):$(id -g)`` to Docker commands

2. **Memory issues:**
   - Reduce the number of parallel processes with ``--n_procs 1``
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

