.. _pipeline_details:

================
Pipeline details
================

This section provides a comprehensive overview of the ``ffrprep`` processing pipeline, detailing each stage from BIDS validation through preprocessing to analysis. Understanding these steps will help you interpret results and customize processing for your specific research needs.

Pipeline Overview
=================

The ``ffrprep`` pipeline consists of three main stages:

1. **BIDS Validation** - Ensures dataset compliance with BIDS standards
2. **Preprocessing** - Filters, re-references, and epochs the EEG data  
3. **Analysis** - Computes evoked responses, time-frequency representations, and FFR metrics

Each stage is implemented as a modular workflow using Nipype, allowing for parallel processing and robust error handling.

.. image:: _static/pipeline_flowchart.png
   :alt: ffrprep pipeline flowchart
   :align: center

Stage 1: BIDS Validation
========================

The first stage validates that your input dataset follows the Brain Imaging Data Structure (BIDS) specification. This ensures reproducibility and compatibility with other neuroimaging tools.

**Purpose:**
Verify dataset structure, file naming conventions, and required metadata files before processing begins.

**Implementation:**
The validation uses the ``bids-validator`` tool with a custom configuration that ignores warnings not relevant to EEG/FFR data.

**Key Functions:**
- `validate_input_dir() <https://spark-csd.github.io/ffrprep/generated/ffrprep.utils.validate_input_dir.html#ffrprep.utils.validate_input_dir>`_ - Main validation function
- Custom validator configuration for EEG-specific requirements

**Validation Steps:**

1. **Directory Structure Check**
   
   - Verifies presence of required BIDS directories (``sub-*/``, ``derivatives/``)
   - Checks for ``dataset_description.json`` and other required metadata files
   - Validates subject/session/task naming conventions

2. **EEG-Specific Validation**
   
   - Confirms presence of EEG data files (``.edf``, ``.bdf``, ``.vhdr``, ``.fif``, ``.set``)
   - Validates channel description files (``*_channels.tsv``)
   - Checks event files (``*_events.tsv``) for proper formatting
   - Verifies EEG-specific metadata in JSON sidecars

3. **Participant Selection**
   
   - Validates requested participant labels exist in dataset
   - Checks for required EEG data for specified participants
   - Reports any missing or incomplete data

**Error Handling:**
If validation fails, ``ffrprep`` provides detailed error messages indicating specific BIDS compliance issues and suggestions for resolution.

**Skip Option:**
Validation can be bypassed using ``--skip_bids_validation`` (not recommended for production analyses).

Stage 2: Preprocessing
=====================

The preprocessing stage converts raw EEG data into clean, epoched data suitable for FFR analysis. This stage implements standard electrophysiological preprocessing steps optimized for frequency-following responses.

**Purpose:**
Transform raw continuous EEG into clean, filtered, and epoched data while preserving FFR-relevant neural signals.

**Implementation:**
Implemented as a Nipype workflow (`create_preprocessing_workflow() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.create_preprocessing_workflow.html#ffrprep.preproc.create_preprocessing_workflow>`_) with the following nodes:

Preprocessing Workflow Nodes
----------------------------

**1. Data Loading Node**

*Function:* `load_data() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.load_data.html#ffrprep.preproc.load_data>`_

*Purpose:* Robustly load EEG data from BIDS datasets using pybids and MNE-BIDS.

*Sub-steps:*
   - Create ``BIDSLayout`` object for dataset querying
   - Query for EEG files matching participant/session/task/run criteria
   - Try multiple file extensions (``.edf``, ``.bdf``, ``.vhdr``, ``.fif``, ``.set``)
   - Load data using ``mne_bids.read_raw_bids()``
   - Extract and validate channel information
   - Load associated event data and metadata

*Outputs:* Raw EEG data object, BIDS path information, original filename

**2. Re-referencing Node**

*Function:* `reference_data() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.reference_data.html#ffrprep.preproc.reference_data>`_

*Purpose:* Apply appropriate reference scheme to reduce common-mode noise and artifacts.

*Sub-steps:*
   - Parse reference channel specification (average, single channel, or channel list)
   - Validate reference channels exist in data
   - Apply re-referencing using MNE's ``set_eeg_reference()``
   - Update channel information and provenance

*Reference Options:*
   - **Average reference** (``--ref_channels average``): Uses all EEG channels
   - **Single channel** (``--ref_channels Cz``): References to one electrode
   - **Multiple channels** (``--ref_channels "Cz,Fz"``): Average of specified channels

*Outputs:* Re-referenced EEG data

**3. Filtering Node**

*Function:* `filter_data() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.filter_data.html#ffrprep.preproc.filter_data>`_

*Purpose:* Apply temporal filtering to remove noise while preserving FFR signals.

*Sub-steps:*
   - Apply high-pass filter to remove slow drifts and DC offsets
   - Apply low-pass filter to remove high-frequency noise
   - Use zero-phase FIR filters to avoid temporal distortions
   - Log filter parameters and transition bands

*Default Parameters:*
   - **High-pass:** 1.0 Hz (removes slow drifts, preserves FFR frequencies)
   - **Low-pass:** 40.0 Hz (removes EMG and high-frequency noise)
   - **Filter design:** Zero-phase FIR with automatic transition bandwidth

*Outputs:* Filtered EEG data

**4. Epoching Node**

*Function:* `create_epochs() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.create_epochs.html#ffrprep.preproc.create_epochs>`_

*Purpose:* Segment continuous data into time-locked epochs around stimulus events.

*Sub-steps:*
   - Load event information from BIDS events file
   - Identify relevant event codes/triggers
   - Extract epochs from ``tmin`` to ``tmax`` around events
   - Apply baseline correction
   - Reject epochs with excessive artifacts
   - Compute and log epoch statistics

*Default Parameters:*
   - **Epoch window:** -0.2 to 0.6 seconds around stimulus onset
   - **Baseline:** -0.2 to 0.0 seconds (pre-stimulus period)
   - **Rejection:** Automatic based on amplitude thresholds

*Artifact Rejection:*
   - Peak-to-peak amplitude thresholds per channel type
   - Automatic bad channel detection
   - Statistical outlier rejection
   - Manual inspection options

*Outputs:* Epoched EEG data, artifact rejection statistics

**5. Quality Control Node**

*Function:* `generate_preprocessing_report() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.generate_preprocessing_report.html#ffrprep.preproc.generate_preprocessing_report>`_

*Purpose:* Create comprehensive quality control reports for preprocessing steps.

*Sub-steps:*
   - Generate raw data quality plots (PSD, channel variance)
   - Visualize filter responses and effects
   - Plot epoch rejection statistics
   - Create event-related potential previews
   - Generate summary statistics and tables
   - Compile HTML report with interactive plots

*Report Contents:*
   - Data loading summary and file information
   - Channel locations and reference scheme
   - Filter responses and spectral effects  
   - Epoching statistics and rejection rates
   - Data quality metrics and recommendations

*Outputs:* HTML report, preprocessing statistics

**6. Save Preprocessing Node**

*Function:* `save_preprocessing() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.save_preprocessing.html#ffrprep.preproc.save_preprocessing>`_

*Purpose:* Save preprocessed data and metadata in BIDS-compatible format.

*Sub-steps:*
   - Create derivatives directory structure
   - Save epoched data in MNE format (``.fif`` files)
   - Generate BIDS-compatible metadata (JSON sidecars)
   - Create processing provenance records
   - Save quality control metrics

*Output Structure:* ::

    derivatives/ffrprep-preprocessing/
    ├── sub-XX/
    │   ├── sub-XX_task-YY_run-ZZ_epo.fif
    │   ├── sub-XX_task-YY_run-ZZ_epo.json
    │   ├── sub-XX_preprocessing_report.html
    │   └── sub-XX_preprocessing_metrics.json

*Outputs:* File paths, processing metadata

Stage 3: Analysis
================

The analysis stage computes evoked responses, time-frequency representations, and FFR-specific metrics from the preprocessed epoched data.

**Purpose:**
Extract meaningful neural measures that characterize the frequency-following response and provide quantitative metrics for statistical analysis.

**Implementation:**
Implemented as a Nipype workflow (`create_analysis_workflow() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.create_analysis_workflow.html#ffrprep.preproc.create_analysis_workflow>`_) with the following nodes:

Analysis Workflow Nodes
-----------------------

**1. Load Preprocessed Data Node**

*Function:* `load_epochs() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.load_epochs.html#ffrprep.preproc.load_epochs>`_

*Purpose:* Load preprocessed epoched data from the preprocessing stage.

*Sub-steps:*
   - Locate preprocessed epoch files in derivatives
   - Load epoched data using MNE
   - Validate data integrity and metadata
   - Extract processing parameters from provenance

*Error Handling:*
   - Checks for preprocessing completion
   - Validates file integrity
   - Reports missing or corrupted files

*Outputs:* Epoched EEG data, preprocessing metadata

**2. Evoked Response Node**

*Function:* `make_evoked() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.make_evoked.html#ffrprep.preproc.make_evoked>`_

*Purpose:* Compute evoked responses by averaging across trials.

*Sub-steps:*
   - Average epochs across trials to compute evoked responses
   - Handle different event types separately if ``--by_event_type`` is specified
   - Compute standard error and confidence intervals
   - Calculate signal-to-noise ratios
   - Generate evoked response statistics

*Event Type Handling:*
   - **Combined:** Average all epochs together (default)
   - **Separate:** Create separate evoked for each event type/condition
   - **Validation:** Ensure sufficient trials per condition

*Statistical Measures:*
   - Trial counts per condition
   - Signal-to-noise ratios
   - Standard error of the mean
   - Confidence intervals

*Outputs:* Evoked response objects, statistical metadata

**3. Time-Frequency Analysis Node**

*Function:* `compute_tfr() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.compute_tfr.html#ffrprep.preproc.compute_tfr>`_

*Purpose:* Compute time-frequency representations to analyze spectral dynamics.

*Sub-steps:*
   - Apply time-frequency decomposition (e.g., Morlet wavelets)
   - Compute power spectral density over time
   - Calculate phase-locking values
   - Generate time-frequency statistics
   - Apply baseline correction in frequency domain

*Methods:*
   - **Morlet wavelets:** Good time-frequency resolution
   - **Multitaper:** Better frequency resolution
   - **Stockwell transform:** Optimal for FFR analysis

*Frequency Bands:*
   - **FFR range:** Typically 80-1000 Hz for speech stimuli
   - **Custom ranges:** Configurable based on stimulus
   - **Harmonics:** Analysis of fundamental and harmonic frequencies

*Outputs:* Time-frequency power, phase-locking values, spectral statistics

**4. FFR Metrics Node**

*Function:* `compute_ffr_metrics() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.compute_ffr_metrics.html#ffrprep.preproc.compute_ffr_metrics>`_

*Purpose:* Calculate quantitative measures specific to frequency-following responses.

*Sub-steps:*
   - Compute stimulus-to-response correlations
   - Calculate phase-locking values at stimulus frequencies
   - Measure response amplitude and latency
   - Compute spectral harmonics analysis
   - Generate summary statistics

*FFR-Specific Metrics:*
   - **Response amplitude:** RMS amplitude in FFR frequency range
   - **Phase-locking:** Consistency of neural phase to stimulus
   - **Spectral correlations:** Stimulus-response spectral similarity
   - **Harmonic analysis:** Fundamental and harmonic component strength
   - **Onset/offset responses:** Transient response characteristics

*Statistical Measures:*
   - Confidence intervals for all metrics
   - Significance testing against noise floor
   - Effect size calculations
   - Multiple comparison corrections

*Outputs:* Quantitative FFR metrics, statistical summaries

**5. Analysis Report Node**

*Function:* `generate_analysis_report() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.generate_analysis_report.html#ffrprep.preproc.generate_analysis_report>`_

*Purpose:* Create comprehensive analysis reports with visualizations.

*Sub-steps:*
   - Generate evoked response plots (waveforms, topographies)
   - Create time-frequency visualizations
   - Plot FFR metrics and statistical summaries
   - Generate comparison plots across conditions
   - Compile interactive HTML report

*Visualizations:*
   - **Evoked waveforms:** Time-domain responses with confidence intervals
   - **Scalp topographies:** Spatial distribution of responses
   - **Time-frequency plots:** Spectrograms and phase-locking maps
   - **FFR metrics plots:** Quantitative measure summaries
   - **Statistical plots:** Significance testing results

*Report Features:*
   - Interactive plots with zooming/panning
   - Downloadable high-resolution figures
   - Statistical tables and summaries
   - Processing parameter documentation

*Outputs:* HTML analysis report, figure files

**6. Save Analysis Node**

*Function:* `save_analysis() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.save_analysis.html#ffrprep.preproc.save_analysis>`_

*Purpose:* Save analysis results in standard formats for further analysis.

*Sub-steps:*
   - Save evoked responses in MNE format
   - Export time-frequency data
   - Save FFR metrics as structured data
   - Create BIDS-compatible metadata
   - Generate analysis provenance

*Output Structure:* ::

    derivatives/ffrprep-analysis/
    ├── sub-XX/
    │   ├── sub-XX_task-YY_run-ZZ_evoked.fif
    │   ├── sub-XX_task-YY_run-ZZ_tfr.h5
    │   ├── sub-XX_task-YY_run-ZZ_metrics.json
    │   ├── sub-XX_analysis_report.html
    │   └── figures/
    │       ├── sub-XX_evoked.png
    │       ├── sub-XX_tfr.png
    │       └── sub-XX_topography.png

*File Formats:*
   - **MNE format (.fif):** For evoked responses (can be loaded in MNE-Python)
   - **HDF5 (.h5):** For time-frequency data (efficient storage)
   - **JSON:** For metrics and metadata (human-readable, machine-parseable)
   - **PNG/SVG:** For publication-ready figures

*Outputs:* Analysis file paths, processing metadata

Pipeline Integration and Quality Control
========================================

**Workflow Management:**
- Each stage implemented as Nipype workflow for reproducibility
- Automatic dependency tracking and parallel execution
- Robust error handling and logging
- Resumable processing after failures

**Quality Control Checkpoints:**
- BIDS validation before processing
- Data quality assessment after loading
- Preprocessing quality metrics and reports
- Analysis validation and statistical checks

**Output Organization:**
- BIDS-compatible directory structure
- Comprehensive metadata and provenance tracking
- Standardized file formats for interoperability
- Version-controlled processing parameters

**Customization Options:**
- Flexible parameter configuration
- Modular workflow components
- Plugin architecture for custom analyses
- Integration with external tools and pipelines

This pipeline design ensures robust, reproducible FFR analysis while maintaining flexibility for diverse research applications.

