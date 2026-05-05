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

Each stage is implemented as a modular Nipype workflow. The CLI parallelizes
across (task, run) iterations within a subject via a ``ProcessPoolExecutor``
sized by ``--n_procs``; for cross-subject scaling on clusters, run one CLI
invocation per subject (e.g. via slurm job arrays). See the
:ref:`Parallelization <parallelization>` section under *Usage* for details.



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
======================

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

*Function:* `epoch_data() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.epoch_data.html#ffrprep.preproc.epoch_data>`_

*Purpose:* Segment the continuously-filtered EEG into time-locked
epochs around stimulus events, applying baseline correction and
amplitude-based rejection.

*Sub-steps:*
   - Load events from the BIDS ``*_events.tsv`` (or use annotations
     embedded in the raw recording when no sidecar is present)
   - Build ``mne.Epochs`` with the requested ``tmin`` / ``tmax``,
     baseline window, picks, and ``reject`` thresholds
   - Apply ``epochs.drop_bad()`` to materialize amplitude-based
     rejection; keep the resulting Epochs object as the workflow
     output

*Default Parameters (FFR-typical, all overridable from the CLI):*
   - **Epoch window:** ``--tmin -0.04`` to ``--tmax 0.4`` seconds
     around stimulus onset
   - **Baseline:** ``--baseline -0.04 0`` seconds (pre-stimulus)
   - **Rejection:** ``--reject-eeg 7.5e-5`` (75 µV peak-to-peak); pass
     ``--no-auto-reject`` to disable

*Outputs:* Epoched EEG data and the post-rejection drop log.

**5. Save Preprocessing Outputs Node**

*Function:* `save_preprocessing_outputs() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.save_preprocessing_outputs.html#ffrprep.preproc.save_preprocessing_outputs>`_

*Purpose:* Persist the epoched data plus a self-describing BIDS
sidecar.

*Sub-steps:*
   - Reconstruct a fresh ``EpochsArray`` from the input data (avoids
     edge cases where the upstream Epochs object carries internal
     state that doesn't round-trip through ``.save()``)
   - Write ``_desc-preproc_epo.fif``
   - Write the sibling ``_desc-preproc_epo.json`` sidecar with
     ``EpochCount`` / ``EpochCountTotal`` / ``EpochCountRejected`` /
     ``RejectionThresholds`` / ``Filtering`` / ``SamplingFrequency``
     / ``EpochTmin`` / ``EpochTmax`` / ``Channels`` plus run /
     session / ``ConcatenatedRuns`` provenance
   - Initialize the per-derivatives ``dataset_description.json`` if
     missing

*Reporting* runs **after** the workflow drains, in the CLI rather
than as a workflow node — see :ref:`reporting <reporting>` below.

*Output Structure:* ::

    derivatives/ffrprep-preprocessing/
    ├── sub-XX/
    │   └── eeg/
    │       ├── sub-XX_task-YY_run-ZZ_desc-preproc_epo.fif
    │       ├── sub-XX_task-YY_run-ZZ_desc-preproc_epo.json
    │       ├── sub-XX_preprocessing_report.html
    │       └── sub-XX_preprocessing.log

The sidecar JSON carries provenance, ``EpochCount`` /
``EpochCountTotal`` / ``EpochCountRejected``, ``RejectionThresholds``,
``Filtering`` (high-pass and low-pass cut-offs), sampling frequency,
and run / session identifiers.

*Outputs:* File paths, processing metadata

Stage 3: Analysis
=================

The analysis stage averages each preprocessed epochs file into an
evoked response and saves it to BIDS-derivatives.

**Purpose:**
Produce per-(task, run) evoked responses suitable for downstream
statistical analysis or visualization, persisted in MNE-readable
format with a self-describing BIDS sidecar.

**Implementation:**
Implemented as a Nipype workflow (`create_analysis_workflow() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.create_analysis_workflow.html#ffrprep.preproc.create_analysis_workflow>`_)
with two nodes. The CLI worker loads the saved
``_desc-preproc_epo.fif`` from the previous stage and passes the
``Epochs`` object directly into the workflow's ``inputnode`` —
loading is not itself a workflow node.

Analysis Workflow Nodes
-----------------------

**1. Evoked Response Node**

*Function:* `make_evoked() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.make_evoked.html#ffrprep.preproc.make_evoked>`_

*Purpose:* Average epochs into one or more evoked responses.

*Sub-steps:*
   - When ``--by_event_type`` is set, partition the input Epochs by
     ``event_id`` and average each subset into its own Evoked
   - Otherwise, average all epochs into a single Evoked

*Outputs:* An ``mne.Evoked`` object (or a dict of them, keyed by
event name when ``--by_event_type`` is on).

**2. Save Analysis Node**

*Function:* `save_analysis_outputs() <https://spark-csd.github.io/ffrprep/generated/ffrprep.preproc.save_analysis_outputs.html#ffrprep.preproc.save_analysis_outputs>`_

*Purpose:* Persist the evoked response and a self-describing sidecar.

*Sub-steps:*
   - Write ``_desc-evoked.fif`` (or one ``_desc-evoked<Condition>.fif``
     per condition under ``--by_event_type``)
   - Write the sibling ``.json`` sidecar with ``AverageCount`` /
     ``Baseline`` / ``SamplingFrequency`` / ``Tmin`` / ``Tmax`` /
     ``Channels`` / ``TaskName`` / ``AnalysisType`` plus run /
     session / ``ConcatenatedRuns`` / ``Condition`` provenance
   - Initialize the per-derivatives ``dataset_description.json`` if
     missing

*Output Structure:* ::

    derivatives/ffrprep-analysis/
    ├── sub-XX/
    │   ├── sub-XX_task-YY_run-ZZ_desc-evoked.fif
    │   ├── sub-XX_task-YY_run-ZZ_desc-evoked.json
    │   ├── sub-XX_analysis_report.html
    │   └── sub-XX_analysis.log

*Outputs:* Saved file paths.

.. _reporting:

Reporting (post-workflow)
-------------------------

The single-file HTML report is built **in the CLI** after the
workflow drains, not as a workflow node. The CLI's
``_build_preproc_report`` and ``_build_analysis_report`` glob the
saved ``_desc-preproc_epo.fif`` / ``_desc-evoked.fif`` files,
build per-(task, run) sections via the
:py:mod:`ffrprep.reports` builders
(``build_raw_section`` / ``build_epoch_section`` /
``build_evoked_section`` / ``make_group``), and render via
``build_subject_report`` / ``build_analysis_report``.

The report's analysis figures (time-frequency representation,
autocorrelation, pitch tracking) and scalar metrics (RMS SNR,
mean band-power) are **computed at report time** by the
:py:func:`ffrprep.reports.evoked_qa` helper from the loaded Evoked
— they are not separately persisted to disk. To recompute them
yourself, see the *Working with Outputs in Python* section of the
:ref:`walkthrough`.

The single-file ``*_report.html`` embeds all figures as inline
base64 PNGs — there is no sibling ``figures/`` directory.

*File Formats:*
   - **MNE format (.fif):** Evoked response, loadable in MNE-Python
   - **JSON:** BIDS sidecar (human-readable, machine-parseable)
   - **HTML:** Self-contained single-file per-subject report

Pipeline Integration and Quality Control
========================================

**Workflow Management:**
- Each stage implemented as a Nipype workflow for per-iteration
  dependency tracking
- Outer-loop parallelism: the CLI dispatches per-(task, run)
  iterations to a ``ProcessPoolExecutor`` sized by ``--n_procs``
- Fail-fast on any iteration error; the exception propagates to the
  CLI entry point
- nipype caches per-iteration intermediates under ``work/`` so
  re-runs that already have a saved ``_desc-preproc_epo.fif`` skip
  the workflow re-execution

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

