=======
Changes
=======

Unreleased
==========

CLI
---

- New flag ``--split-by-trial-type`` (``BooleanOptionalAction``,
  default ``True``) on ``ffrprep``: emit per-trial-type epoched and
  evoked files alongside the combined output. Pass
  ``--no-split-by-trial-type`` for the previous single-combined
  behaviour. The legacy ``--by_event_type`` flag is kept as a
  SUPPRESS'd alias for one release.
- New flag ``--trial-types A B …`` to restrict per-trial-type outputs
  to a subset.
- New flag ``--difference-pairs A:B [C:D …]`` to compute difference
  evokeds across explicit pairs. For 2-type datasets the difference
  is auto-emitted; for 3+ types this flag is required to opt in.
- New flag ``--filter-method {fir,iir}`` (default ``fir``): ``iir`` applies
  zero-phase (forward-backward) first-order Butterworth high-pass and
  low-pass filters (12 dB/octave overall), matching the ``butter`` /
  ``filtfilt`` band-pass used in some published FFR pipelines.
- New flag ``--reject-mode {ptp,abs}`` (default ``ptp``): ``abs`` drops
  epochs whose absolute amplitude reaches ``--reject-eeg`` at any sample
  (``max|x| >= threshold``, drop reason ``ABS_AMP``) instead of MNE's
  peak-to-peak criterion. The preprocessing sidecar records
  ``RejectionMode`` (``peak-to-peak`` / ``absolute-amplitude``) next to
  ``RejectionThresholds``.
- Fixed: ``--reject-eeg 0`` now disables automatic rejection as its help
  text documents (``parse_reject``). It previously built a ``{"eeg": 0.0}``
  threshold, which rejects every epoch. Negative thresholds are now an
  argument error.
- New flag ``--no-report``: skip HTML report generation (preprocessing
  and analysis) while still writing all derivatives. Intended for bulk
  runs over many subjects, where report figures dominate runtime.
- New flags for resumable, disk-friendly bulk runs: ``--skip-existing``
  (skip (task, run) iterations whose sidecar outputs already exist),
  ``--clean-work-dir`` (delete each iteration's Nipype working files after
  it succeeds; logs are kept) and ``--keep-epochs`` /
  ``--no-keep-epochs`` (default keep; ``--no-keep-epochs`` deletes the
  ``*_epo.fif`` files after analysis and keeps the JSON sidecars).
- New flag ``--with-stimuli`` on ``ffrprep-download example``
  (``BooleanOptionalAction``, default ``False``): additionally
  fetches the BIDS ``/stimuli/`` directory needed by stimulus-aware
  analyses (e.g. ``corr_stim_to_resp``) and augments every
  ``events.tsv`` with the matching ``stim_file`` column.

Group-level analysis
--------------------

- ``ffrprep <bids_dir> <output_dir> group`` now runs, replacing the
  previous "Currently only participant-level analysis is supported."
  stub. New module ``ffrprep.group`` aggregates already-computed
  participant-level derivatives from ``output_dir`` — it does not
  read ``bids_dir`` or re-run preprocessing/analysis:

  - ``discover_group_inputs``: globs ``ffrprep-analysis/sub-*/`` and
    ``ffrprep-preprocessing/sub-*/eeg/`` for combined evoked, diff
    evoked, and preprocessing epochs files, grouped by
    ``(task, session, run)``.
  - ``compute_grand_average``: ``mne.grand_average`` across subjects
    for each (task, session, run) with >= 2 contributing subjects.
  - ``compute_subject_metrics``: recomputes RMS SNR, band power, and
    response consistency per subject directly from saved derivatives
    (nothing is recomputed from raw data).
  - Outputs land under ``output_dir/ffrprep-group/``: grand-average
    ``_desc-grandAverage_ave.fif`` (+ JSON sidecar with contributing
    subjects), a per-(task, run) ``_metrics.tsv``, and a single
    ``group_report.html``.
  - Deliberately aggregation-only, not inferential: no group-level
    statistics are computed, matching the scope other BIDS Apps
    (e.g. MRIQC) use for their own "group" level.
- Group-level FFR metrics (opt-in): ``--f0`` adds ``rms_snr_polarity_sum``,
  ``f0_uv`` and ``upper_harmonics_uv`` (band-averaged FFT amplitude at the
  harmonics of F0, computed on the sum of the two per-trial-type
  averages; see ``--n-harmonics``, ``--harmonic-bin-hz``,
  ``--harmonic-window``); ``--stimulus`` adds ``stim2resp_r/z/lag_ms``
  (and ``stim2resp_lim_*`` with ``--xcorr-lag-range``); ``--n-trials-presented``
  adds ``usable_pct``. ``discover_group_inputs`` now also returns the
  per-trial-type evoked files (``"by_type"``).
- QC flags on the group metrics table: ``--min-usable-pct`` (needs
  ``--n-trials-presented``) and ``--min-snr`` add ``qc_usable_pct_ok`` /
  ``qc_snr_ok``, ``qc_include`` and ``qc_reason`` columns (``add_qc_flags``).
  Subjects are flagged, never dropped: the table and the grand averages keep
  every subject, and the report summary shows how many were flagged.
- The group metrics TSV now has a BIDS-style data dictionary,
  ``task-<task>[_run-<run>]_metrics.json``, beside it (``build_metrics_dictionary``):
  a description and units for every column, with the actual windows and
  thresholds of the run embedded. Covariate columns take their description,
  levels and units from the JSON sidecar next to the covariate TSV
  (``participants.json``, ``phenotype/*.json``) when one exists.
  ``merge_covariates`` gains ``return_sources`` and ``save_group_metrics``
  a ``dictionary`` argument.
- ``compute_grand_average`` renames single-channel evokeds that carry different
  channel names across sites (e.g. ``A32`` vs ``Cz``) to a common name before
  averaging; mismatched multi-channel sets raise a clear ``ValueError``.
- The group metrics TSV joins subject-level covariates from
  ``<bids_dir>/participants.tsv`` and ``--covariates`` (``merge_covariates``);
  the report section switches to per-metric histograms for cohorts above
  40 subjects (the per-subject values stay in the TSV).
- ``ffrprep.reports`` gains ``build_group_report`` and
  ``build_metrics_table_section``; ``build_subject_report`` /
  ``build_analysis_report`` pass new ``entity_label`` / ``meta_label``
  template variables so the shared ``subject_report.html.j2`` template
  can render a report with no single subject (existing rendered output
  for per-subject reports is unchanged).
- ``setup_derivatives_directories`` gains a ``create_group=False``
  flag to materialize ``ffrprep-group/``.

Preprocessing
-------------

- Per-trial-type filenames now use BIDS-valid alphanumeric labels:
  ``_bids_label`` drops non-alphanumeric separators and capitalizes
  each token, so a ``da_pol-1`` trial type is written as
  ``_desc-preprocDaPol1_epo.fif`` / ``_desc-evokedDaPol1.fif`` /
  ``_desc-evokedDiffDaPol1VsDaPol2.fif`` instead of leaking ``_`` and
  ``-`` into the ``desc`` entity. Alphanumeric labels (``positive``,
  ``10``) are unchanged, and sidecars still record the raw trial-type
  name in ``Condition`` / ``DifferenceOf``.
- Fixed: the preprocessing sidecar's ``Sources`` / ``RawSources`` now name the
  raw recording that actually exists (EDF, BDF, BrainVision, EEGLAB or FIF,
  including the session directory and every run of a concatenated output).
  They were hard-coded to ``_eeg.bdf``; when no raw file is found the fields
  are omitted instead of recording a path that does not exist.
- Fixed: the preprocessing sidecar's ``EpochCountTotal`` / ``EpochCountRejected``
  are now per trial type. They were derived from ``len(epochs.drop_log)``,
  which spans every event (the other condition's trials and non-analysed
  markers included), so a two-polarity recording reported ~2x the trials
  and charged each rejected trial to every file. ``epoch_data`` now records
  per-condition counts while the events array is still aligned with the
  drop log; epochs built elsewhere fall back to the previous behaviour.
- ``epoch_data`` accepts ``trial_types=`` to narrow the discovered
  event_id mapping to a subset; raises ``ValueError`` if a
  requested name is absent so typos surface immediately.
- ``save_preprocessing_outputs`` accepts a ``dict[str, mne.Epochs]``
  in addition to a scalar Epochs; dict input writes
  ``_desc-preproc{Cond}_epo.fif`` per trial type and the per-file
  sidecar carries a ``Condition`` field. Scalar input keeps today's
  ``_desc-preproc_epo.fif`` filename + single ``Path`` return.
- ``save_preprocessing_node`` wraps the split: ``split_by_trial_type=True``
  fans the input Epochs out into a ``{cond: epochs[cond]}`` dict
  before forwarding to ``save_preprocessing_outputs``.
- ``_save_one_preproc_epochs`` now passes ``events`` + ``event_id``
  through to the ``EpochsArray`` reconstruction so trial-type
  metadata survives the save / load round-trip (previously stripped
  silently; broke the per-condition flow).

Analysis
--------

- Fixed: ``plot_pitch_and_conf`` no longer calls ``plt.show()``. On any
  local (non-Docker) install where matplotlib defaults to an
  interactive backend (e.g. ``macosx`` on a Mac, ``TkAgg``/``QtAgg`` on
  many Linux desktops), this call blocked ``--stage analysis`` /
  ``--stage both`` runs indefinitely waiting for a GUI window to
  close, since ``evoked_qa`` invokes it during every analysis report
  build. CI/Docker never hit this because tests force ``Agg`` and the
  container images run headless. The figure was already captured via
  ``plt.gcf()`` immediately after the call (see ``reports.evoked_qa``),
  so nothing depended on the interactive display.
- ``save_analysis_outputs`` accepts a structured payload
  ``{"by_type": dict, "combined": Evoked, "diff": dict}`` (any
  subset). Filenames:

  - per-type: ``_desc-evoked{Cond}.fif``
  - combined: ``_desc-evoked.fif``
  - difference: ``_desc-evokedDiff{A}Vs{B}.fif``

  Sidecars carry ``Condition`` (per-type) or ``DifferenceOf: [A, B]``
  (diff); combined omits both. Scalar Evoked input is treated as the
  combined output.

- ``create_analysis_workflow`` wires a new ``build_analysis_payload``
  helper that produces the structured payload from a single Epochs
  object; the workflow inputnode gains ``difference_pairs``.
- New helpers in ``ffrprep.analysis`` (from PR #35 + post-merge
  refactor):

  - ``compute_phase_consistency`` / ``plot_phase_consistency`` /
    ``plot_phase_consistency_masked``: phase-consistency across two
    polarities + sum / difference.
  - ``corr_stim_to_resp`` / ``corr_resp_to_resp``: cross-correlation
    peak r + lag. Now delegate to a shared ``_xcorr_normalized``
    helper that also returns the full correlation curve for
    plotting.
  - ``response_consistency``: mean pairwise Pearson correlation
    across epochs.
    Computed with a single ``np.corrcoef`` matrix instead of a
    Python loop of ``scipy.stats.pearsonr`` calls (same values and
    pair ordering; ~0.3 s vs ~9 min for 3000 epochs x 4147 samples).
  - ``compute_fft``: amplitude spectrum helper.
  - ``harmonic_amplitudes``: FFT amplitude (zero-padded, single-sided,
    microvolts) of a response window, averaged in a ``bin_hz``-wide
    band around each harmonic of ``f0``; returns the per-harmonic values,
    the fundamental, and the summed upper harmonics. Defaults follow the
    /da/ measure of Whiteford et al. (2025) (60-180 ms, 100 Hz, 10
    harmonics, 60 Hz bins).
  - ``stim_to_resp_xcorr``: maximum stimulus-to-response correlation in
    MATLAB ``xcorr(..., 'coeff')`` form (no mean removal) with an
    optional lag window, returning ``(r, Fisher z, lag_ms)``.
  - ``load_wav_mono`` / ``resample_signal``: WAV reader and polyphase
    resampler used to bring a stimulus to the EEG sampling rate.

- Analysis worker granularity changed from per-file to
  per-(task, run) group. ``_collect_analysis_groups`` stitches
  per-condition preproc files via ``mne.concatenate_epochs`` (with
  ``event_id`` preserved) and the workflow runs once per group.

Reporting
---------

- Per-(task, run) groups: each (task, run) gets one Raw section, one
  Epoched section per trial type, and per-type / combined / diff
  Evoked sections under one heading.
- New summary rows on per-condition Evoked sections:

  - ``Stim correlation (peak r)`` + ``Stim correlation (lag, ms)``,
    derived from the BIDS ``stim_file`` column in ``events.tsv``
    (silent no-op when the column or file is missing).

- New figure on per-condition Evoked sections: stim ↔ response
  cross-correlation curve with the peak marked.
- Combined Evoked sections add a second pair of rows / figure for
  the **envelope** correlation (combined ≈ ENV proxy in FFR, so
  ``|hilbert(stim)|`` is the natural reference).
- Difference Evoked sections add a single raw-waveform stim
  correlation row + figure (diff ≈ TFS proxy).
- Epoched sections (≥ 10 trials) gain a ``Mean trial-to-trial r``
  row via ``response_consistency``.
- New Phase Consistency section per (task, run) when exactly two
  per-condition preproc files exist. Uses seaborn's ``flare_r``
  colormap; subplot titles surface the trial-type names. Default in
  single-subject reports is **unmasked**; group-level callers can
  pass ``mask=True`` for the significance-masked variant.

Datasets
--------

- New ``download_stimuli(dataset_path=None)`` fetches the OSF
  stimulus files into ``<dataset>/ffrprep_raw_data/stimuli/`` per
  the BIDS spec, then augments every ``sub-*/eeg/*_events.tsv``
  with a ``stim_file`` column based on a ``trial_type ->
  filename`` lookup (``STIM_FILE_MAP``).
- Existing ``download_example_data`` / ``download_raw_data`` accept
  a ``with_stimuli=False`` kwarg; True triggers the stimulus
  download after the EEG data.

Dependencies
------------

- New runtime dependency: ``seaborn>=0.13`` (used solely for its
  ``flare_r`` colormap registration).

Internal
--------

- Build / run-provenance helpers updated for the per-condition
  filenames: ``_check_preproc_output_or_raise`` glob widened to
  ``*_desc-preproc*_epo.fif``; ``_propagate_run_provenance`` derives
  the analysis base-stem via ``str.partition("_desc-preproc")``.
- ``build_analysis_payload`` imports its dependencies via
  ``importlib.import_module`` so the Nipype Function-node subprocess
  resolves them without inheriting the parent's globals.
- ``test_system_pipeline`` helper now points ``_derivatives_root``
  at the explicit ``output_dir`` argument (the BIDS-App
  ``--output-dir`` positional is now honoured end-to-end).
- Production Docker image installs runtime deps only
  (``uv sync --frozen --no-dev``); the in-image ``test``
  subcommand needs a separate ``--with-tests`` rebuild.
