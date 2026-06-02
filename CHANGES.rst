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
- New flag ``--with-stimuli`` on ``ffrprep-download example``
  (``BooleanOptionalAction``, default ``False``): additionally
  fetches the BIDS ``/stimuli/`` directory needed by stimulus-aware
  analyses (e.g. ``corr_stim_to_resp``) and augments every
  ``events.tsv`` with the matching ``stim_file`` column.

Preprocessing
-------------

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
  - ``compute_fft``: amplitude spectrum helper.

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
