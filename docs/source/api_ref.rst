.. _api_ref:

.. currentmodule:: ffrprep

Reference API
=============

.. contents:: **List of modules**
   :local:

.. _ref_preproc:

:mod:`ffrprep.preproc` - Preprocessing functions
------------------------------------------------
.. automodule:: ffrprep.preproc
   :no-members:
   :no-inherited-members:

.. currentmodule:: ffrprep.preproc

.. autosummary::
   :template: function.rst
   :toctree: generated/

   load_data
   reference_data
   filter_data
   epoch_data
   make_evoked
   make_combined_evoked
   make_difference_evokeds
   build_analysis_payload
   save_preprocessing_outputs
   save_analysis_outputs
   create_preprocessing_workflow
   create_analysis_workflow


.. _ref_analysis:

:mod:`ffrprep.analysis` - FFR analysis functions
------------------------------------------------
.. automodule:: ffrprep.analysis
   :no-members:
   :no-inherited-members:

.. currentmodule:: ffrprep.analysis

.. autosummary::
   :template: function.rst
   :toctree: generated/

   compute_power
   rms_snr
   autocorrelation
   compute_pitch_and_conf
   plot_pitch_and_conf
   compute_phase_consistency
   plot_phase_consistency
   plot_phase_consistency_masked
   corr_stim_to_resp
   corr_resp_to_resp
   response_consistency
   compute_fft


.. _ref_reports:

:mod:`ffrprep.reports` - Report builders
----------------------------------------
.. automodule:: ffrprep.reports
   :no-members:
   :no-inherited-members:

.. currentmodule:: ffrprep.reports

.. autosummary::
   :template: function.rst
   :toctree: generated/

   build_raw_section
   build_epoch_section
   build_evoked_section
   build_phase_consistency_section
   make_group
   build_subject_report
   build_analysis_report
   evoked_qa
   epoch_qa


.. _ref_dataset:

:mod:`ffrprep.datasets` - Dataset functions
-------------------------------------------
.. automodule:: ffrprep.datasets
   :no-members:
   :no-inherited-members:

.. currentmodule:: ffrprep.datasets

.. autosummary::
   :template: function.rst
   :toctree: generated/

   download_example_data
   download_raw_data
   download_epoch_data
   download_stimuli


.. _ref_utils:

:mod:`ffrprep.utils` - Utility functions
----------------------------------------
.. automodule:: ffrprep.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: ffrprep.utils

.. autosummary::
   :template: function.rst
   :toctree: generated/

   validate_input_dir
