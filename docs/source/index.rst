

.. .. image:: _static/ffrprep_logo.png
..    :align: center
..    :width: 400px

.. centered:: ffrprep: A BIDS-App for preprocessing and analysing FFR EEG data.


.. image:: https://img.shields.io/github/actions/workflow/status/sitek/ffrprep/.github%2Fworkflows%2Fformatting_tests.yml
        :target: https://github.com/sitek/ffrprep/actions/workflows/formatting_tests.yml

.. image:: https://img.shields.io/pypi/v/ffrprep.svg
        :target: https://pypi.python.org/pypi/ffrprep

.. image:: https://img.shields.io/docker/pulls/sitek/ffrprep
    :alt: Dockerpulls
    :target: https://cloud.docker.com/u/sitek/repository/docker/sitek/ffrprep

.. image:: https://img.shields.io/github/repo-size/sitek/ffrprep.svg
        :target: https://github.com/sitek/ffrprep.zip

.. image:: https://img.shields.io/github/issues/sitek/ffrprep.svg
        :target: https://github.com/sitek/ffrprep/issues

.. image:: https://img.shields.io/github/license/sitek/ffrprep.svg
        :target: https://github.com/sitek/ffrprep




|


Introduction
============

``ffrprep`` is a `BIDS-App <https://bids-apps.neuroimaging.io>`_
that preprocesses frequency-following response (FFR) EEG data and
emits per-trial-type, combined, and difference evoked responses
alongside FFR-specific scalar metrics and HTML
reports.

This documentation showcases the respective functionality and provides details concerning
its application and modules.
If you still have questions after going through what's provided here you can refer to
the :ref:`api_ref` or ask a question on `GitHub <https://github.com/sitek/ffrprep/issues>`_.

|

Quickstart
==========

Process two subjects (both preprocessing and analysis stages, four
parallel workers per subject) with the published Docker image:

.. code-block:: bash

    docker run --rm \
      -v /path/to/your/bids_dataset:/data:rw \
      sitek/ffrprep:latest \
      /data /data/derivatives participant \
      --participant_label 01 02 \
      --stage both \
      --n_procs 4

Outputs land under
``/path/to/your/bids_dataset/derivatives/ffrprep-preprocessing/``
and ``…/ffrprep-analysis/``, each carrying a
``sub-XX_<stage>_report.html`` per subject. See
:ref:`installation` for image pinning and a Singularity build, and
:ref:`usage` for the full list of CLI flags plus worked examples
(per-trial-type analysis, explicit difference pairs, cluster
invocations).

|

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: 🛠️ Installation
        :link: installation
        :link-type: doc

        Set up ``ffrprep`` via Docker, Singularity, or a local Python
        install with ``uv``.

    .. grid-item-card:: 🚀 Usage
        :link: usage
        :link-type: doc

        Command-line reference and ready-to-copy example invocations
        for typical FFR preprocessing + analysis runs.

    .. grid-item-card:: 📘 Tutorial walkthrough
        :link: walkthrough
        :link-type: doc

        End-to-end walkthrough: download the example dataset, run the
        pipeline, and inspect the generated reports.

    .. grid-item-card:: 🧠 Pipeline details
        :link: pipeline_details
        :link-type: doc

        Per-stage breakdown of the BIDS validation, preprocessing,
        and analysis workflows, including the file layout each
        stage produces.

|

Support & Contributing
======================

Bug reports, feature requests, and pull requests go through the
GitHub issue tracker. Usage questions are best asked on
`NeuroStars <https://neurostars.org/tags/ffrprep>`_ under the
``ffrprep`` tag, where the broader BIDS / neuroinformatics
community sees them and prior answers are searchable. If you build
something on top of ``ffrprep`` (custom downstream analyses,
additional report sections, extra QC metrics), the section
builders in :py:mod:`ffrprep.reports` accept ``extra_summary`` and
``extra_figures`` kwargs so you can fold caller-computed scalars
or figures into the existing per-section tables and galleries
without forking the pipeline.

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: 🐛 Report issues
        :class-card: sd-border-0
        :shadow: sm

        Bug reports, feature requests, and PRs welcome.

        +++
        `Open an issue on GitHub <https://github.com/sitek/ffrprep/issues>`_

    .. grid-item-card:: 💬 Get help
        :class-card: sd-border-0
        :shadow: sm

        Usage questions get the broadest audience on NeuroStars
        under the ``ffrprep`` tag.

        +++
        `Ask on NeuroStars <https://neurostars.org/tags/ffrprep>`_

    .. grid-item-card:: 📚 API reference
        :link: api_ref
        :link-type: doc
        :class-card: sd-border-0
        :shadow: sm

        Public functions of ``ffrprep.preproc``, ``ffrprep.analysis``,
        ``ffrprep.reports``, ``ffrprep.datasets``, and ``ffrprep.utils``.

        +++
        Module-by-module reference

    .. grid-item-card:: 🔄 Changelog
        :link: release-history
        :link-type: doc
        :class-card: sd-border-0
        :shadow: sm

        Per-version list of CLI flags, file-layout changes, and
        new helpers.

        +++
        Release history

|

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   usage
   pipeline_details
   walkthrough
   api_ref
   release-history
   min_versions
