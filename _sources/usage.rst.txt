.. _usage:

======
Usage
======

The general usage of ``ffrprep`` is ADD DESCRIPTION HERE.
The exact command to run ``ffrprep`` depends on the Installation method and user. Regarding the latter, ``ffrprep`` 
can either be used as a ``command line tool`` or directly within ``python``. Please refer to the `Tutorial <https://YourGitHubHandle.github.io/ffrprep/walkthrough>`_ for a more detailed walkthrough.

Here's a very conceptual example of running ``ffrprep`` via ``CLI``: ::

    ffrprep 
    ffrprep optional_arguments

and here from within ``python``: ::

    from ffrprep import ffrprep_function
    from ffrprep import ffrprep_function

    result = ffrprep_function(input)

    result = ffrprep_function(input, optional_arguments)

Below, we will focus on the ``CLI`` version. Thus, if you are interested in using ``ffrprep`` directly within ``python``,
please check the `Examples <https://YourGitHubHandle.github.io/ffrprep/auto_examples/index>`_.

Sub-section of Usage focusing on CLI
===========================================

Command-Line Arguments
======================
.. argparse::
  :ref: ffrprep.ffrprep_cli.get_parser
  :prog: ffrprep
  :nodefault:
  :nodefaultconst:

Example Call(s)
---------------

Below you'll find two examples calls that hopefully help
you to familiarize yourself with ``ffrprep`` and its options.

Example 1
~~~~~~~~~

.. code-block:: bash

    ffrprep \
    input
    optional_arguments

Here's what's in this call:

- The 1st positional argument is 
- The 2nd positional argument indicates that 


Example 2
~~~~~~~~~

.. code-block:: bash

    ffrprep \
    input
    optional_arguments
    optional_arguments

Here's what's in this call:

- The 1st positional argument is 
- The 2nd positional argument indicates that 
- The 3rd positional argument indicates that 


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