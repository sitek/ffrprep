.. _usage:

======
Usage
======

The general usage of ``package_name`` is ADD DESCRIPTION HERE.
The exact command to run ``package_name`` depends on the Installation method and user. Regarding the latter, ``package_name`` 
can either be used as a ``command line tool`` or directly within ``python``. Please refer to the `Tutorial <https://YourGitHubHandle.github.io/package_name/walkthrough>`_ for a more detailed walkthrough.

Here's a very conceptual example of running ``package_name`` via ``CLI``: ::

    package_name 
    package_name optional_arguments

and here from within ``python``: ::

    from package_name import package_name_function
    from package_name import package_name_function

    result = package_name_function(input)

    result = package_name_function(input, optional_arguments)

Below, we will focus on the ``CLI`` version. Thus, if you are interested in using ``package_name`` directly within ``python``,
please check the `Examples <https://YourGitHubHandle.github.io/package_name/auto_examples/index>`_.

Sub-section of Usage focusing on CLI
===========================================

Command-Line Arguments
======================
.. argparse::
  :ref: package_name.package_name_cli.get_parser
  :prog: package_name
  :nodefault:
  :nodefaultconst:

Example Call(s)
---------------

Below you'll find two examples calls that hopefully help
you to familiarize yourself with ``package_name`` and its options.

Example 1
~~~~~~~~~

.. code-block:: bash

    package_name \
    input
    optional_arguments

Here's what's in this call:

- The 1st positional argument is 
- The 2nd positional argument indicates that 


Example 2
~~~~~~~~~

.. code-block:: bash

    package_name \
    input
    optional_arguments
    optional_arguments

Here's what's in this call:

- The 1st positional argument is 
- The 2nd positional argument indicates that 
- The 3rd positional argument indicates that 


Support and communication
=========================

The documentation of this project is found here: https://YourGitHubHandle.github.io/package_name.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/YourGitHubHandle/package_name/issues.

If you have a problem or would like to ask a question about how to use ``package_name``,
please submit a question to `NeuroStars.org <http://neurostars.org/tags/package_name>`_ with an ``package_name`` tag.
NeuroStars.org is a platform similar to StackOverflow but dedicated to neuroinformatics.

All previous ``package_name`` questions are available here:
http://neurostars.org/tags/package_name/

Not running on a local machine? - Data transfer
===============================================

Please contact you local system administrator regarding
possible and favourable transfer options (e.g., `rsync <https://rsync.samba.org/>`_
or `FileZilla <https://filezilla-project.org/>`_).

A very comprehensive approach would be `Datalad
<http://www.datalad.org/>`_, which will handle data transfers with the
appropriate settings and commands.
Datalad also performs version control over your data.