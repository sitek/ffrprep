============
Installation
============

In general, there are two distinct ways to install and use ``package_name``:
either through virtualization/container technology, that is `Docker`_ or
`Singularity`_, or in a `Bare metal version (Python 3.10+)`_.
Once you are ready to run ``package_name``, see `Usage <https://YourGitHubHandle.github.io/package_name/usage>`_ for details.

Docker
======

In order to run ```package_name``` in a Docker container, Docker must be `installed
<https://docs.docker.com/engine/installation/>`_ on your system.
Once Docker is installed, you can get ``package_name`` through  running one of the following
commands in the terminal of your choice.

Option 1: pulling from the `dockerhub registry <https://hub.docker.com/repository/docker/YourGitHubHandle/package_name/general>`_ :


.. code-block:: bash

    docker pull YourGitHubHandle/package_name:version

Option 2: pulling from the `github container registry <https://github.com/YourGitHubHandle/package_name/pkgs/container/package_name>`_ :

.. code-block:: bash

    docker pull ghcr.io/YourGitHubHandle/package_name:version

Where ``version`` is the specific version of ``package_name`` you would like to use. For example, if you want 
to employ the ``latest``/most up to date ``version`` you can either run 

.. code-block:: bash

    docker pull YourGitHubHandle/package_name:latest

.. code-block:: bash

    docker pull ghcr.io/YourGitHubHandle/package_name:latest

or the same command withouth the ``:latest`` tag, as ``Docker`` searches for the ``latest`` tag by default.
However, as the ``latest`` version is subject to changes and not necessarily in synch with the most recent ``numbered version``, it 
is recommend to utilize the latter to ensure reproducibility. For example, if you want to employ ``package_name v0.0.1`` the command would look as follows:

.. code-block:: bash

    docker pull YourGitHubHandle/package_name:v0.0.1

.. code-block:: bash

    docker pull ghcr.io/YourGitHubHandle/package_name:v0.0.1

After the command finished (it may take a while depending on your internet connection),
you can run ``package_name`` like this:

.. code-block:: bash

    $ docker run -ti --rm \
        YourGitHubHandle/package_name:latest \
        

Please have a look at the examples under `Usage <https://YourGitHubHandle.github.io/package_name/usage>`_ to get more information
about and familiarize yourself with ``package_name``'s functionality.

Singularity
===========

For security reasons, many HPCs do not allow Docker containers, but support/allow `Singularity <https://github.com/singularityware/singularity>`_ containers. Depending
on the ``Singularity`` version available to you, there are two options to get ``package_name`` as
a ``Singularity image``.

Preparing a Singularity image (Singularity version >= 2.5)
----------------------------------------------------------
If the version of Singularity on your HPC is modern enough you can create a ``Singularity
image`` directly on the HCP.
This is as simple as: 

.. code-block:: bash

    $ singularity build /my_images/package_name-<version>.simg docker://YourGitHubHandle/package_name:<version>

Where ``<version>`` should be replaced with the desired version of ``package_name`` that you want to download.
For example, if you want to use ``package_name v0.0.4``, the command would look as follows.

.. code-block:: bash

    $ singularity build /my_images/package_name-v0.0.4.simg docker://YourGitHubHandle/package_name:v0.0.4


Preparing a Singularity image (Singularity version < 2.5)
---------------------------------------------------------
In this case, start with a machine (e.g., your personal computer) with ``Docker`` installed and
the use `docker2singularity <https://github.com/singularityware/docker2singularity>`_ to
create a ``Singularity image``. You will need an active internet connection and some time. 

.. code-block:: bash

    $ docker run --privileged -t --rm \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /absolute/path/to/output/folder:/output \
        singularityware/docker2singularity \
        YourGitHubHandle/package_name:<version>

Where ``<version>`` should be replaced with the desired version of ```package_name``` that you want
to download and ``/absolute/path/to/output/folder`` with the absolute path where the created ``Singularity image``
should be stored. Sticking with the example of ``package_name v0.0.4`` this would look as follows:

.. code-block:: bash

    $ docker run --privileged -t --rm \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /absolute/path/to/output/folder:/output \
        singularityware/docker2singularity \
        YourGitHubHandle/package_name:v0.0.4

Beware of the back slashes, expected for Windows systems. The above command would translate to Windows systems as follows:

.. code-block:: bash

    $ docker run --privileged -t --rm \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v D:\host\path\where\to\output\singularity\image:/output \
        singularityware/docker2singularity \
        YourGitHubHandle/package_name:<version>


You can then transfer the resulting ``Singularity image`` to the HPC, for example, using ``scp``. ::

    $ scp YourGitHubHandle_package_name<version>.simg <user>@<hcpserver.edu>:/my_images

Where ``<version>`` should be replaced with the version of ``package_name`` that you used to create the ``Singularity image``, ``<user>``
with your ``user name`` on the HPC and ``<hcpserver.edu>`` with the address of the HPC.  

Running a Singularity Image
---------------------------

.. code-block:: bash

    $ singularity run --cleanenv /my_images/package_name-<version>.simg \
       

.. note::

    Make sure to check the name of the created ``Singularity image`` as that might
    diverge based on the method you used. Here and going forward it is assumed that you used ``Singularity >= 2.5``
    and thus ``package_name-<version>.simg`` instead of ``YourGitHubHandle_package_name<version>.simg``.   


.. note::

   Singularity by default `exposes all environment variables from the host inside
   the container <https://github.com/singularityware/singularity/issues/445>`_.
   Because of this your host libraries (such as nipype) could be accidentally used
   instead of the ones inside the container - if they are included in ``PYTHONPATH``.
   To avoid such situation we recommend using the ``--cleanenv`` singularity flag
   in production use. For example: ::

    $ singularity run --cleanenv /my_images/package_name-<version>.simg \
       


   or, unset the ``PYTHONPATH`` variable before running: ::

    $ unset PYTHONPATH; singularity /my_images/package_name-<version>.simg \
       

.. note::

   Depending on how ``Singularity`` is configured on your cluster it might or might not
   automatically ``bind`` (``mount`` or ``expose``) ``host folders`` to the container.
   If this is not done automatically you will need to ``bind`` the necessary folders using
   the ``-B <host_folder>:<container_folder>`` ``Singularity`` argument.
   For example: ::

    $ singularity run --cleanenv -B path/to/package_name/on_host:/package_name \
        /my_images/package_name-<version>.simg \
        

Bare metal version (Python 3.10+)
===========================================

``package_name`` is written using Python 3.10 (or above).
Until the first official version/release will be provided, `package_name`'s bare metal version can be installed by opening a terminal and running the following:

.. code-block:: bash

    git clone https://github.com/YourGitHubHandle/package_name.git
    cd package_name
    pip install .

Please note that you need to have at least `Python 3.10` installed.

Check your installation with the ``--version`` argument:

.. code-block:: bash

    $ package_name --version
