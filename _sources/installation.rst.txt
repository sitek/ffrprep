============
Installation
============

In general, there are two distinct ways to install and use ``ffrprep``:
either through virtualization/container technology, that is `Docker`_ or
`Singularity`_, or in a `Bare metal version (Python 3.10+)`_.
Once you are ready to run ``ffrprep``, see `Usage <https://spark-csd.github.io/ffrprep/usage>`_ for details.

Docker
======

In order to run ```ffrprep``` in a Docker container, Docker must be `installed
<https://docs.docker.com/engine/installation/>`_ on your system.
Once Docker is installed, you can get ``ffrprep`` through  running one of the following
commands in the terminal of your choice.

Option 1: pulling from the `dockerhub registry <https://hub.docker.com/repository/docker/spark-csd/ffrprep/general>`_ :


.. code-block:: bash

    docker pull spark-csd/ffrprep:version

Option 2: pulling from the `github container registry <https://github.com/spark-csd/ffrprep/pkgs/container/ffrprep>`_ :

.. code-block:: bash

    docker pull ghcr.io/spark-csd/ffrprep:version

Where ``version`` is the specific version of ``ffrprep`` you would like to use. For example, if you want 
to employ the ``latest``/most up to date ``version`` you can either run 

.. code-block:: bash

    docker pull spark-csd/ffrprep:latest

.. code-block:: bash

    docker pull ghcr.io/spark-csd/ffrprep:latest

or the same command withouth the ``:latest`` tag, as ``Docker`` searches for the ``latest`` tag by default.
However, as the ``latest`` version is subject to changes and not necessarily in synch with the most recent ``numbered version``, it 
is recommend to utilize the latter to ensure reproducibility. For example, if you want to employ ``ffrprep v0.0.1`` the command would look as follows:

.. code-block:: bash

    docker pull spark-csd/ffrprep:v0.0.1

.. code-block:: bash

    docker pull ghcr.io/spark-csd/ffrprep:v0.0.1

After the command finished (it may take a while depending on your internet connection),
you can run ``ffrprep`` like this:

.. code-block:: bash

    $ docker run -ti --rm \
        spark-csd/ffrprep:latest \
        

Please have a look at the examples under `Usage <https://spark-csd.github.io/ffrprep/usage>`_ to get more information
about and familiarize yourself with ``ffrprep``'s functionality.

Singularity
===========

For security reasons, many HPCs do not allow Docker containers, but support/allow `Singularity <https://github.com/singularityware/singularity>`_ containers. Depending
on the ``Singularity`` version available to you, there are two options to get ``ffrprep`` as
a ``Singularity image``.

Preparing a Singularity image (Singularity version >= 2.5)
----------------------------------------------------------
If the version of Singularity on your HPC is modern enough you can create a ``Singularity
image`` directly on the HCP.
This is as simple as: 

.. code-block:: bash

    $ singularity build /my_images/ffrprep-<version>.simg docker://spark-csd/ffrprep:<version>

Where ``<version>`` should be replaced with the desired version of ``ffrprep`` that you want to download.
For example, if you want to use ``ffrprep v0.0.4``, the command would look as follows.

.. code-block:: bash

    $ singularity build /my_images/ffrprep-v0.0.4.simg docker://spark-csd/ffrprep:v0.0.4


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
        spark-csd/ffrprep:<version>

Where ``<version>`` should be replaced with the desired version of ```ffrprep``` that you want
to download and ``/absolute/path/to/output/folder`` with the absolute path where the created ``Singularity image``
should be stored. Sticking with the example of ``ffrprep v0.0.4`` this would look as follows:

.. code-block:: bash

    $ docker run --privileged -t --rm \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /absolute/path/to/output/folder:/output \
        singularityware/docker2singularity \
        spark-csd/ffrprep:v0.0.4

Beware of the back slashes, expected for Windows systems. The above command would translate to Windows systems as follows:

.. code-block:: bash

    $ docker run --privileged -t --rm \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v D:\host\path\where\to\output\singularity\image:/output \
        singularityware/docker2singularity \
        spark-csd/ffrprep:<version>


You can then transfer the resulting ``Singularity image`` to the HPC, for example, using ``scp``. ::

    $ scp spark-csd_ffrprep<version>.simg <user>@<hcpserver.edu>:/my_images

Where ``<version>`` should be replaced with the version of ``ffrprep`` that you used to create the ``Singularity image``, ``<user>``
with your ``user name`` on the HPC and ``<hcpserver.edu>`` with the address of the HPC.  

Running a Singularity Image
---------------------------

.. code-block:: bash

    $ singularity run --cleanenv /my_images/ffrprep-<version>.simg \
       

.. note::

    Make sure to check the name of the created ``Singularity image`` as that might
    diverge based on the method you used. Here and going forward it is assumed that you used ``Singularity >= 2.5``
    and thus ``ffrprep-<version>.simg`` instead of ``spark-csd_ffrprep<version>.simg``.   


.. note::

   Singularity by default `exposes all environment variables from the host inside
   the container <https://github.com/singularityware/singularity/issues/445>`_.
   Because of this your host libraries (such as nipype) could be accidentally used
   instead of the ones inside the container - if they are included in ``PYTHONPATH``.
   To avoid such situation we recommend using the ``--cleanenv`` singularity flag
   in production use. For example: ::

    $ singularity run --cleanenv /my_images/ffrprep-<version>.simg \
       


   or, unset the ``PYTHONPATH`` variable before running: ::

    $ unset PYTHONPATH; singularity /my_images/ffrprep-<version>.simg \
       

.. note::

   Depending on how ``Singularity`` is configured on your cluster it might or might not
   automatically ``bind`` (``mount`` or ``expose``) ``host folders`` to the container.
   If this is not done automatically you will need to ``bind`` the necessary folders using
   the ``-B <host_folder>:<container_folder>`` ``Singularity`` argument.
   For example: ::

    $ singularity run --cleanenv -B path/to/ffrprep/on_host:/ffrprep \
        /my_images/ffrprep-<version>.simg \
        

Bare metal version (Python 3.10+)
===========================================

``ffrprep`` is written using Python 3.10 (or above).
Until the first official version/release will be provided, `ffrprep`'s bare metal version can be installed by opening a terminal and running the following:

.. code-block:: bash

    git clone https://github.com/spark-csd/ffrprep.git
    cd ffrprep
    pip install .

Please note that you need to have at least `Python 3.10` installed.

Check your installation with the ``--version`` argument:

.. code-block:: bash

    $ ffrprep --version
