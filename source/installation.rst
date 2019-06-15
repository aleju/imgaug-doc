============
Installation
============

The library uses python, which must be installed.
Python 2.7, 3.4, 3.5, 3.6 and 3.7 are supported.

The below sections explain how to install the library in anaconda or via pip.
If you don't know what anaconda (or conda) are, simply use pip instead as
that should always work.

------------------------
Installation in Anaconda
------------------------

To install in anaconda simply perform the following commands ::

    conda config --add channels conda-forge
    conda install imgaug

Note that you may also use the pip-based installation commands described below.
They work with and without anaconda.


-------------------
Installation in pip
-------------------

Install Requirements
--------------------

To install all requirements, use ::

    pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely

Note that if you already have OpenCV, you might not need ``opencv-python``.
If you get any "permission denied" errors, try adding ``sudo`` in front of the command.
If your encounter issues installing ``Shapely`` you can skip that library.
It is only imported when actually needed. At least polygon and line string
augmentation will likely crash without it.

Install Library
---------------

Once the required packages are available, ``imgaug`` can be installed using
the following command::

    pip install imgaug

This installs the latest version from pypi, which often lags behind the
latest version on github by a few months.
To instead get the very latest version use ::

    pip install git+https://github.com/aleju/imgaug



------------------------
Installation from Source
------------------------

In rare cases, one might prefer to install directly from source.
This is possible using ::

    git clone https://github.com/aleju/imgaug.git && cd imgaug && python setup.py install

Note that this is effectively identical to using ``pip install <github link>``.


---------
Uninstall
---------

To deinstall the library use ::

    conda remove imgaug

on anaconda and ::

    pip uninstall imgaug

otherwise.