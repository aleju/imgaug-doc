============
Installation
============

The library uses python, which must be installed.
Python 2.7, 3.4, 3.5, 3.6, 3.7 and 3.8 are supported.

The below sections explain how to install the library in anaconda and pip.
If you don't know what anaconda (aka conda) is, simply use pip instead.

------------------------
Installation in Anaconda
------------------------

To install in anaconda simply perform the following commands ::

    conda config --add channels conda-forge
    conda install imgaug

Note that you may also use the pip-based installation commands described below.
They work with and without anaconda.

To also be able to use the augmenters in ``imgaug.augmenters.imgcorruptlike``,
you have to manually install the ``imagecorruptions`` package::

    pip install imagecorruptions

-------------------
Installation in pip
-------------------

To install the library via ``pip``, simply execute::

    pip install imgaug

This installs the latest version from pypi.

If you encounter any problems with ``Shapely``, try the following::

    pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio
    pip install --no-dependencies imgaug

The first command installs manually all dependencies except ``Shapely``, the
second only the library.

The version installed above is the latest official release from pypi. That
release often lags behind the latest version from github by a few months.
To instead install the very latest version of imgaug use ::

    pip install git+https://github.com/aleju/imgaug.git

Independent of whether you install from pypi or github, in order to
be able to use the augmenters in ``imgaug.augmenters.imgcorruptlike``,
you have to manually install the ``imagecorruptions`` package::

    pip install imagecorruptions


---------
Uninstall
---------

To deinstall the library use ::

    conda remove imgaug

on anaconda and ::

    pip uninstall imgaug

otherwise.