***************
augmenters.blur
***************

GaussianBlur
------------

Augmenter to blur images using gaussian kernels.

Blur each image with a gaussian kernel with a sigma of ``3.0``::

    aug = iaa.GaussianBlur(sigma=(0.0, 3.0))

.. figure:: ../../images/overview_of_augmenters/blur/gaussianblur.jpg
    :alt: GaussianBlur


AverageBlur
-----------

Blur an image by computing simple means over neighbourhoods.

Blur each image using a mean over neihbourhoods that have a random size
between 2x2 and 11x11::

    aug = iaa.AverageBlur(k=(2, 11))

.. figure:: ../../images/overview_of_augmenters/blur/averageblur.jpg
    :alt: AverageBlur

Blur each image using a mean over neihbourhoods that have random sizes,
which can vary between 5 and 11 in height and 1 and 3 in width::

    aug = iaa.AverageBlur(k=((5, 11), (1, 3)))

.. figure:: ../../images/overview_of_augmenters/blur/averageblur_mixed.jpg
    :alt: AverageBlur varying height/width


MedianBlur
----------

Blur an image by computing median values over neighbourhoods.

Blur each image using a median over neihbourhoods that have a random size
between 3x3 and 11x11::

    aug = iaa.MedianBlur(k=(3, 11))

.. figure:: ../../images/overview_of_augmenters/blur/medianblur.jpg
    :alt: MedianBlur


BilateralBlur
-------------

TODO


MotionBlur
----------

TODO

