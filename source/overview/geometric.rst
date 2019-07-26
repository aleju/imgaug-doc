********************
augmenters.geometric
********************


Affine
------

Augmenter to apply affine transformations to images.

Scale images to a value of 50 to 150% of their original size::

    aug = iaa.Affine(scale=(0.5, 1.5))

.. figure:: ../../images/overview_of_augmenters/geometric/affine_scale.jpg
    :alt: Affine scale

Scale images to a value of 50 to 150% of their original size,
but do this independently per axis (i.e. sample two values per image)::

    aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})

.. figure:: ../../images/overview_of_augmenters/geometric/affine_scale_independently.jpg
    :alt: Affine scale independently

Translate images by -20 to +20% on x- and y-axis independently::

    aug = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})

.. figure:: ../../images/overview_of_augmenters/geometric/affine_translate_percent.jpg
    :alt: Affine translate percent

Translate images by -20 to 20 pixels on x- and y-axis independently::

    aug = iaa.Affine(translate_px={"x": (-20, 20), "y": (-20, 20)})

.. figure:: ../../images/overview_of_augmenters/geometric/affine_translate_px.jpg
    :alt: Affine translate pixel

Rotate images by -45 to 45 degrees::

    aug = iaa.Affine(rotate=(-45, 45))

.. figure:: ../../images/overview_of_augmenters/geometric/affine_rotate.jpg
    :alt: Affine rotate

Shear images by -16 to 16 degrees::

    aug = iaa.Affine(shear=(-16, 16))

.. figure:: ../../images/overview_of_augmenters/geometric/affine_shear.jpg
    :alt: Affine shear

When applying affine transformations, new pixels are often generated, e.g. when
translating to the left, pixels are generated on the right. Various modes
exist to set how these pixels are ought to be filled. Below code shows an
example that uses all modes, sampled randomly per image. If the mode is
``constant`` (fill all with one constant value), then a random brightness
between 0 and 255 is used::

    aug = iaa.Affine(translate_percent={"x": -0.20}, mode=ia.ALL, cval=(0, 255))

.. figure:: ../../images/overview_of_augmenters/geometric/affine_fill.jpg
    :alt: Affine fill modes


PiecewiseAffine
---------------

Augmenter that places a regular grid of points on an image and randomly
moves the neighbourhood of these point around via affine transformations.
This leads to local distortions.

Distort images locally by moving points around, each with a distance v (percent
relative to image size), where v is sampled per point from ``N(0, z)``
``z`` is sampled per image from the range 0.01 to 0.05::

    aug = iaa.PiecewiseAffine(scale=(0.01, 0.05))

.. figure:: ../../images/overview_of_augmenters/geometric/piecewiseaffine.jpg
    :alt: PiecewiseAffine

.. figure:: ../../images/overview_of_augmenters/geometric/piecewiseaffine_checkerboard.jpg
    :alt: PiecewiseAffine

Effect of increasing ``scale`` from 0.01 to 0.3 in 8 steps:

.. figure:: ../../images/overview_of_augmenters/geometric/piecewiseaffine_vary_scales.jpg
    :alt: PiecewiseAffine varying scales

PiecewiseAffine works by placing a regular grid of points on the image
and moving them around. By default this grid consists of 4x4 points.
The below image shows the effect of increasing that value from 2x2 to 16x16
in 8 steps:

.. figure:: ../../images/overview_of_augmenters/geometric/piecewiseaffine_vary_grid.jpg
    :alt: PiecewiseAffine varying grid

.. note::

    This augmenter is very slow. See :ref:`performance`.
    Try to use ``ElasticTransformation`` instead, which is at least 10x
    faster.

.. note::

    For coordinate-based inputs (keypoints, bounding boxes, polygons,
    ...), this augmenter still has to perform an image-based augmentation,
    which will make it significantly slower for such inputs than other
    augmenters. See :ref:`performance`.


PerspectiveTransform
--------------------

TODO


ElasticTransformation
---------------------

Augmenter to transform images by moving pixels locally around using
displacement fields.

Distort images locally by moving individual pixels around following
a distortions field with strength 0.25. The strength of the movement is
sampled per pixel from the range 0 to 5.0::

    aug = iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)

.. figure:: ../../images/overview_of_augmenters/geometric/elastictransformations.jpg
    :alt: ElasticTransformation

Effect of keeping sigma fixed at 0.25 and increasing alpha from 0 to 5.0
in 8 steps:

.. figure:: ../../images/overview_of_augmenters/geometric/elastictransformations_vary_alpha.jpg
    :alt: ElasticTransformation varying alpha

Effect of keeping alpha fixed at 2.5 and increasing sigma from 0.01 to 1.0
in 8 steps:

.. figure:: ../../images/overview_of_augmenters/geometric/elastictransformations_vary_sigmas.jpg
    :alt: ElasticTransformation varying sigma

.. note::

    For coordinate-based inputs (keypoints, bounding boxes, polygons,
    ...), this augmenter still has to perform an image-based augmentation,
    which will make it significantly slower for such inputs than other
    augmenters. See :ref:`performance`.


Rot90
-----

TODO

