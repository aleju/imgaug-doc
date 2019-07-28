****************
augmenters.blend
****************

.. note::

    It is not recommended to use blending augmenter with child augmenters
    that change the geometry of images (e.g. horizontal flips, affine
    transformations) if you *also* want to augment coordinates (e.g.
    keypoints, bounding boxes, polygons, ...), as it is not clear which of
    the two coordinate results (first or second branch) should be used as the
    coordinates after augmentation. Currently, all blending augmenters try
    to use the augmented coordinates of the branch that makes up most of the
    augmented image.


Alpha
-----

Alpha-blend two image sources using an alpha/opacity value.


Currently, if ``factor >= 0.5`` (per image), the results of the first
branch are used as the new coordinates, otherwise the results of the
second branch.

Convert each image to pure grayscale and alpha-blend the result with the
original image using an alpha of ``50%``, thereby removing about ``50%`` of
all color. This is equivalent to ``iaa.Grayscale(0.5)``. ::

    aug = iaa.Alpha(0.5, iaa.Grayscale(1.0))

.. figure:: ../../images/overview_of_augmenters/blend/alpha_050_grayscale.jpg
    :alt: Alpha-blend images with grayscale images

Same as in the previous example, but the alpha factor is sampled uniformly
from the interval ``[0.0, 1.0]`` once per image, thereby removing a random
fraction of all colors. This is equivalent to
``iaa.Grayscale((0.0, 1.0))``. ::

    aug = iaa.Alpha((0.0, 1.0), iaa.Grayscale(1.0))

.. figure:: ../../images/overview_of_augmenters/blend/alpha_uniform_factor.jpg
    :alt: Alpha-blend images with grayscale images using a random factor

First, rotate each image by a random degree sampled uniformly from the
interval ``[-20, 20]``. Then, alpha-blend that new image with the original
one using a random factor sampled uniformly from the interval
``[0.0, 1.0]``. For ``50%`` of all images, the blending happens
channel-wise and the factor is sampled independently per channel
(``per_channel=0.5``). As a result, e.g. the red channel may look visibly
rotated (factor near ``1.0``), while the green and blue channels may not
look rotated (factors near ``0.0``). ::

    aug = iaa.Alpha(
        (0.0, 1.0),
        iaa.Affine(rotate=(-20, 20)),
        per_channel=0.5)

.. figure:: ../../images/overview_of_augmenters/blend/alpha_affine_per_channel.jpg
    :alt: Alpha-blend images channelwise with rotated ones

Apply two branches of augmenters -- ``A`` and ``B`` -- *independently*
to input images and alpha-blend the results of these branches using a
factor ``f``. Branch ``A`` increases image pixel intensities by ``100``
and ``B`` multiplies the pixel intensities by ``0.2``. ``f`` is sampled
uniformly from the interval ``[0.0, 1.0]`` per image. The resulting images
contain a bit of ``A`` and a bit of ``B``. ::

    aug = iaa.Alpha(
        (0.0, 1.0),
        first=iaa.Add(100),
        second=iaa.Multiply(0.2))

.. figure:: ../../images/overview_of_augmenters/blend/alpha_two_branches.jpg
    :alt: Alpha with two branches

Apply median blur to each image and alpha-blend the result with the original
image using an alpha factor of either exactly ``0.25`` or exactly ``0.75``
(sampled once per image). ::

    aug = iaa.Alpha([0.25, 0.75], iaa.MedianBlur(13))

.. figure:: ../../images/overview_of_augmenters/blend/alpha_with_choice.jpg
    :alt: Alpha with a list of factors to use


AlphaElementwise
----------------

Alpha-blend two image sources using alpha/opacity values sampled per pixel.

This is the same as ``Alpha``, except that the opacity factor is
sampled once per *pixel* instead of once per *image* (or a few times per
image, if ``Alpha.per_channel`` is set to ``True``).

Currently, if ``factor >= 0.5`` (per pixel), the results of the first
branch are used as the new coordinates, otherwise the results of the
second branch.

Convert each image to pure grayscale and alpha-blend the result with the
original image using an alpha of ``50%`` for all pixels, thereby removing
about ``50%`` of all color. This is equivalent to ``iaa.Grayscale(0.5)``.
This is also equivalent to ``iaa.Alpha(0.5, iaa.Grayscale(1.0))``, as
the opacity has a fixed value of ``0.5`` and is hence identical for all
pixels. ::

    aug = iaa.AlphaElementwise(0.5, iaa.Grayscale(1.0))

.. figure:: ../../images/overview_of_augmenters/blend/alphaelementwise_050_grayscale.jpg
    :alt: Alpha-blend images pixelwise with grayscale images

Same as in the previous example, but the alpha factor is sampled uniformly
from the interval ``[0.0, 1.0]`` once per pixel, thereby removing a random
fraction of all colors from each pixel. This is equivalent to
``iaa.Grayscale((0.0, 1.0))``. ::

    aug = iaa.AlphaElementwise((0, 1.0), iaa.Grayscale(1.0))

.. figure:: ../../images/overview_of_augmenters/blend/alphaelementwise_uniform_factor.jpg
    :alt: Alpha-blend images pixelwise with grayscale images using a random factor

First, rotate each image by a random degree sampled uniformly from the
interval ``[-20, 20]``. Then, alpha-blend that new image with the original
one using a random factor sampled uniformly from the interval
``[0.0, 1.0]`` per pixel. For ``50%`` of all images, the blending happens
channel-wise and the factor is sampled independently per pixel *and*
channel (``per_channel=0.5``). As a result, e.g. the red channel may look
visibly rotated (factor near ``1.0``), while the green and blue channels
may not look rotated (factors near ``0.0``). ::

    aug = iaa.AlphaElementwise(
        (0.0, 1.0),
        iaa.Affine(rotate=(-20, 20)),
        per_channel=0.5)

.. figure:: ../../images/overview_of_augmenters/blend/alphaelementwise_affine_per_channel.jpg
    :alt: Alpha-blend images pixelwise and channelwise with rotated ones

Apply two branches of augmenters -- ``A`` and ``B`` -- *independently*
to input images and alpha-blend the results of these branches using a
factor ``f``. Branch ``A`` increases image pixel intensities by ``100``
and ``B`` multiplies the pixel intensities by ``0.2``. ``f`` is sampled
uniformly from the interval ``[0.0, 1.0]`` per pixel. The resulting images
contain a bit of ``A`` and a bit of ``B``. ::

    aug = iaa.AlphaElementwise(
        (0.0, 1.0),
        first=iaa.Add(100),
        second=iaa.Multiply(0.2))

.. figure:: ../../images/overview_of_augmenters/blend/alphaelementwise_two_branches.jpg
    :alt: AlphaElementwise with two branches

Apply median blur to each image and alpha-blend the result with the
original image using an alpha factor of either exactly ``0.25`` or
exactly ``0.75`` (sampled once per pixel). ::

    aug = iaa.AlphaElementwise([0.25, 0.75], iaa.MedianBlur(13))

.. figure:: ../../images/overview_of_augmenters/blend/alphaelementwise_with_choice.jpg
    :alt: AlphaElementwise with a list of factors to use


SimplexNoiseAlpha
-----------------

TODO


FrequencyNoiseAlpha
-------------------

TODO

