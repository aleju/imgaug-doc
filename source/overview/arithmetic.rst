*********************
augmenters.arithmetic
*********************


Add
---

Add a value to all pixels in an image.

Add random values between -40 and 40 to images, with each value being sampled
once per image and then being the same for all pixels::

    aug = iaa.Add((-40, 40))

.. figure:: ../../images/overview_of_augmenters/arithmetic/add.jpg
    :alt: Add

Add random values between -40 and 40 to images. In 50% of all images the
values differ per channel (3 sampled value). In the other 50% of all images
the value is the same for all channels::

    aug = iaa.Add((-40, 40), per_channel=0.5)

.. figure:: ../../images/overview_of_augmenters/arithmetic/add_per_channel.jpg
    :alt: Add per channel


AddElementwise
--------------

Add values to the pixels of images with possibly different values
for neighbouring pixels.

Add random values between -40 and 40 to images, with each value being sampled
per pixel::

    aug = iaa.AddElementwise((-40, 40))

.. figure:: ../../images/overview_of_augmenters/arithmetic/addelementwise.jpg
    :alt: AddElementwise

Add random values between -40 and 40 to images. In 50% of all images the
values differ per channel (3 sampled values per pixel).
In the other 50% of all images the value is the same for all channels per pixel::

    aug = iaa.AddElementwise((-40, 40), per_channel=0.5)

.. figure:: ../../images/overview_of_augmenters/arithmetic/addelementwise_per_channel.jpg
    :alt: AddElementwise per channel


AdditiveGaussianNoise
---------------------

Add gaussian noise (aka white noise) to images.

Add gaussian noise to an image, sampled once per pixel from a normal
distribution ``N(0, s)``, where ``s`` is sampled per image and varies between
0 and 0.05\*255::

    aug = iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))

.. figure:: ../../images/overview_of_augmenters/arithmetic/additivegaussiannoise.jpg
    :alt: AdditiveGaussianNoise

Add gaussian noise to an image, sampled once per pixel from a normal
distribution ``N(0, 0.05*255)``::

    aug = iaa.AdditiveGaussianNoise(scale=0.05*255)

.. figure:: ../../images/overview_of_augmenters/arithmetic/additivegaussiannoise_large.jpg
    :alt: AdditiveGaussianNoise large

Add gaussian noise from ``N(0, 0.05*255)`` to an image. For 50% of all images,
a single value is sampled for each pixel and re-used for all three channels
of that pixel. For the other 50% of all images, three values are sampled
per pixel (i.e. channelwise sampling).::

    aug = iaa.AdditiveGaussianNoise(scale=0.05*255, per_channel=0.5)

.. figure:: ../../images/overview_of_augmenters/arithmetic/additivegaussiannoise_per_channel.jpg
    :alt: AdditiveGaussianNoise per channel


AdditiveLaplaceNoise
---------------------

Add noise sampled from laplace distributions elementwise to images.

The laplace distribution is similar to the gaussian distribution, but
puts more weight on the long tail. Hence, this noise will add more
outliers (very high/low values). It is somewhere between gaussian noise and
salt and pepper noise.

Add laplace noise with ``loc=0`` and varying ``scale`` from ``0`` to
``0.5*255``::

    aug = iaa.AdditiveLaplaceNoise(scale=(0, 0.2*255))

.. figure:: ../../images/overview_of_augmenters/arithmetic/additivelaplacenoise.jpg
    :alt: AdditiveLaplaceNoise

Add gaussian noise to an image, sampled once per pixel from a laplace
distribution ``Laplace(0, 0.05*255)``::

    aug = iaa.AdditiveLaplaceNoise(scale=0.2*255)

.. figure:: ../../images/overview_of_augmenters/arithmetic/additivelaplacenoise_large.jpg
    :alt: AdditiveLaplaceNoise large

Add laplace noise from ``N(0, 0.05*255)`` to an image. For 50% of all images,
a single value is sampled for each pixel and re-used for all three channels
of that pixel. For the other 50% of all images, three values are sampled
per pixel (i.e. channelwise sampling).::

    aug = iaa.AdditiveLaplaceNoise(scale=0.2*255, per_channel=0.5)

.. figure:: ../../images/overview_of_augmenters/arithmetic/additivelaplacenoise_per_channel.jpg
    :alt: AdditiveLaplaceNoise per channel


AdditivePoissonNoise
---------------------

TODO


Multiply
--------

Multiply all pixels in an image with a specific value, thereby making the
image darker or brighter.

Multiply each image with a random value between 0.5 and 1.5::

    aug = iaa.Multiply((0.5, 1.5))

.. figure:: ../../images/overview_of_augmenters/arithmetic/multiply.jpg
    :alt: Multiply

Multiply 50% of all images with a random value between 0.5 and 1.5
and multiply the remaining 50% channel-wise, i.e. sample one multiplier
independently per channel::

    aug = iaa.Multiply((0.5, 1.5), per_channel=0.5)

.. figure:: ../../images/overview_of_augmenters/arithmetic/multiply_per_channel.jpg
    :alt: Multiply per channel


MultiplyElementwise
-------------------

Multiply values of pixels with possibly different values for neighbouring
pixels, making each pixel darker or brighter.

Multiply each pixel with a random value between 0.5 and 1.5::

    aug = iaa.MultiplyElementwise((0.5, 1.5))

.. figure:: ../../images/overview_of_augmenters/arithmetic/multiplyelementwise.jpg
    :alt: MultiplyElementwise

Multiply in 50% of all images each pixel with random values between 0.5 and 1.5
and multiply in the remaining 50% of all images the pixels channel-wise, i.e.
sample one multiplier independently per channel and pixel::

    aug = iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5)

.. figure:: ../../images/overview_of_augmenters/arithmetic/multiplyelementwise_per_channel.jpg
    :alt: MultiplyElementwise per channel


Dropout
-------

Augmenter that sets a certain fraction of pixels in images to zero.

Sample per image a value p from the range 0<=p<=0.2 and then drop p percent
of all pixels in the image (i.e. convert them to black pixels)::

    aug = iaa.Dropout(p=(0, 0.2))

.. figure:: ../../images/overview_of_augmenters/arithmetic/dropout.jpg
    :alt: Dropout

Sample per image a value p from the range 0<=p<=0.2 and then drop p percent
of all pixels in the image (i.e. convert them to black pixels), but
do this independently per channel in 50% of all images::

    aug = iaa.Dropout(p=(0, 0.2), per_channel=0.5)

.. figure:: ../../images/overview_of_augmenters/arithmetic/dropout_per_channel.jpg
    :alt: Dropout per channel


CoarseDropout
-------------

Augmenter that sets rectangular areas within images to zero.

Drop 2% of all pixels by converting them to black pixels, but do
that on a lower-resolution version of the image that has 50% of the original
size, leading to 2x2 squares being dropped::

    aug = iaa.CoarseDropout(0.02, size_percent=0.5)

.. figure:: ../../images/overview_of_augmenters/arithmetic/coarsedropout.jpg
    :alt: CoarseDropout

Drop 0 to 5% of all pixels by converting them to black pixels, but do
that on a lower-resolution version of the image that has 5% to 50% of the
original size, leading to large rectangular areas being dropped::

    aug = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))

.. figure:: ../../images/overview_of_augmenters/arithmetic/coarsedropout_both_uniform.jpg
    :alt: CoarseDropout p and size uniform

Drop 2% of all pixels by converting them to black pixels, but do
that on a lower-resolution version of the image that has 50% of the original
size, leading to 2x2 squares being dropped. Also do this in 50% of all
images channel-wise, so that only the information of some channels in set
to 0 while others remain untouched::

    aug = iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5)

.. figure:: ../../images/overview_of_augmenters/arithmetic/coarsedropout_per_channel.jpg
    :alt: CoarseDropout per channel


ReplaceElementwise
------------------

TODO


ImpulseNoise
------------

TODO


SaltAndPepper
-------------

TODO


CoarseSaltAndPepper
-------------------

TODO


Salt
----

TODO


CoarseSalt
----------

TODO


Pepper
------

TODO


CoarsePepper
------------

TODO


Invert
------

Augmenter that inverts all values in images, i.e. sets a pixel from value
``v`` to ``255-v``.

Invert in 50% of all images all pixels::

    aug = iaa.Invert(0.5)

.. figure:: ../../images/overview_of_augmenters/arithmetic/invert.jpg
    :alt: Invert

For 50% of all images, invert all pixels in these images with 25% probability
(per image). In the remaining 50% of all images, invert 25% of all channels::

    aug = iaa.Invert(0.25, per_channel=0.5)

.. figure:: ../../images/overview_of_augmenters/arithmetic/invert_per_channel.jpg
    :alt: Invert per channel


ContrastNormalization
---------------------

Augmenter that changes the contrast of images.

Normalize contrast by a factor of 0.5 to 1.5, sampled randomly per image::

    aug = iaa.ContrastNormalization((0.5, 1.5))

.. figure:: ../../images/overview_of_augmenters/arithmetic/contrastnormalization.jpg
    :alt: ContrastNormalization

Normalize contrast by a factor of 0.5 to 1.5, sampled randomly per image
and for 50% of all images also independently per channel::

    aug = iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)

.. figure:: ../../images/overview_of_augmenters/arithmetic/contrastnormalization_per_channel.jpg
    :alt: ContrastNormalization per channel


JpegCompression
---------------

TODO

