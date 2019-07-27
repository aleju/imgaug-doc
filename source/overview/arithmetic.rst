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

Add noise sampled from gaussian distributions elementwise to images.

Add gaussian noise to an image, sampled once per pixel from a normal
distribution ``N(0, s)``, where ``s`` is sampled per image and varies between
``0`` and ``0.2*255``::

    aug = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))

.. figure:: ../../images/overview_of_augmenters/arithmetic/additivegaussiannoise.jpg
    :alt: AdditiveGaussianNoise

Add gaussian noise to an image, sampled once per pixel from a normal
distribution ``N(0, 0.05*255)``::

    aug = iaa.AdditiveGaussianNoise(scale=0.2*255)

.. figure:: ../../images/overview_of_augmenters/arithmetic/additivegaussiannoise_large.jpg
    :alt: AdditiveGaussianNoise large

Add laplace noise to an image, sampled channelwise from
``N(0, 0.2*255)`` (i.e. three independent samples per pixel)::

    aug = iaa.AdditiveGaussianNoise(scale=0.2*255, per_channel=True)

.. figure:: ../../images/overview_of_augmenters/arithmetic/additivegaussiannoise_per_channel.jpg
    :alt: AdditiveGaussianNoise per channel

.. Add gaussian noise from ``N(0, 0.05*255)`` to an image. For 50% of all images,
    a single value is sampled for each pixel and re-used for all three channels
    of that pixel. For the other 50% of all images, three values are sampled
    per pixel (i.e. channelwise sampling).::

        aug = iaa.AdditiveGaussianNoise(scale=0.2*255, per_channel=0.5)

    .. figure:: ../../images/overview_of_augmenters/arithmetic/additivegaussiannoise_per_channel.jpg
        :alt: AdditiveGaussianNoise per channel


AdditiveLaplaceNoise
---------------------

Add noise sampled from laplace distributions elementwise to images.

The laplace distribution is similar to the gaussian distribution, but
puts more weight on the long tail. Hence, this noise will add more
outliers (very high/low values). It is somewhere between gaussian noise and
salt and pepper noise.

Add laplace noise to an image, sampled once per pixel from ``Laplace(0, s)``,
where ``s`` is sampled per image and varies between ``0`` and ``0.2*255``::

    aug = iaa.AdditiveLaplaceNoise(scale=(0, 0.2*255))

.. figure:: ../../images/overview_of_augmenters/arithmetic/additivelaplacenoise.jpg
    :alt: AdditiveLaplaceNoise

Add laplace noise to an image, sampled once per pixel from
``Laplace(0, 0.2*255)``::

    aug = iaa.AdditiveLaplaceNoise(scale=0.2*255)

.. figure:: ../../images/overview_of_augmenters/arithmetic/additivelaplacenoise_large.jpg
    :alt: AdditiveLaplaceNoise large

Add laplace noise to an image, sampled channelwise from
``Laplace(0, 0.2*255)`` (i.e. three independent samples per pixel)::

    aug = iaa.AdditiveLaplaceNoise(scale=0.2*255, per_channel=True)

.. figure:: ../../images/overview_of_augmenters/arithmetic/additivelaplacenoise_per_channel.jpg
    :alt: AdditiveLaplaceNoise per channel

.. Add laplace noise from ``N(0, 0.05*255)`` to an image. For 50% of all images,
    a single value is sampled for each pixel and re-used for all three channels
    of that pixel. For the other 50% of all images, three values are sampled
    per pixel (i.e. channelwise sampling).::

        aug = iaa.AdditiveLaplaceNoise(scale=0.2*255, per_channel=0.5)

    .. figure:: ../../images/overview_of_augmenters/arithmetic/additivelaplacenoise_per_channel.jpg
        :alt: AdditiveLaplaceNoise per channel


AdditivePoissonNoise
---------------------

Add noise sampled from poisson distributions elementwise to images.

Poisson noise is comparable to gaussian noise, as e.g. generated via
``AdditiveGaussianNoise``. As poisson distributions produce only positive
numbers, the sign of the sampled values are here randomly flipped.

Values of around ``20.0`` for ``lam`` lead to visible noise (for ``uint8``).
Values of around ``40.0`` for ``lam`` lead to very visible noise (for
``uint8``).
It is recommended to usually set ``per_channel`` to ``True``.

Add poisson noise to an image, sampled once per pixel from ``Poisson(lam)``,
where ``lam`` is sampled per image and varies between ``0`` and ``40``::

    aug = iaa.AdditivePoissonNoise(scale=(0, 40))

.. figure:: ../../images/overview_of_augmenters/arithmetic/additivepoissonnoise.jpg
    :alt: AdditivePoissonNoise

Add poisson noise to an image, sampled once per pixel from ``Poisson(40)``::

    aug = iaa.AdditivePoissonNoise(40)

.. figure:: ../../images/overview_of_augmenters/arithmetic/additivepoissonnoise_large.jpg
    :alt: AdditivePoissonNoise large

Add poisson noise to an image, sampled channelwise from
``Poisson(40)`` (i.e. three independent samples per pixel)::

    aug = iaa.AdditivePoissonNoise(scale=40, per_channel=True)

.. figure:: ../../images/overview_of_augmenters/arithmetic/additivepoissonnoise_per_channel.jpg
    :alt: AdditivePoissonNoise per channel

.. Add poisson noise from ``Poisson(40)`` to an image. For 50% of all images,
    a single value is sampled for each pixel and re-used for all three channels
    of that pixel. For the other 50% of all images, three values are sampled
    per pixel (i.e. channelwise sampling).::

        aug = iaa.AdditivePoissonNoise(scale=40, per_channel=0.5)

    .. figure:: ../../images/overview_of_augmenters/arithmetic/additivepoissonnoise_per_channel.jpg
        :alt: AdditivePoissonNoise per channel


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

Replace pixels in an image with new values.

Replace ``10%`` of all pixels with either the value ``0`` or the value
``255``::

    aug = ReplaceElementwise(0.1, [0, 255])

.. figure:: ../../images/overview_of_augmenters/arithmetic/replaceelementwise.jpg
    :alt: ReplaceElementwise

For ``50%`` of all images, replace ``10%`` of all pixels with either the value
``0`` or the value ``255`` (same as in the previous example). For the other
``50%`` of all images, replace *channelwise* ``10%`` of all pixels with either
the value ``0`` or the value ``255``. So, it will be very rare for each pixel
to have all channels replaced by ``255`` or ``0``. ::

    aug = ReplaceElementwise(0.1, [0, 255], per_channel=0.5)

.. figure:: ../../images/overview_of_augmenters/arithmetic/replaceelementwise_per_channel_050.jpg
    :alt: ReplaceElementwise per channel at 50%

Replace ``10%`` of all pixels by gaussian noise centered around ``128``. Both
the replacement mask and the gaussian noise are sampled for ``50%`` of all
images. ::

    import imgaug.parameters as iap
    aug = ReplaceElementwise(0.1, iap.Normal(128, 0.4*128), per_channel=0.5)

.. figure:: ../../images/overview_of_augmenters/arithmetic/replaceelementwise_gaussian_noise.jpg
    :alt: ReplaceElementwise with gaussian noise

Replace ``10%`` of all pixels by gaussian noise centered around ``128``. Sample
the replacement mask at a lower resolution (``8x8`` pixels) and upscale it to
the image size, resulting in coarse areas being replaced by gaussian noise. ::

    import imgaug.parameters as iap
    aug = ReplaceElementwise(
        iap.FromLowerResolution(iap.Binomial(0.1), size_px=8),
        iap.Normal(128, 0.4*128),
        per_channel=0.5)

.. figure:: ../../images/overview_of_augmenters/arithmetic/replaceelementwise_gaussian_noise_coarse.jpg
    :alt: ReplaceElementwise with gaussian noise in coarse areas


ImpulseNoise
------------

Add impulse noise to images.

This is identical to ``SaltAndPepper``, except that ``per_channel`` is
always set to ``True``.

Replace ``10%`` of all pixels with impulse noise::

    aug = iaa.ImpulseNoise(0.1)

.. figure:: ../../images/overview_of_augmenters/arithmetic/impulsenoise.jpg
    :alt: ImpulseNoise


SaltAndPepper
-------------

Replace pixels in images with salt/pepper noise (white/black-ish colors).

Replace ``5%`` of all pixels with salt and pepper noise::

    aug = iaa.SaltAndPepper(0.05)

.. figure:: ../../images/overview_of_augmenters/arithmetic/saltandpepper.jpg
    :alt: SaltAndPepper

Replace *channelwise* ``5%`` of all pixels with salt and pepper
noise::

    aug = iaa.SaltAndPepper(0.05, per_channel=True)

.. figure:: ../../images/overview_of_augmenters/arithmetic/saltandpepper_per_channel.jpg
    :alt: SaltAndPepper


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

