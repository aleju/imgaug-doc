*******************
augmenters.contrast
*******************

GammaContrast
-------------

Adjust image contrast by scaling pixel values to ``255*((v/255)**gamma)``.

Values in the range ``gamma=(0.5, 2.0)`` seem to be sensible.

Modify the contrast of images according to ``255*((v/255)**gamma)``,
where ``v`` is a pixel value and ``gamma`` is sampled uniformly from
the interval ``[0.5, 2.0]`` (once per image)::

    import imgaug.augmenters as iaa
    aug = iaa.GammaContrast((0.5, 2.0))

.. figure:: ../../images/overview_of_augmenters/contrast/gammacontrast.jpg
    :alt: GammaContrast

Same as in the previous example, but ``gamma`` is sampled once per image
*and* channel::

    aug = iaa.GammaContrast((0.5, 2.0), per_channel=True)

.. figure:: ../../images/overview_of_augmenters/contrast/gammacontrast_per_channel.jpg
    :alt: GammaContrast per_channel


SigmoidContrast
---------------

Adjust image contrast to ``255*1/(1+exp(gain*(cutoff-I_ij/255)))``.

Values in the range ``gain=(5, 20)`` and ``cutoff=(0.25, 0.75)`` seem to
be sensible.

Modify the contrast of images according to
``255*1/(1+exp(gain*(cutoff-v/255)))``, where ``v`` is a pixel value,
``gain`` is sampled uniformly from the interval ``[3, 10]`` (once per
image) and ``cutoff`` is sampled uniformly from the interval
``[0.4, 0.6]`` (also once per image). ::

    import imgaug.augmenters as iaa
    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))

.. figure:: ../../images/overview_of_augmenters/contrast/sigmoidcontrast.jpg
    :alt: SigmoidContrast

Same as in the previous example, but ``gain`` and ``cutoff`` are each
sampled once per image *and* channel::

    aug = iaa.SigmoidContrast(
        gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True)

.. figure:: ../../images/overview_of_augmenters/contrast/sigmoidcontrast_per_channel.jpg
    :alt: SigmoidContrast per_channel


LogContrast
-----------

Adjust image contrast by scaling pixels to ``255*gain*log_2(1+v/255)``.

This augmenter is fairly similar to
``imgaug.augmenters.arithmetic.Multiply``.

Modify the contrast of images according to ``255*gain*log_2(1+v/255)``,
where ``v`` is a pixel value and ``gain`` is sampled uniformly from the
interval ``[0.6, 1.4]`` (once per image)::

    import imgaug.augmenters as iaa
    aug = iaa.LogContrast(gain=(0.6, 1.4))

.. figure:: ../../images/overview_of_augmenters/contrast/logcontrast.jpg
    :alt: LogContrast

Same as in the previous example, but ``gain`` is sampled once per image
*and* channel::

    aug = iaa.LogContrast(gain=(0.6, 1.4), per_channel=True)

.. figure:: ../../images/overview_of_augmenters/contrast/logcontrast_per_channel.jpg
    :alt: LogContrast per_channel


LinearContrast
--------------

Adjust contrast by scaling each pixel to ``127 + alpha*(v-127)``.

Modify the contrast of images according to `127 + alpha*(v-127)``,
where ``v`` is a pixel value and ``alpha`` is sampled uniformly from the
interval ``[0.4, 1.6]`` (once per image)::

    import imgaug.augmenters as iaa
    aug = iaa.LinearContrast((0.4, 1.6))

.. figure:: ../../images/overview_of_augmenters/contrast/linearcontrast.jpg
    :alt: LinearContrast

Same as in the previous example, but ``alpha`` is sampled once per image
*and* channel::

    aug = iaa.LinearContrast((0.4, 1.6), per_channel=True)

.. figure:: ../../images/overview_of_augmenters/contrast/linearcontrast_per_channel.jpg
    :alt: LinearContrast per_channel


AllChannelsCLAHE
----------------

Apply CLAHE to all channels of images in their original colorspaces.

CLAHE (Contrast Limited Adaptive Histogram Equalization) performs
histogram equilization within image patches, i.e. over local
neighbourhoods.

In contrast to ``imgaug.augmenters.contrast.CLAHE``, this augmenter
operates directly on all channels of the input images. It does not
perform any colorspace transformations and does not focus on specific
channels (e.g. ``L`` in ``Lab`` colorspace).

Create an augmenter that applies CLAHE to all channels of input images::

    import imgaug.augmenters as iaa
    aug = iaa.AllChannelsCLAHE()

.. figure:: ../../images/overview_of_augmenters/contrast/allchannelsclahe.jpg
    :alt: AllChannelsCLAHE with default settings

Same as in the previous example, but the `clip_limit` used by CLAHE is
uniformly sampled per image from the interval ``[1, 10]``. Some images
will therefore have stronger contrast than others (i.e. higher clip limit
values). ::

    aug = iaa.AllChannelsCLAHE(clip_limit=(1, 10))

.. figure:: ../../images/overview_of_augmenters/contrast/allchannelsclahe_random_clip_limit.jpg
    :alt: AllChannelsCLAHE with random clip_limit

Same as in the previous example, but the `clip_limit` is sampled per
image *and* channel, leading to different levels of contrast for each
channel::

    aug = iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True)

.. figure:: ../../images/overview_of_augmenters/contrast/allchannelsclahe_per_channel.jpg
    :alt: AllChannelsCLAHE with random clip_limit and per_channel


CLAHE
-----

TODO


AllChannelsHistogramEqualization
--------------------------------

TODO


HistogramEqualization
---------------------

TODO

