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

TODO


LinearContrast
--------------

TODO


AllChannelsCLAHE
----------------

TODO


CLAHE
-----

TODO


AllChannelsHistogramEqualization
--------------------------------

TODO


HistogramEqualization
---------------------

TODO

