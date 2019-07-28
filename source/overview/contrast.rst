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

TODO


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

