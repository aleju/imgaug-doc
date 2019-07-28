****************
augmenters.color
****************

WithColorspace
--------------

Apply child augmenters within a specific colorspace.

This augumenter takes a source colorspace A and a target colorspace B
as well as children C. It changes images from A to B, then applies the
child augmenters C and finally changes the colorspace back from B to A.
See also ChangeColorspace() for more.

Convert to ``HSV`` colorspace, add a value between ``0`` and ``50``
(uniformly sampled per image) to the Hue channel, then convert back to the
input colorspace (``RGB``). ::

    import imgaug.augmenters as iaa
    aug = iaa.WithColorspace(
        to_colorspace="HSV",
        from_colorspace="RGB",
        children=iaa.WithChannels(
            0,
            iaa.Add((0, 50))
        )
    )

.. figure:: ../../images/overview_of_augmenters/color/withcolorspace.jpg
    :alt: WithColorspace


WithHueAndSaturation
--------------------

TODO


MultiplyHueAndSaturation
------------------------

TODO


MultiplyHue
-----------

TODO


MultiplySaturation
------------------

TODO



AddToHueAndSaturation
---------------------

TODO


AddToHue
--------

TODO


AddToSaturation
---------------

TODO


ChangeColorspace
----------------

Augmenter to change the colorspace of images.

The following example shows how to change the colorspace from RGB to HSV,
then add 50-100 to the first channel, then convert back to RGB.
This increases the hue value of each image. ::

    aug = iaa.Sequential([
        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
        iaa.WithChannels(0, iaa.Add((50, 100))),
        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
    ])

.. figure:: ../../images/overview_of_augmenters/color/changecolorspace.jpg
    :alt: Change colorspace


Grayscale
---------

Augmenter to convert images to their grayscale versions.

Change images to grayscale and overlay them with the original image by varying
strengths, effectively removing 0 to 100% of the color::

    aug = iaa.Grayscale(alpha=(0.0, 1.0))

.. figure:: ../../images/overview_of_augmenters/color/grayscale.jpg
    :alt: Grayscale

Visualization of increasing ``alpha`` from 0.0 to 1.0 in 8 steps:

.. figure:: ../../images/overview_of_augmenters/color/grayscale_vary_alpha.jpg
    :alt: Grayscale vary alpha


KMeansColorQuantization
-----------------------

TODO


UniformColorQuantization
------------------------

TODO

