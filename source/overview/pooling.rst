******************
augmenters.pooling
******************

AveragePooling
--------------

Apply average pooling to images.

This augmenter pools images with kernel sizes ``H x W`` by averaging the
pixel values within these windows. For e.g. ``2 x 2`` this halves the image
size. Optionally, the augmenter will automatically re-upscale the image
to the input size (by default this is activated).

This augmenter does not affect heatmaps, segmentation maps or
coordinates-based augmentables (e.g. keypoints, bounding boxes, ...).

Note that this augmenter is very similar to ``AverageBlur``.
``AverageBlur`` applies averaging within windows of given kernel size
*without* striding, while ``AveragePooling`` applies striding corresponding
to the kernel size, with optional upscaling afterwards. The upscaling
is configured to create "pixelated"/"blocky" images by default.


Create an augmenter that always pools with a kernel size of ``2 x 2``::

    import imgaug.augmenters as iaa
    aug = AveragePooling(2)

.. figure:: ../../images/overview_of_augmenters/pooling/averagepooling.jpg
    :alt: AveragePooling

Create an augmenter that always pools with a kernel size of ``2 x 2``
and does *not* resize back to the input image size, i.e. the resulting
images have half the resolution::

    aug = AveragePooling(2, keep_size=False)

.. figure:: ../../images/overview_of_augmenters/pooling/averagepooling_keep_size_false.jpg
    :alt: AveragePooling with keep_size=False

Create an augmenter that always pools either with a kernel size
of ``2 x 2`` or ``8 x 8``::

    aug = AveragePooling([2, 8])

.. figure:: ../../images/overview_of_augmenters/pooling/averagepooling_choice.jpg
    :alt: AveragePooling with a choice of two kernel sizes

Create an augmenter that always pools with a kernel size of
``1 x 1`` (does nothing) to ``7 x 7``. The kernel sizes are always
symmetric. ::

    aug = AveragePooling((1, 7))

.. figure:: ../../images/overview_of_augmenters/pooling/averagepooling_uniform.jpg
    :alt: AveragePooling with a uniform distribution over kernel sizes

Create an augmenter that always pools with a kernel size of
``H x W`` where ``H`` and ``W`` are both sampled independently from the
range ``[1..7]``. E.g. resulting kernel sizes could be ``3 x 7``
or ``5 x 1``. ::

    aug = AveragePooling(((1, 7), (1, 7)))

.. figure:: ../../images/overview_of_augmenters/pooling/averagepooling_unsymmetric.jpg
    :alt: AveragePooling with unsymmetric kernel sizes


MaxPooling
----------

TODO


MinPooling
----------

TODO


MedianPooling
-------------

TODO

