***************
augmenters.flip
***************

HorizontalFlip
--------------

Alias for ``Fliplr``.


VericalFlip
--------------

Alias for ``Flipud``.


Fliplr
------

Flip/mirror input images horizontally.

Flip 50% of all images horizontally::

    aug = iaa.Fliplr(0.5)

NOTE: the default probability is 0, so to flip all images, do::

    aug = iaa.Fliplr(1)

.. figure:: ../../images/overview_of_augmenters/flip/fliplr.jpg
    :alt: Horizontal flip


Flipud
------

Flip/mirror input images vertically.

Flip 50% of all images vertically::

    aug = iaa.Flipud(0.5)

NOTE: the default probability is 0, so to flip all images, do::

    aug = iaa.Flipud(1)

.. figure:: ../../images/overview_of_augmenters/flip/flipud.jpg
    :alt: Vertical flip

