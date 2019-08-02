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

.. note ::

    The default value for the probability is ``0.0``.
    So, to flip *all* input image use ``Fliplr(1.0)`` and *not* just
    ``Fliplr()``.

**Example.**
Flip 50% of all images horizontally::

    import imgaug.augmenters as iaa
    aug = iaa.Fliplr(0.5)

.. figure:: ../../images/overview_of_augmenters/flip/fliplr.jpg
    :alt: Horizontal flip


Flipud
------

Flip/mirror input images vertically.

.. note ::

    The default value for the probability is ``0.0``.
    So, to flip *all* input image use ``Flipud(1.0)`` and *not* just
    ``Flipud()``.

**Example.**
Flip 50% of all images vertically::

    aug = iaa.Flipud(0.5)

.. figure:: ../../images/overview_of_augmenters/flip/flipud.jpg
    :alt: Vertical flip

