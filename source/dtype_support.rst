=============
dtype support
=============

The function ``augment_images()``, which all augmenters in ``imgaug`` offer,
works by default with numpy arrays. In most cases, these arrays will have the numpy dtype ``uint8``,
i.e. the images will have values in the range ``[0, 255]``. This is the datatype returned by
most image loading functions. Sometimes however you may want to augment other datatypes,
e.g. ``float64``. While all augmenters support ``uint8``, the support for other datatypes varies.
The tables further below show which datatype is supported by which augmenter (alongside the dtype
support in some helper functions). The API of each augmenter may contain more details.

Note: Whenever possible it is suggested to use ``uint8`` as that is the most thoroughly tested
dtype. In general, the use of large dtypes (i.e. ``uint64``, ``int64``, ``float128``) is
discouraged, even when they are marked as supported. That is because writing correct tests for
these dtypes can be difficult as no larger dtypes are available to which values can be temporarily
cast. Additionally, when using inputs for which precise discrete values are important (e.g.
segmentation maps, where an accidental change by ``1`` would break the map), medium sized dtypes
(``uint32``, ``int32``) should be used with caution. This is because other libraries may convert
temporarily to ``float64``, which could lead to inaccuracies for some numbers.

Legend
------

Support level (color of table cells):

    * Green: Datatype is considered supported by the augmenter.
    * Yellow: Limited support for the datatype, e.g. due to inaccuracies around large values.
      See the API for the respective augmenter for more details.
    * Red: Datatype is not supported by the augmenter.

Test level (symbols in table cells):

    * ``+++``: Datatype support is thoroughly tested (via unittests or integration tests).
    * ``++``: Datatype support is tested, though not thoroughly.
    * ``+``: Datatype support is indirectly tested via tests for other augmenters.
    * ``-``: Datatype support is not tested.
    * ``?``: Unknown support level for the datatype.

imgaug helper functions
-----------------------

.. figure:: ../images/dtype_support/imgaug_imgaug.png
    :alt: dtype support imgaug.imgaug

    Dtype support of helper functions in ``imgaug``,
    e.g. ``import imgaug; imgaug.imresize_single_image(array, size)``.

imgaug.augmenters.meta
----------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_meta.png
    :alt: dtype support for augmenters in imgaug.augmenters.meta

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.meta``.

imgaug.augmenters.arithmetic
----------------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_arithmetic.png
    :alt: dtype support for augmenters in imgaug.augmenters.arithmetic

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.arithmetic``.

imgaug.augmenters.blend
-----------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_blend.png
    :alt: dtype support for augmenters in imgaug.augmenters.blend

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.blend``.

imgaug.augmenters.blur
----------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_blur.png
    :alt: dtype support for augmenters in imgaug.augmenters.blur

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.blur``.

imgaug.augmenters.collections
-----------------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_collections.png
    :alt: dtype support for augmenters in imgaug.augmenters.collections

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.collections``.

imgaug.augmenters.color
-----------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_color.png
    :alt: dtype support for augmenters in imgaug.augmenters.color

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.color``.

imgaug.augmenters.contrast
--------------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_contrast.png
    :alt: dtype support for augmenters in imgaug.augmenters.contrast

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.contrast``.

imgaug.augmenters.convolutional
-------------------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_convolutional.png
    :alt: dtype support for augmenters in imgaug.augmenters.convolutional

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.convolutional``.

imgaug.augmenters.debug
-----------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_debug.png
    :alt: dtype support for augmenters in imgaug.augmenters.debug

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.debug``.

imgaug.augmenters.edges
-----------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_edges.png
    :alt: dtype support for augmenters in imgaug.augmenters.edges

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.edges``.

imgaug.augmenters.flip
----------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_flip.png
    :alt: dtype support for augmenters in imgaug.augmenters.flip

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.flip``.

imgaug.augmenters.geometric
---------------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_geometric.png
    :alt: dtype support for augmenters in imgaug.augmenters.geometric

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.geometric``.

imgaug.augmenters.imgcorruptlike
--------------------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_imgcorruptlike.png
    :alt: dtype support for augmenters in imgaug.augmenters.imgcorruptlike

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.imgcorruptlike``.

imgaug.augmenters.pillike
-------------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_pillike.png
    :alt: dtype support for augmenters in imgaug.augmenters.pillike

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.pillike``.

imgaug.augmenters.segmentation
------------------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_segmentation.png
    :alt: dtype support for augmenters in imgaug.augmenters.segmentation

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.segmentation``.

imgaug.augmenters.size
----------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_size.png
    :alt: dtype support for augmenters in imgaug.augmenters.size

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.size``.

imgaug.augmenters.weather
-------------------------

.. figure:: ../images/dtype_support/imgaug_augmenters_weather.png
    :alt: dtype support for augmenters in imgaug.augmenters.weather

    Image dtypes supported by augmenters and helper functions in
    ``imgaug.augmenters.weather``.