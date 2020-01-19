*************************
augmenters.imgcorruptlike
*************************

GaussianNoise
-------------

Wrapper around :func:`~imagecorruptions.corruptions.gaussian_noise`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.GaussianNoise`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.gaussian_noise`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.GaussianNoise(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/gaussiannoise.jpg
    :alt: GaussianNoise


ShotNoise
---------

Wrapper around :func:`~imagecorruptions.corruptions.shot_noise`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.ShotNoise`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.shot_noise`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.ShotNoise(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/shotnoise.jpg
    :alt: ShotNoise


ImpulseNoise
------------

Wrapper around :func:`~imagecorruptions.corruptions.impulse_noise`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.ImpulseNoise`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.impulse_noise`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.ImpulseNoise(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/impulsenoise.jpg
    :alt: ImpulseNoise


SpeckleNoise
------------

Wrapper around :func:`~imagecorruptions.corruptions.speckle_noise`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.SpeckleNoise`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.speckle_noise`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.SpeckleNoise(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/specklenoise.jpg
    :alt: SpeckleNoise


GaussianBlur
------------

Wrapper around :func:`~imagecorruptions.corruptions.gaussian_blur`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.GaussianBlur`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.gaussian_blur`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.GaussianBlur(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/gaussianblur.jpg
    :alt: GaussianBlur


GlassBlur
------------

Wrapper around :func:`~imagecorruptions.corruptions.glass_blur`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.GlassBlur`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.glass_blur`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.GlassBlur(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/glassblur.jpg
    :alt: GlassBlur


DefocusBlur
------------

Wrapper around :func:`~imagecorruptions.corruptions.defocus_blur`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.DefocusBlur`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.defocus_blur`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.DefocusBlur(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/defocusblur.jpg
    :alt: DefocusBlur


MotionBlur
------------

Wrapper around :func:`~imagecorruptions.corruptions.motion_blur`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.MotionBlur`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.motion_blur`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.MotionBlur(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/motionblur.jpg
    :alt: MotionBlur


ZoomBlur
------------

Wrapper around :func:`~imagecorruptions.corruptions.zoom_blur`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.ZoomBlur`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.zoom_blur`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.ZoomBlur(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/zoomblur.jpg
    :alt: ZoomBlur


Fog
------------

Wrapper around :func:`~imagecorruptions.corruptions.fog`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.Fog`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.fog`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.Fog(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/fog.jpg
    :alt: Fog


Frost
------------

Wrapper around :func:`~imagecorruptions.corruptions.frost`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.Frost`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.frost`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.Frost(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/frost.jpg
    :alt: Frost


Snow
------------

Wrapper around :func:`~imagecorruptions.corruptions.snow`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.Snow`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.snow`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.Snow(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/snow.jpg
    :alt: Snow


Spatter
------------

Wrapper around :func:`~imagecorruptions.corruptions.spatter`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.Spatter`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.spatter`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.Spatter(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/spatter.jpg
    :alt: Spatter


Contrast
------------

Wrapper around :func:`~imagecorruptions.corruptions.contrast`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.Contrast`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.contrast`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.Contrast(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/contrast.jpg
    :alt: Contrast


Brightness
------------

Wrapper around :func:`~imagecorruptions.corruptions.brightness`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.Brightness`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.brightness`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.Brightness(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/brightness.jpg
    :alt: Brightness


Saturate
------------

Wrapper around :func:`~imagecorruptions.corruptions.saturate`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.Saturate`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.saturate`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.Saturate(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/saturate.jpg
    :alt: Saturate


JpegCompression
---------------

Wrapper around :func:`~imagecorruptions.corruptions.jpeg_compression`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.JpegCompression`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.jpeg_compression`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.JpegCompression(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/jpegcompression.jpg
    :alt: JpegCompression


Pixelate
------------

Wrapper around :func:`~imagecorruptions.corruptions.jpeg_compression`.

.. note::

    This augmenter only affects images. Other data is not changed.

Wrapper around :func:`~imagecorruptions.corruptions.pixelate`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.Pixelate`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.pixelate`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.Pixelate(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/pixelate.jpg
    :alt: Pixelate


ElasticTransform
----------------

Wrapper around :func:`~imagecorruptions.corruptions.elastic_transform`.

.. note::

    This augmenter only affects images. Other data is not changed.

API link: :class:`~imgaug.augmenters.imgcorruptlike.ElasticTransform`

**Example.**
Create an augmenter around
:func:`~imagecorruptions.corruptions.elastic_transform`.
Apply it to images using e.g. ``aug(images=[image1, image2, ...])``::

    import imgaug.augmenters as iaa
    aug = iaa.imgcorruptlike.ElasticTransform(severity=2)

.. figure:: ../../images/overview_of_augmenters/imgcorruptlike/elastictransform.jpg
    :alt: ElasticTransform
