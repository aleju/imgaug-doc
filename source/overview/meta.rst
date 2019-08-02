***************
augmenters.meta
***************

Sequential
----------

List augmenter that may contain other augmenters to apply in sequence
or random order.

**Example.**
Apply in predefined order::

    aug = iaa.Sequential([
        iaa.Affine(translate_px={"x":-40}),
        iaa.AdditiveGaussianNoise(scale=0.1*255)
    ])

.. figure:: ../../images/overview_of_augmenters/meta/sequential.jpg
    :alt: Sequential

**Example.**
Apply in random order (note that the order is sampled once per batch and then
the same for all images within the batch)::

    aug = iaa.Sequential([
          iaa.Affine(translate_px={"x":-40}),
          iaa.AdditiveGaussianNoise(scale=0.1*255)
    ], random_order=True)

.. figure:: ../../images/overview_of_augmenters/meta/sequential_random_order.jpg
    :alt: Sequential with random order


SomeOf
------

List augmenter that applies only some of its children to images.

**Example.**
Apply two of four given augmenters::

    aug = iaa.SomeOf(2, [
        iaa.Affine(rotate=45),
        iaa.AdditiveGaussianNoise(scale=0.2*255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ])

.. figure:: ../../images/overview_of_augmenters/meta/someof.jpg
    :alt: SomeOf

**Example.**
Apply ``0`` to ``<max>`` given augmenters (where ``<max>`` is automatically
replaced with the number of children)::

    aug = iaa.SomeOf((0, None), [
        iaa.Affine(rotate=45),
        iaa.AdditiveGaussianNoise(scale=0.2*255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ])

.. figure:: ../../images/overview_of_augmenters/meta/someof_0_to_none.jpg
    :alt: SomeOf 0 to None

**Example.**
Pick two of four given augmenters and apply them in random order::

    aug = iaa.SomeOf(2, [
        iaa.Affine(rotate=45),
        iaa.AdditiveGaussianNoise(scale=0.2*255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ], random_order=True)

.. figure:: ../../images/overview_of_augmenters/meta/someof_random_order.jpg
    :alt: SomeOf random order


OneOf
-----

Augmenter that always executes exactly one of its children.

**Example.**
Apply one of four augmenters to each image::

    aug = iaa.OneOf([
        iaa.Affine(rotate=45),
        iaa.AdditiveGaussianNoise(scale=0.2*255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ])

.. figure:: ../../images/overview_of_augmenters/meta/oneof.jpg
    :alt: OneOf


Sometimes
---------

Augment only p percent of all images with one or more augmenters.

**Example.**
Apply gaussian blur to about 50% of all images::

    aug = iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=2.0))

.. figure:: ../../images/overview_of_augmenters/meta/sometimes.jpg
    :alt: Sometimes

**Example.**
Apply gaussian blur to about 50% of all images. Apply a mixture of affine
rotations and sharpening to the other 50%. ::

    aug = iaa.Sometimes(
          0.5,
          iaa.GaussianBlur(sigma=2.0),
          iaa.Sequential([iaa.Affine(rotate=45), iaa.Sharpen(alpha=1.0)])
      )

.. figure:: ../../images/overview_of_augmenters/meta/sometimes_if_else.jpg
    :alt: Sometimes if else


WithChannels
------------

Apply child augmenters to specific channels.

**Example.**
Increase each pixel's R-value (redness) by ``10`` to ``100``::

    aug = iaa.WithChannels(0, iaa.Add((10, 100)))

.. figure:: ../../images/overview_of_augmenters/meta/withchannels.jpg
    :alt: WithChannels

**Example.**
Rotate each image's red channel by ``0`` to ``45`` degrees::

    aug = iaa.WithChannels(0, iaa.Affine(rotate=(0, 45)))

.. figure:: ../../images/overview_of_augmenters/meta/withchannels_affine.jpg
    :alt: WithChannels + Affine


Noop
----

Augmenter that never changes input images ("no operation"). ::

    aug = iaa.Noop()

.. figure:: ../../images/overview_of_augmenters/meta/noop.jpg
    :alt: Noop



Lambda
------

Augmenter that calls a lambda function for each batch of input image.

**Example.**
Replace in every image each fourth row with black pixels::

    def img_func(images, random_state, parents, hooks):
        for img in images:
            img[::4] = 0
        return images

    def keypoint_func(keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    aug = iaa.Lambda(img_func, keypoint_func)

.. figure:: ../../images/overview_of_augmenters/meta/lambda.jpg
    :alt: Lambda


AssertLambda
------------

Augmenter that runs an assert on each batch of input images
using a lambda function as condition.

TODO examples


AssertShape
-----------

Augmenter to make assumptions about the shape of input image(s)
and keypoints.

**Example.**
Check if each image in a batch has shape ``32x32x3``, otherwise raise an
exception::

    seq = iaa.Sequential([
        iaa.AssertShape((None, 32, 32, 3)),
        iaa.Fliplr(0.5) # only executed if shape matches
    ])

**Example.**
Check if each image in a batch has a height in the range ``32<=x<64``,
a width of exactly ``64`` and either ``1`` or ``3`` channels::

    seq = iaa.Sequential([
        iaa.AssertShape((None, (32, 64), 32, [1, 3])),
        iaa.Fliplr(0.5)
    ])


ChannelShuffle
--------------

Randomize the order of channels in input images.

**Example.**
Shuffle all channels of 35% of all images::

    import imgaug.augmenters as iaa
    aug = iaa.ChannelShuffle(0.35)

.. figure:: ../../images/overview_of_augmenters/meta/channelshuffle.jpg
    :alt: ChannelShuffle

**Example.**
Shuffle only channels ``0`` and ``1`` of 35% of all images. As the new
channel orders ``0, 1`` and ``1, 0`` are both valid outcomes of the
shuffling, it means that for ``0.35 * 0.5 = 0.175`` or 17.5% of all images
the order of channels ``0`` and ``1`` is inverted. ::

    aug = iaa.ChannelShuffle(0.35, channels=[0, 1])

.. figure:: ../../images/overview_of_augmenters/meta/channelshuffle_limited_channels.jpg
    :alt: ChannelShuffle

