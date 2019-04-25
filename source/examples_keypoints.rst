===================
Examples: Keypoints
===================

`imgaug` can handle not only images, but also keypoints/landmarks on these.
E.g. if an image is rotated during augmentation, the library can also rotate
all landmarks correspondingly.


Notebook
--------

A jupyter notebook for keypoint augmentation is available at
:ref:`page_jupyter_notebooks`. The notebooks are usually more up to date
and contain more examples than the ReadTheDocs documentation.


A simple example
----------------

The following example loads an image and places four keypoints on it.
The image is then augmented to be brighter, slightly rotated and scaled.
These augmentations are also applied to the keypoints.
The image is then shown before and after augmentation (with keypoints drawn
on it).

::

    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug.augmentables import Keypoint, KeypointsOnImage


    ia.seed(1)

    image = ia.quokka(size=(256, 256))
    kps = KeypointsOnImage([
        Keypoint(x=65, y=100),
        Keypoint(x=75, y=200),
        Keypoint(x=100, y=100),
        Keypoint(x=200, y=80)
    ], shape=image.shape)

    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect keypoints
        iaa.Affine(
            rotate=10,
            scale=(0.5, 0.7)
        ) # rotate by exactly 10deg and scale to 50-70%, affects keypoints
    ])

    # Augment keypoints and images.
    image_aug, kps_aug = seq(image=image, keypoints=kps)

    # print coordinates before/after augmentation (see below)
    # use after.x_int and after.y_int to get rounded integer coordinates
    for i in range(len(kps.keypoints)):
        before = kps.keypoints[i]
        after = kps_aug.keypoints[i]
        print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
            i, before.x, before.y, after.x, after.y)
        )

    # image with keypoints before/after augmentation (shown below)
    image_before = kps.draw_on_image(image, size=7)
    image_after = kps_aug.draw_on_image(image_aug, size=7)

Console output of the example::

    Keypoint 0: (65.00000000, 100.00000000) -> (97.86113503, 107.69632182)
    Keypoint 1: (75.00000000, 200.00000000) -> (93.93710117, 160.01366917)
    Keypoint 2: (100.00000000, 100.00000000) -> (115.85492750, 110.86911292)
    Keypoint 3: (200.00000000, 80.00000000) -> (169.07878659, 109.65206321)

.. figure:: ../images/examples_keypoints/simple.jpg
    :alt: Simple keypoint augmentation example

    Image with keypoints, before (left) and after (right)
    augmentation. Keypoints are shown in green and drawn in after the
    augmentation process.
