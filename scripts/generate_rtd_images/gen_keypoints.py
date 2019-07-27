from __future__ import print_function, division

from .utils import save, grid


def main():
    """Generate all example images for the chapter `Examples: Keypoints`
    in the documentation."""
    chapter_examples_keypoints_simple()


def chapter_examples_keypoints_simple():
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

    # ------------

    save(
        "examples_keypoints",
        "simple.jpg",
        grid([image_before, image_after], cols=2, rows=1),
        quality=90
    )


if __name__ == "__main__":
    main()
