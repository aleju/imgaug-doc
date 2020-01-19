from __future__ import print_function, division

import imgaug as ia
import imgaug.augmenters as iaa

from . import utils
from .utils import run_and_save_augseq


def main():
    chapter_augmenters_sequential()
    chapter_augmenters_someof()
    chapter_augmenters_oneof()
    chapter_augmenters_sometimes()
    chapter_augmenters_withchannels()
    chapter_augmenters_identity()
    chapter_augmenters_noop()
    chapter_augmenters_lambda()
    chapter_augmenters_assertlambda()
    chapter_augmenters_assertshape()
    chapter_augmenters_channelshuffle()
    chapter_augmenters_removecbasbyoutofimagefraction()
    chapter_augmenters_clipcbastoimageplanes()


def chapter_augmenters_sequential():
    aug = iaa.Sequential([
        iaa.Affine(translate_px={"x":-40}),
        iaa.AdditiveGaussianNoise(scale=0.2*255)
    ])
    run_and_save_augseq(
        "meta/sequential.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.Sequential([
        iaa.Affine(translate_px={"x":-40}),
        iaa.AdditiveGaussianNoise(scale=0.2*255)
    ], random_order=True)
    run_and_save_augseq(
        "meta/sequential_random_order.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*4)], cols=4, rows=4
    )


def chapter_augmenters_someof():
    aug = iaa.SomeOf(2, [
        iaa.Affine(rotate=45),
        iaa.AdditiveGaussianNoise(scale=0.2*255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ])
    run_and_save_augseq(
        "meta/someof.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.SomeOf((0, None), [
        iaa.Affine(rotate=45),
        iaa.AdditiveGaussianNoise(scale=0.2*255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ])
    run_and_save_augseq(
        "meta/someof_0_to_none.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.SomeOf(2, [
        iaa.Affine(rotate=45),
        iaa.AdditiveGaussianNoise(scale=0.2*255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ], random_order=True)
    run_and_save_augseq(
        "meta/someof_random_order.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_oneof():
    aug = iaa.OneOf([
        iaa.Affine(rotate=45),
        iaa.AdditiveGaussianNoise(scale=0.2*255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ])
    run_and_save_augseq(
        "meta/oneof.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_sometimes():
    aug = iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=2.0))
    run_and_save_augseq(
        "meta/sometimes.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2,
        seed=2
    )

    aug = iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=2.0),
        iaa.Sequential([iaa.Affine(rotate=45), iaa.Sharpen(alpha=1.0)])
    )
    run_and_save_augseq(
        "meta/sometimes_if_else.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )


def chapter_augmenters_withchannels():
    aug = iaa.WithChannels(0, iaa.Add((10, 100)))
    run_and_save_augseq(
        "meta/withchannels.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.WithChannels(0, iaa.Affine(rotate=(0, 45)))
    run_and_save_augseq(
        "meta/withchannels_affine.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_identity():
    aug = iaa.Identity()
    run_and_save_augseq(
        "meta/identity.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4)], cols=4, rows=1
    )


def chapter_augmenters_noop():
    aug = iaa.Noop()
    run_and_save_augseq(
        "meta/noop.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4)], cols=4, rows=1
    )


def chapter_augmenters_lambda():
    def img_func(images, random_state, parents, hooks):
        for img in images:
            img[::4] = 0
        return images

    def keypoint_func(keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    aug = iaa.Lambda(img_func, keypoint_func)
    run_and_save_augseq(
        "meta/lambda.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_assertlambda():
    pass


def chapter_augmenters_assertshape():
    pass


def chapter_augmenters_channelshuffle():
    fn_start = "meta/channelshuffle"

    aug = iaa.ChannelShuffle(0.35)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(8*3)], cols=8, rows=3
    )

    aug = iaa.ChannelShuffle(0.35, channels=[0, 1])
    run_and_save_augseq(
        fn_start + "_limited_channels.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(8*3)], cols=8, rows=3
    )


def chapter_augmenters_removecbasbyoutofimagefraction():
    fn_start = "meta/removecbasbyoutofimagefraction"

    image = ia.quokka_square((100, 100))
    bb = ia.BoundingBox(x1=50-25, y1=0, x2=50+25, y2=100)
    bbsoi = ia.BoundingBoxesOnImage([bb], shape=image.shape)

    # example 1
    aug = iaa.Sequential([
        iaa.Affine(translate_px={"x": (-100, 100)}),
        iaa.RemoveCBAsByOutOfImageFraction(0.5)
    ])

    images_aug, bbsois_aug = aug(images=[image] * (2*4),
                                 bounding_boxes=[bbsoi] * (2*4))
    images_drawn = [bbsoi_aug.draw_on_image(image_aug)
                    for image_aug, bbsoi_aug in zip(images_aug, bbsois_aug)]

    utils.save(
        "overview_of_augmenters",
        fn_start + ".jpg",
        utils.grid(images_drawn, cols=6, rows=3),
        quality=95
    )

    # example 2
    aug_without = iaa.Affine(translate_px={"x": 51})
    aug_with = iaa.Sequential([
        iaa.Affine(translate_px={"x": 51}),
        iaa.RemoveCBAsByOutOfImageFraction(0.5)
    ])

    image_without, bbsoi_without = aug_without(
        image=image, bounding_boxes=bbsoi)
    image_with, bbsoi_with = aug_with(
        image=image, bounding_boxes=bbsoi)

    assert len(bbsoi_without.bounding_boxes) == 1
    assert len(bbsoi_with.bounding_boxes) == 0

    images_aug = [bbsoi_without.draw_on_image(image_without),
                  bbsoi_with.draw_on_image(image_with)]

    utils.save(
        "overview_of_augmenters",
        fn_start + "_comparison.jpg",
        utils.grid(images_aug, cols=2, rows=1),
        quality=95
    )


def chapter_augmenters_clipcbastoimageplanes():
    fn_start = "meta/clipcbastoimageplanes"

    image = ia.quokka_square((100, 100))
    bb = ia.BoundingBox(x1=50-25, y1=0, x2=50+25, y2=100)
    bbsoi = ia.BoundingBoxesOnImage([bb], shape=image.shape)

    aug = iaa.Sequential([
        iaa.Affine(translate_px={"x": (-100, 100)}),
        iaa.ClipCBAsToImagePlanes()
    ])

    images_aug, bbsois_aug = aug(images=[image] * (2*4),
                                 bounding_boxes=[bbsoi] * (2*4))
    images_drawn = [bbsoi_aug.draw_on_image(image_aug)
                    for image_aug, bbsoi_aug in zip(images_aug, bbsois_aug)]

    utils.save(
        "overview_of_augmenters",
        fn_start + ".jpg",
        utils.grid(images_drawn, cols=4, rows=2),
        quality=95
    )


if __name__ == "__main__":
    main()
