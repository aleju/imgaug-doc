from __future__ import print_function, division

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import run_and_save_augseq


def main():
    chapter_augmenters_sequential()
    chapter_augmenters_someof()
    chapter_augmenters_oneof()
    chapter_augmenters_sometimes()
    chapter_augmenters_withchannels()
    chapter_augmenters_noop()
    chapter_augmenters_lambda()
    chapter_augmenters_assertlambda()
    chapter_augmenters_assertshape()


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
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
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


def chapter_augmenters_noop():
    aug = iaa.Noop()
    run_and_save_augseq(
        "meta/noop.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
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


if __name__ == "__main__":
    main()
