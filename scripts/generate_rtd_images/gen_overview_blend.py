from __future__ import print_function, division

import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap

from .utils import run_and_save_augseq


def main():
    chapter_augmenters_alpha()
    chapter_augmenters_alphaelementwise()
    chapter_augmenters_simplexnoisealpha()
    chapter_augmenters_frequencynoisealpha()


def chapter_augmenters_alpha():
    fn_start = "blend/alpha"
    aug_cls = iaa.Alpha

    aug = aug_cls(0.5, iaa.Grayscale(1.0))
    run_and_save_augseq(
        fn_start + "_050_grayscale.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4)], cols=4, rows=1)

    aug = aug_cls((0.0, 1.0), iaa.Grayscale(1.0))
    run_and_save_augseq(
        fn_start + "_uniform_factor.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls(
        (0.0, 1.0),
        iaa.Affine(rotate=(-20, 20)),
        per_channel=0.5)
    run_and_save_augseq(
        fn_start + "_affine_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls(
        (0.0, 1.0),
        first=iaa.Add(100),
        second=iaa.Multiply(0.2))
    run_and_save_augseq(
        fn_start + "_two_branches.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls([0.25, 0.75], iaa.MedianBlur(13))
    run_and_save_augseq(
        fn_start + "_with_choice.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_alphaelementwise():
    fn_start = "blend/alphaelementwise"
    aug_cls = iaa.AlphaElementwise

    aug = aug_cls(0.5, iaa.Grayscale(1.0))
    run_and_save_augseq(
        fn_start + "_050_grayscale.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4)], cols=4, rows=1)

    aug = aug_cls((0.0, 1.0), iaa.Grayscale(1.0))
    run_and_save_augseq(
        fn_start + "_uniform_factor.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls(
        (0.0, 1.0),
        iaa.Affine(rotate=(-20, 20)),
        per_channel=0.5)
    run_and_save_augseq(
        fn_start + "_affine_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls(
        (0.0, 1.0),
        first=iaa.Add(100),
        second=iaa.Multiply(0.2))
    run_and_save_augseq(
        fn_start + "_two_branches.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls([0.25, 0.75], iaa.MedianBlur(13))
    run_and_save_augseq(
        fn_start + "_with_choice.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_simplexnoisealpha():
    fn_start = "blend/simplexnoisealpha"

    aug = iaa.SimplexNoiseAlpha(iaa.EdgeDetect(1.0))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.SimplexNoiseAlpha(
        iaa.EdgeDetect(1.0),
        upscale_method="nearest")
    run_and_save_augseq(
        fn_start + "_nearest.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)

    aug = iaa.SimplexNoiseAlpha(
        iaa.EdgeDetect(1.0),
        upscale_method="linear")
    run_and_save_augseq(
        fn_start + "_linear.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)

    aug = iaa.SimplexNoiseAlpha(
        iaa.EdgeDetect(1.0),
        sigmoid_thresh=iap.Normal(10.0, 5.0))
    run_and_save_augseq(
        fn_start + "_sigmoid_thresh_normal.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_frequencynoisealpha():
    fn_start = "blend/frequencynoisealpha"

    aug = iaa.FrequencyNoiseAlpha(first=iaa.EdgeDetect(1.0))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.FrequencyNoiseAlpha(
        first=iaa.EdgeDetect(1.0),
        upscale_method="nearest")
    run_and_save_augseq(
        fn_start + "_nearest.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)

    aug = iaa.FrequencyNoiseAlpha(
        first=iaa.EdgeDetect(1.0),
        upscale_method="linear")
    run_and_save_augseq(
        fn_start + "_linear.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)

    aug = iaa.FrequencyNoiseAlpha(
        first=iaa.EdgeDetect(1.0),
        upscale_method="linear",
        exponent=-2,
        sigmoid=False)
    run_and_save_augseq(
        fn_start + "_clouds.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.FrequencyNoiseAlpha(
        first=iaa.EdgeDetect(1.0),
        sigmoid_thresh=iap.Normal(10.0, 5.0))
    run_and_save_augseq(
        fn_start + "_sigmoid_thresh_normal.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


if __name__ == "__main__":
    main()
