from __future__ import print_function, division

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import run_and_save_augseq


def main():
    chapter_augmenters_gaussianblur()
    chapter_augmenters_averageblur()
    chapter_augmenters_medianblur()
    chapter_augmenters_bilateralblur()
    chapter_augmenters_motionblur()
    chapter_augmenters_meanshiftblur()


def chapter_augmenters_gaussianblur():
    aug = iaa.GaussianBlur(sigma=(0.0, 3.0))
    run_and_save_augseq(
        "blur/gaussianblur.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(16)], cols=4, rows=4)


def chapter_augmenters_averageblur():
    aug = iaa.AverageBlur(k=(2, 11))
    run_and_save_augseq(
        "blur/averageblur.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(16)], cols=4, rows=4)

    aug = iaa.AverageBlur(k=((5, 11), (1, 3)))
    run_and_save_augseq(
        "blur/averageblur_mixed.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(16)], cols=4, rows=4)


def chapter_augmenters_medianblur():
    aug = iaa.MedianBlur(k=(3, 11))
    run_and_save_augseq(
        "blur/medianblur.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(16)], cols=4, rows=4)

    # median doesnt support this
    #aug = iaa.MedianBlur(k=((5, 11), (1, 3)))
    #run_and_save_augseq(
    #    "medianblur_mixed.jpg", aug,
    #    [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2,
    #    quality=75
    #)


def chapter_augmenters_bilateralblur():
    fn_start = "blur/bilateralblur"

    aug = iaa.BilateralBlur(
        d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(16)], cols=4, rows=4)


def chapter_augmenters_motionblur():
    fn_start = "blur/motionblur"

    aug = iaa.MotionBlur(k=15)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(16)], cols=4, rows=4)

    aug = iaa.MotionBlur(k=15, angle=[-45, 45])
    run_and_save_augseq(
        fn_start + "_angle.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(16)], cols=4, rows=4)


def chapter_augmenters_meanshiftblur():
    fn_start = "blur/meanshiftblur"

    aug = iaa.MeanShiftBlur()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(16)], cols=4, rows=4)


if __name__ == "__main__":
    main()
