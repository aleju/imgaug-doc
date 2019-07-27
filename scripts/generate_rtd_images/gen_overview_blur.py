from __future__ import print_function, division

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import run_and_save_augseq


def main():
    chapter_augmenters_gaussianblur()
    chapter_augmenters_averageblur()
    chapter_augmenters_medianblur()


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


if __name__ == "__main__":
    main()
