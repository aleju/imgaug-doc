from __future__ import print_function, division

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import run_and_save_augseq


def main():
    chapter_augmenters_fliplr()
    chapter_augmenters_flipud()


def chapter_augmenters_fliplr():
    aug = iaa.Fliplr(0.5)
    run_and_save_augseq(
        "flip/fliplr.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )


def chapter_augmenters_flipud():
    aug = iaa.Flipud(0.5)
    run_and_save_augseq(
        "flip/flipud.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )


if __name__ == "__main__":
    main()
