from __future__ import print_function, division

import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import run_and_save_augseq


def main():
    chapter_augmenters_superpixels()


def chapter_augmenters_superpixels():
    aug = iaa.Superpixels(p_replace=0.5, n_segments=64)
    run_and_save_augseq(
        "segmentation/superpixels_50_64.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.Superpixels(p_replace=(0.1, 1.0), n_segments=(16, 128))
    run_and_save_augseq(
        "segmentation/superpixels.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    #ps = [1/8*i for i in range(8)]
    ps = np.linspace(0, 1.0, num=8)
    run_and_save_augseq(
        "segmentation/superpixels_vary_p.jpg",
        [iaa.Superpixels(p_replace=p, n_segments=64) for p in ps],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1
    )

    ns = [16*i for i in range(1, 9)]
    run_and_save_augseq(
        "segmentation/superpixels_vary_n.jpg",
        [iaa.Superpixels(p_replace=1.0, n_segments=n) for n in ns],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1
    )


if __name__ == "__main__":
    main()
