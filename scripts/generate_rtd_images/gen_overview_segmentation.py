from __future__ import print_function, division

import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import run_and_save_augseq


def main():
    chapter_augmenters_superpixels()
    chapter_augmenters_voronoi()
    chapter_augmenters_uniformvoronoi()
    chapter_augmenters_regulargridvoronoi()
    chapter_augmenters_relativeregulargridvoronoi()


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


def chapter_augmenters_voronoi():
    fn_start = "segmentation/voronoi"
    aug_cls = iaa.Voronoi

    points_sampler = iaa.RegularGridPointsSampler(n_cols=20, n_rows=40)
    aug = aug_cls(points_sampler)
    run_and_save_augseq(
        fn_start + "_regular_grid.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2
    )

    points_sampler = iaa.DropoutPointsSampler(
        iaa.RelativeRegularGridPointsSampler(
            n_cols_frac=(0.05, 0.2),
            n_rows_frac=0.1),
        0.2)
    aug = aug_cls(points_sampler)
    run_and_save_augseq(
        fn_start + "_complex.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2
    )


def chapter_augmenters_uniformvoronoi():
    fn_start = "segmentation/uniformvoronoi"
    aug_cls = iaa.UniformVoronoi

    aug = aug_cls((100, 500))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2
    )

    aug = aug_cls(250, p_replace=0.9, max_size=None)
    run_and_save_augseq(
        fn_start + "_p_replace_max_size.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2
    )


def chapter_augmenters_regulargridvoronoi():
    fn_start = "segmentation/regulargridvoronoi"
    aug_cls = iaa.RegularGridVoronoi

    aug = aug_cls(10, 20)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2
    )

    aug = aug_cls(
        (10, 30), 20, p_drop_points=0.0, p_replace=0.9, max_size=None)
    run_and_save_augseq(
        fn_start + "_p_replace_max_size.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2
    )


def chapter_augmenters_relativeregulargridvoronoi():
    fn_start = "segmentation/relativeregulargridvoronoi"
    aug_cls = iaa.RelativeRegularGridVoronoi

    aug = aug_cls(0.1, 0.25)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2
    )

    aug = aug_cls(
        (0.03, 0.1), 0.1, p_drop_points=0.0, p_replace=0.9, max_size=512)
    run_and_save_augseq(
        fn_start + "_p_replace_max_size.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2
    )


if __name__ == "__main__":
    main()
