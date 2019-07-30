from __future__ import print_function, division

import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import run_and_save_augseq, checkerboard


def main():
    chapter_augmenters_affine()
    chapter_augmenters_piecewiseaffine()
    chapter_augmenters_perspectivetransform()
    chapter_augmenters_elastictransformation()


def chapter_augmenters_affine():
    aug = iaa.Affine(scale=(0.5, 1.5))
    run_and_save_augseq(
        "geometric/affine_scale.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})
    run_and_save_augseq(
        "geometric/affine_scale_independently.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
    run_and_save_augseq(
        "geometric/affine_translate_percent.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.Affine(translate_px={"x": (-20, 20), "y": (-20, 20)})
    run_and_save_augseq(
        "geometric/affine_translate_px.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.Affine(rotate=(-45, 45))
    run_and_save_augseq(
        "geometric/affine_rotate.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.Affine(shear=(-16, 16))
    run_and_save_augseq(
        "geometric/affine_shear.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.Affine(translate_percent={"x": -0.20}, mode=ia.ALL, cval=(0, 255))
    run_and_save_augseq(
        "geometric/affine_fill.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )


def chapter_augmenters_piecewiseaffine():
    aug = iaa.PiecewiseAffine(scale=(0.01, 0.05))
    run_and_save_augseq(
        "geometric/piecewiseaffine.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2)

    aug = iaa.PiecewiseAffine(scale=(0.01, 0.05))
    run_and_save_augseq(
        "geometric/piecewiseaffine_checkerboard.jpg", aug,
        [checkerboard(size=(128, 128)) for _ in range(8)], cols=4, rows=2)

    scales = np.linspace(0.0, 0.3, num=8)
    run_and_save_augseq(
        "geometric/piecewiseaffine_vary_scales.jpg",
        [iaa.PiecewiseAffine(scale=scale) for scale in scales],
        [checkerboard(size=(128, 128)) for _ in range(8)], cols=8, rows=1)

    gridvals = [2, 4, 6, 8, 10, 12, 14, 16]
    run_and_save_augseq(
        "geometric/piecewiseaffine_vary_grid.jpg",
        [iaa.PiecewiseAffine(scale=0.05, nb_rows=g, nb_cols=g) for g in gridvals],
        [checkerboard(size=(128, 128)) for _ in range(8)], cols=8, rows=1)


def chapter_augmenters_perspectivetransform():
    fn_start = "geometric/perspectivetransform"

    aug = iaa.PerspectiveTransform(scale=(0.01, 0.15))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*3)], cols=4, rows=3)

    aug = iaa.PerspectiveTransform(scale=(0.01, 0.15), keep_size=False)
    run_and_save_augseq(
        fn_start + "_keep_size_false.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*3)], cols=4, rows=3)


def chapter_augmenters_elastictransformation():
    aug = iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)
    run_and_save_augseq(
        "geometric/elastictransformations.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2)

    alphas = np.linspace(0.0, 5.0, num=8)
    run_and_save_augseq(
        "geometric/elastictransformations_vary_alpha.jpg",
        [iaa.ElasticTransformation(alpha=alpha, sigma=0.25) for alpha in alphas],
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=8, rows=1)

    sigmas = np.linspace(0.01, 1.0, num=8)
    run_and_save_augseq(
        "geometric/elastictransformations_vary_sigmas.jpg",
        [iaa.ElasticTransformation(alpha=2.5, sigma=sigma) for sigma in sigmas],
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=8, rows=1)


if __name__ == "__main__":
    main()
