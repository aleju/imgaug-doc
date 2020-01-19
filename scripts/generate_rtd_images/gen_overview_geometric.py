from __future__ import print_function, division

import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import run_and_save_augseq, checkerboard, save


def main():
    chapter_augmenters_affine()
    chapter_augmenters_scalex()
    chapter_augmenters_scaley()
    chapter_augmenters_translatex()
    chapter_augmenters_translatey()
    chapter_augmenters_rotate()
    chapter_augmenters_shearx()
    chapter_augmenters_sheary()
    chapter_augmenters_piecewiseaffine()
    chapter_augmenters_perspectivetransform()
    chapter_augmenters_elastictransformation()
    chapter_augmenters_rot90()
    chapter_augmenters_withpolarwarping()
    chapter_augmenters_jigsaw()


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


def chapter_augmenters_scalex():
    fn_start = "geometric/scalex"

    image = ia.quokka(size=(128, 128))

    aug = iaa.ScaleX((0.5, 1.5))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [image for _ in range(4*1)], cols=4, rows=1)


def chapter_augmenters_scaley():
    fn_start = "geometric/scaley"

    image = ia.quokka(size=(128, 128))

    aug = iaa.ScaleY((0.5, 1.5))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [image for _ in range(4*1)], cols=4, rows=1)


def chapter_augmenters_translatex():
    fn_start = "geometric/translatex"

    image = ia.quokka(size=(128, 128))

    aug = iaa.TranslateX(px=(-20, 20))
    run_and_save_augseq(
        fn_start + "_absolute.jpg", aug,
        [image for _ in range(4*1)], cols=4, rows=1)

    aug = iaa.TranslateX(percent=(-0.1, 0.1))
    run_and_save_augseq(
        fn_start + "_relative.jpg", aug,
        [image for _ in range(4*1)], cols=4, rows=1)


def chapter_augmenters_translatey():
    fn_start = "geometric/translatey"

    image = ia.quokka(size=(128, 128))

    aug = iaa.TranslateY(px=(-20, 20))
    run_and_save_augseq(
        fn_start + "_absolute.jpg", aug,
        [image for _ in range(4*1)], cols=4, rows=1)

    aug = iaa.TranslateY(percent=(-0.1, 0.1))
    run_and_save_augseq(
        fn_start + "_relative.jpg", aug,
        [image for _ in range(4*1)], cols=4, rows=1)


def chapter_augmenters_rotate():
    fn_start = "geometric/rotate"

    image = ia.quokka(size=(128, 128))

    aug = iaa.Rotate((-45, 45))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [image for _ in range(4*1)], cols=4, rows=1)


def chapter_augmenters_shearx():
    fn_start = "geometric/shearx"

    image = ia.quokka(size=(128, 128))

    aug = iaa.ShearX((-20, 20))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [image for _ in range(4*1)], cols=4, rows=1)


def chapter_augmenters_sheary():
    fn_start = "geometric/sheary"

    image = ia.quokka(size=(128, 128))

    aug = iaa.ShearY((0.5, 1.5))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [image for _ in range(4*1)], cols=4, rows=1)


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


def chapter_augmenters_rot90():
    fn_start = "geometric/rot90"

    image = ia.quokka(size=(128, 128))
    image = image[:, :-40]

    save(
        "overview_of_augmenters",
        fn_start + "_base_image.jpg",
        image,
        quality=90
    )

    aug = iaa.Rot90(1)
    run_and_save_augseq(
        fn_start + "_k_is_1.jpg", aug,
        [image for _ in range(4*1)], cols=4, rows=1)

    aug = iaa.Rot90([1, 3])
    run_and_save_augseq(
        fn_start + "_k_is_1_or_3.jpg", aug,
        [image for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.Rot90((1, 3))
    run_and_save_augseq(
        fn_start + "_k_is_1_or_2_or_3.jpg", aug,
        [image for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.Rot90((1, 3), keep_size=False)
    run_and_save_augseq(
        fn_start + "_keep_size_false.jpg", aug,
        [image for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_withpolarwarping():
    fn_start = "geometric/withpolarwarping"

    image = ia.quokka(size=(128, 128))

    aug = iaa.WithPolarWarping(iaa.CropAndPad(percent=(-0.1, 0.1)))
    run_and_save_augseq(
        fn_start + "_cropandpad.jpg", aug,
        [image for _ in range(4*1)], cols=4, rows=1)

    aug = iaa.WithPolarWarping(
        iaa.Affine(
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-35, 35),
            scale=(0.8, 1.2),
            shear={"x": (-15, 15), "y": (-15, 15)}
        )
    )
    run_and_save_augseq(
        fn_start + "_affine.jpg", aug,
        [image for _ in range(4*4)], cols=4, rows=4)

    aug = iaa.WithPolarWarping(iaa.AveragePooling((2, 8)))
    run_and_save_augseq(
        fn_start + "_averagepooling.jpg", aug,
        [image for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_jigsaw():
    fn_start = "geometric/jigsaw"

    image = ia.quokka(size=(128, 128))

    aug = iaa.Jigsaw(nb_rows=10, nb_cols=10)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [image for _ in range(4*1)], cols=4, rows=1)

    aug = iaa.Jigsaw(nb_rows=(1, 4), nb_cols=(1, 4))
    run_and_save_augseq(
        fn_start + "_random_grid.jpg", aug,
        [image for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.Jigsaw(nb_rows=10, nb_cols=10, max_steps=(1, 5))
    run_and_save_augseq(
        fn_start + "_random_max_steps.jpg", aug,
        [image for _ in range(4*2)], cols=4, rows=2)


if __name__ == "__main__":
    main()
