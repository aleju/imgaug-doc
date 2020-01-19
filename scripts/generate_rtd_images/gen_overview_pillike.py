from __future__ import print_function, division

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import run_and_save_augseq


def main():
    chapter_augmenters_solarize()
    chapter_augmenters_posterize()
    chapter_augmenters_equalize()
    chapter_augmenters_autocontrast()
    chapter_augmenters_enhancecolor()
    chapter_augmenters_enhancecontrast()
    chapter_augmenters_enhancebrightness()
    chapter_augmenters_enhancesharpness()
    chapter_augmenters_filterblur()
    chapter_augmenters_filtersmooth()
    chapter_augmenters_filtersmoothmore()
    chapter_augmenters_filteredgeenhance()
    chapter_augmenters_filteredgeenhancemore()
    chapter_augmenters_filterfindedges()
    chapter_augmenters_filtercontour()
    chapter_augmenters_filteremboss()
    chapter_augmenters_filtersharpen()
    chapter_augmenters_filterdetail()
    chapter_augmenters_affine()


def chapter_augmenters_solarize():
    fn_start = "pillike/solarize"

    aug = iaa.Solarize(0.5, threshold=(32, 128))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*3)], cols=4, rows=3)


def chapter_augmenters_posterize():
    pass


def chapter_augmenters_equalize():
    fn_start = "pillike/equalize"

    aug = iaa.pillike.Equalize()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_autocontrast():
    fn_start = "pillike/autocontrast"

    aug = iaa.pillike.Autocontrast()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.pillike.Autocontrast((10, 20), per_channel=True)
    run_and_save_augseq(
        fn_start + "_vary_cutoff.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*3)], cols=4, rows=3)


def chapter_augmenters_enhancecolor():
    fn_start = "pillike/enhancecolor"

    aug = iaa.pillike.EnhanceColor()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*3)], cols=4, rows=3)


def chapter_augmenters_enhancecontrast():
    fn_start = "pillike/enhancecontrast"

    aug = iaa.pillike.EnhanceContrast()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*3)], cols=4, rows=3)


def chapter_augmenters_enhancebrightness():
    fn_start = "pillike/enhancebrightness"

    aug = iaa.pillike.EnhanceBrightness()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*3)], cols=4, rows=3)


def chapter_augmenters_enhancesharpness():
    fn_start = "pillike/enhancesharpness"

    aug = iaa.pillike.EnhanceSharpness()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*3)], cols=4, rows=3)


def chapter_augmenters_filterblur():
    fn_start = "pillike/filterblur"

    aug = iaa.pillike.FilterBlur()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)


def chapter_augmenters_filtersmooth():
    fn_start = "pillike/filtersmooth"

    aug = iaa.pillike.FilterSmooth()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)


def chapter_augmenters_filtersmoothmore():
    fn_start = "pillike/filtersmoothmore"

    aug = iaa.pillike.FilterSmoothMore()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)


def chapter_augmenters_filteredgeenhance():
    fn_start = "pillike/filteredgeenhance"

    aug = iaa.pillike.FilterEdgeEnhance()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)


def chapter_augmenters_filteredgeenhancemore():
    fn_start = "pillike/filteredgeenhancemore"

    aug = iaa.pillike.FilterEdgeEnhanceMore()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)


def chapter_augmenters_filterfindedges():
    fn_start = "pillike/filterfindedges"

    aug = iaa.pillike.FilterFindEdges()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)


def chapter_augmenters_filtercontour():
    fn_start = "pillike/filtercontour"

    aug = iaa.pillike.FilterContour()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)


def chapter_augmenters_filteremboss():
    fn_start = "pillike/filteremboss"

    aug = iaa.pillike.FilterEmboss()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)


def chapter_augmenters_filtersharpen():
    fn_start = "pillike/filtersharpen"

    aug = iaa.pillike.FilterSharpen()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)


def chapter_augmenters_filterdetail():
    fn_start = "pillike/filterdetail"

    aug = iaa.pillike.FilterDetail()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)


def chapter_augmenters_affine():
    fn_start = "pillike/affine"

    aug = iaa.pillike.Affine(scale={"x": (0.8, 1.2), "y": (0.5, 1.5)})
    run_and_save_augseq(
        fn_start + "_scale.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.pillike.Affine(translate_px={"x": 0, "y": [-10, 10]},
                             fillcolor=128)
    run_and_save_augseq(
        fn_start + "_translate_fillcolor.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.pillike.Affine(rotate=(-20, 20), fillcolor=(0, 256))
    run_and_save_augseq(
        fn_start + "_rotate_fillcolor.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


if __name__ == "__main__":
    main()
