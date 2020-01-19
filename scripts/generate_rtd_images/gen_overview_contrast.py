from __future__ import print_function, division

import cv2

import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap

from .utils import run_and_save_augseq


def main():
    chapter_augmenters_gammacontrast()
    chapter_augmenters_sigmoidcontrast()
    chapter_augmenters_logcontrast()
    chapter_augmenters_linearcontrast()
    chapter_augmenters_allchannelsclahe()
    chapter_augmenters_clahe()
    chapter_augmenters_allchannelshistogramequalization()
    chapter_augmenters_histogramequalization()


def chapter_augmenters_gammacontrast():
    fn_start = "contrast/gammacontrast"

    aug = iaa.GammaContrast((0.5, 2.0))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.GammaContrast((0.5, 2.0), per_channel=True)
    run_and_save_augseq(
        fn_start + "_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_sigmoidcontrast():
    fn_start = "contrast/sigmoidcontrast"

    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.SigmoidContrast(
        gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True)
    run_and_save_augseq(
        fn_start + "_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_logcontrast():
    fn_start = "contrast/logcontrast"

    aug = iaa.LogContrast(gain=(0.4, 1.6))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.LogContrast(gain=(0.4, 1.6), per_channel=True)
    run_and_save_augseq(
        fn_start + "_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_linearcontrast():
    fn_start = "contrast/linearcontrast"

    aug = iaa.LinearContrast((0.6, 1.4))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.LinearContrast((0.6, 1.4), per_channel=True)
    run_and_save_augseq(
        fn_start + "_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_allchannelsclahe():
    fn_start = "contrast/allchannelsclahe"

    aug = iaa.AllChannelsCLAHE()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.AllChannelsCLAHE(clip_limit=(1, 10))
    run_and_save_augseq(
        fn_start + "_random_clip_limit.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True)
    run_and_save_augseq(
        fn_start + "_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_clahe():
    fn_start = "contrast/clahe"

    aug = iaa.CLAHE()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.CLAHE(clip_limit=(1, 10))
    run_and_save_augseq(
        fn_start + "_clip_limit.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.CLAHE(tile_grid_size_px=(3, 21))
    run_and_save_augseq(
        fn_start + "_grid_sizes_uniform.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.CLAHE(
        tile_grid_size_px=iap.Discretize(iap.Normal(loc=7, scale=2)),
        tile_grid_size_px_min=3)
    run_and_save_augseq(
        fn_start + "_grid_sizes_gaussian.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.CLAHE(tile_grid_size_px=((3, 21), [3, 5, 7]))
    run_and_save_augseq(
        fn_start + "_grid_sizes.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.CLAHE(
        from_colorspace=iaa.CLAHE.BGR,
        to_colorspace=iaa.CLAHE.HSV)
    quokka_bgr = cv2.cvtColor(ia.quokka(size=(128, 128)), cv2.COLOR_RGB2BGR)
    run_and_save_augseq(
        fn_start + "_bgr_to_hsv.jpg", aug,
        [quokka_bgr for _ in range(4*2)], cols=4, rows=2,
        image_colorspace="BGR")


def chapter_augmenters_allchannelshistogramequalization():
    fn_start = "contrast/allchannelshistogramequalization"

    aug = iaa.AllChannelsHistogramEqualization()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)

    aug = iaa.Alpha((0.0, 1.0), iaa.AllChannelsHistogramEqualization())
    run_and_save_augseq(
        fn_start + "_alpha.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*4)], cols=4, rows=4)


def chapter_augmenters_histogramequalization():
    fn_start = "contrast/histogramequalization"

    aug = iaa.HistogramEqualization()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)

    aug = iaa.Alpha((0.0, 1.0), iaa.HistogramEqualization())
    run_and_save_augseq(
        fn_start + "_alpha.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*4)], cols=4, rows=4)

    aug = iaa.HistogramEqualization(
        from_colorspace=iaa.HistogramEqualization.BGR,
        to_colorspace=iaa.HistogramEqualization.HSV)
    quokka_bgr = cv2.cvtColor(ia.quokka(size=(128, 128)), cv2.COLOR_RGB2BGR)
    run_and_save_augseq(
        fn_start + "_bgr_to_hsv.jpg", aug,
        [quokka_bgr for _ in range(4*1)], cols=4, rows=1,
        image_colorspace="RGB")


if __name__ == "__main__":
    main()
