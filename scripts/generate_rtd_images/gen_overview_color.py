from __future__ import print_function, division

import os

import numpy as np
import cv2
import imageio

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import run_and_save_augseq


FILE_DIR = os.path.realpath(os.path.dirname(__file__))
INPUT_IMAGES_DIR = os.path.join(FILE_DIR, "..", "..", "images", "input_images")


def main():
    chapter_augmenters_withcolorspace()
    chapter_augmenters_withbrightnesschannels()
    chapter_augmenters_multiplyandaddtobrightness()
    chapter_augmenters_multiplybrightness()
    chapter_augmenters_addtobrightness()
    chapter_augmenters_withhueandsaturation()
    chapter_augmenters_multiplyhueandsaturation()
    chapter_augmenters_multiplyhue()
    chapter_augmenters_multiplysaturation()
    chapter_augmenters_removesaturation()
    chapter_augmenters_addtohueandsaturation()
    chapter_augmenters_addtohue()
    chapter_augmenters_addtosaturation()
    chapter_augmenters_kmeanscolorquantization()
    chapter_augmenters_uniformcolorquantization()
    chapter_augmenters_uniformcolorquantizationtonbits()
    chapter_augmenters_changecolorspace()
    chapter_augmenters_grayscale()
    chapter_augmenters_grayscalecolorwise()
    chapter_augmenters_removesaturationcolorwise()
    chapter_augmenters_changecolortemperature()


def chapter_augmenters_withcolorspace():
    fn_start = "color/withcolorspace"

    aug = iaa.WithColorspace(
        to_colorspace="HSV",
        from_colorspace="RGB",
        children=iaa.WithChannels(
            0,
            iaa.Add((0, 50))
        )
    )
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_withbrightnesschannels():
    fn_start = "color/withbrightnesschannels"

    aug = iaa.WithBrightnessChannels(iaa.Add((-50, 50)))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.WithBrightnessChannels(
        iaa.Add((-50, 50)), to_colorspace=[iaa.CSPACE_Lab, iaa.CSPACE_HSV])
    run_and_save_augseq(
        fn_start + "_to_colorspace.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_multiplyandaddtobrightness():
    fn_start = "color/multiplyandaddtobrightness"

    aug = iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5),
                                         add=(-30, 30))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_multiplybrightness():
    fn_start = "color/multiplybrightness"

    aug = iaa.MultiplyBrightness((0.5, 1.5))

    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_addtobrightness():
    fn_start = "color/addtobrightness"

    aug = iaa.AddToBrightness((-30, 30))

    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_withhueandsaturation():
    fn_start = "color/withhueandsaturation"

    aug = iaa.WithHueAndSaturation(
        iaa.WithChannels(0, iaa.Add((0, 50)))
    )
    run_and_save_augseq(
        fn_start + "_add_to_hue.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.WithHueAndSaturation([
        iaa.WithChannels(0, iaa.Add((-30, 10))),
        iaa.WithChannels(1, [
            iaa.Multiply((0.5, 1.5)),
            iaa.LinearContrast((0.75, 1.25))
        ])
    ])
    run_and_save_augseq(
        fn_start + "_modify_both.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_multiplyhueandsaturation():
    fn_start = "color/multiplyhueandsaturation"

    aug = iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5))
    run_and_save_augseq(
        fn_start + "_mul_hue.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.MultiplyHueAndSaturation(mul_saturation=(0.5, 1.5))
    run_and_save_augseq(
        fn_start + "_mul_saturation.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_multiplyhue():
    fn_start = "color/multiplyhue"

    aug = iaa.MultiplyHue((0.5, 1.5))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_multiplysaturation():
    fn_start = "color/multiplysaturation"

    aug = iaa.MultiplySaturation((0.5, 1.5))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_removesaturation():
    fn_start = "color/removesaturation"

    aug = iaa.RemoveSaturation()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.RemoveSaturation(1.0)
    run_and_save_augseq(
        fn_start + "_all.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_addtohueandsaturation():
    fn_start = "color/addtohueandsaturation"

    aug = iaa.AddToHueAndSaturation((-50, 50), per_channel=True)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_addtohue():
    fn_start = "color/addtohue"

    aug = iaa.AddToHue((-50, 50))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_addtosaturation():
    fn_start = "color/addtosaturation"

    aug = iaa.AddToSaturation((-50, 50))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_kmeanscolorquantization():
    fn_start = "color/kmeanscolorquantization"

    aug = iaa.KMeansColorQuantization()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.KMeansColorQuantization(n_colors=8)
    run_and_save_augseq(
        fn_start + "_with_8_colors.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.KMeansColorQuantization(n_colors=(4, 16))
    run_and_save_augseq(
        fn_start + "_with_random_n_colors.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*3)], cols=4, rows=3
    )

    aug = iaa.KMeansColorQuantization(
        from_colorspace=iaa.ChangeColorspace.BGR)
    quokka_bgr = cv2.cvtColor(ia.quokka(size=(128, 128)), cv2.COLOR_RGB2BGR)
    run_and_save_augseq(
        fn_start + "_from_bgr.jpg", aug,
        [quokka_bgr for _ in range(8)], cols=4, rows=2,
        image_colorspace="BGR"
    )

    aug = iaa.KMeansColorQuantization(
        to_colorspace=[iaa.ChangeColorspace.RGB, iaa.ChangeColorspace.HSV])
    run_and_save_augseq(
        fn_start + "_in_rgb_or_hsv.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_uniformcolorquantization():
    fn_start = "color/uniformcolorquantization"

    aug = iaa.UniformColorQuantization()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.UniformColorQuantization(n_colors=8)
    run_and_save_augseq(
        fn_start + "_with_8_colors.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.UniformColorQuantization(n_colors=(4, 16))
    run_and_save_augseq(
        fn_start + "_with_random_n_colors.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*3)], cols=4, rows=3,
        seed=2
    )

    aug = iaa.UniformColorQuantization(
        from_colorspace=iaa.ChangeColorspace.BGR,
        to_colorspace=[iaa.ChangeColorspace.RGB, iaa.ChangeColorspace.HSV])
    quokka_bgr = cv2.cvtColor(ia.quokka(size=(128, 128)), cv2.COLOR_RGB2BGR)
    run_and_save_augseq(
        fn_start + "_in_rgb_or_hsv.jpg", aug,
        [quokka_bgr for _ in range(8)], cols=4, rows=2,
        image_colorspace="BGR"
    )


def chapter_augmenters_uniformcolorquantizationtonbits():
    fn_start = "color/uniformcolorquantizationtonbits"

    aug = iaa.UniformColorQuantizationToNBits()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*3)], cols=4, rows=3
    )

    aug = iaa.UniformColorQuantizationToNBits(nb_bits=(2, 8))
    run_and_save_augseq(
        fn_start + "_2_to_8.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*3)], cols=4, rows=3
    )

    aug = iaa.UniformColorQuantizationToNBits(
        from_colorspace=iaa.CSPACE_BGR,
        to_colorspace=[iaa.CSPACE_RGB, iaa.CSPACE_HSV])
    run_and_save_augseq(
        fn_start + "_with_random_n_colors.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*3)], cols=4, rows=3,
        seed=2
    )


def chapter_augmenters_changecolorspace():
    aug = iaa.Sequential([
        iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
        iaa.WithChannels(0, iaa.Add((50, 100))),
        iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
    ])
    run_and_save_augseq(
        "color/changecolorspace.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_grayscale():
    aug = iaa.Grayscale(alpha=(0.0, 1.0))
    run_and_save_augseq(
        "color/grayscale.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    #alphas = [1/8*i for i in range(8)]
    alphas = np.linspace(0, 1.0, num=8)
    run_and_save_augseq(
        "color/grayscale_vary_alpha.jpg",
        [iaa.Grayscale(alpha=alpha) for alpha in alphas],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1
    )


def chapter_augmenters_grayscalecolorwise():
    fn_start = "color/grayscalecolorwise"

    image = imageio.imread(
        os.path.join(INPUT_IMAGES_DIR,
                     "1280px-Vincent_Van_Gogh_-_Wheatfield_with_Crows.jpg"))
    image = ia.imresize_single_image(image, 0.125)

    aug = iaa.GrayscaleColorwise()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [image] * (3*5), cols=3, rows=5
    )

    aug = iaa.GrayscaleColorwise(nb_bins=200, smoothness=0.5)
    run_and_save_augseq(
        fn_start + "_many_bins.jpg", aug,
        [image] * (3*5), cols=3, rows=5
    )

    aug = iaa.GrayscaleColorwise(nb_bins=2, smoothness=0.0)
    run_and_save_augseq(
        fn_start + "_two_bins.jpg", aug,
        [image] * (3*5), cols=3, rows=5,
    )

    import imgaug.parameters as iap
    aug = iaa.GrayscaleColorwise(nb_bins=2, smoothness=0.0,
                                 alpha=iap.Beta(0.5, 0.5))
    run_and_save_augseq(
        fn_start + "_beta.jpg", aug,
        [image] * (3*5), cols=3, rows=5,
    )


def chapter_augmenters_removesaturationcolorwise():
    fn_start = "color/removesaturationcolorwise"

    image = imageio.imread(
        os.path.join(INPUT_IMAGES_DIR,
                     "1280px-Vincent_Van_Gogh_-_Wheatfield_with_Crows.jpg"))
    image = ia.imresize_single_image(image, 0.125)

    aug = iaa.RemoveSaturationColorwise()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [image] * (3*5), cols=3, rows=5
    )


def chapter_augmenters_changecolortemperature():
    fn_start = "color/changecolortemperature"

    image = imageio.imread(
        os.path.join(INPUT_IMAGES_DIR,
                     "Pahalgam_Valley.jpg"))
    image = ia.imresize_single_image(image, 0.2)

    aug = iaa.ChangeColorTemperature((1100, 10000))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [image] * 8, cols=3, rows=2
    )


if __name__ == "__main__":
    main()
