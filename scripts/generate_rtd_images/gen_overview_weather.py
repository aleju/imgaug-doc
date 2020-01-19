from __future__ import print_function, division

import imageio

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import run_and_save_augseq, save

LANDSCAPE_IMAGE = imageio.imread(
    "https://upload.wikimedia.org/wikipedia/commons/8/89/"
    "Kukle%2CCzech_Republic..jpg",
    format="jpg")
# 960x1280 -> 96x128
LANDSCAPE_IMAGE = ia.imresize_single_image(LANDSCAPE_IMAGE, 0.1)


def main():
    save("overview_of_augmenters", "weather/input_image.jpg", LANDSCAPE_IMAGE)
    chapter_augmenters_fastsnowylandscape()
    chapter_augmenters_clouds()
    chapter_augmenters_fog()
    chapter_augmenters_cloudlayer()
    chapter_augmenters_snowflakes()
    chapter_augmenters_snowflakeslayer()
    chapter_augmenters_rain()
    chapter_augmenters_rainlayer()


def chapter_augmenters_fastsnowylandscape():
    fn_start = "weather/fastsnowylandscape"
    image = LANDSCAPE_IMAGE

    aug = iaa.FastSnowyLandscape(
        lightness_threshold=140,
        lightness_multiplier=2.5
    )
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [image for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.FastSnowyLandscape(
        lightness_threshold=[128, 200],
        lightness_multiplier=(1.5, 3.5)
    )
    run_and_save_augseq(
        fn_start + "_random_choice.jpg", aug,
        [image for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.FastSnowyLandscape(
        lightness_threshold=(100, 255),
        lightness_multiplier=(1.0, 4.0)
    )
    run_and_save_augseq(
        fn_start + "_random_uniform.jpg", aug,
        [image for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_clouds():
    fn_start = "weather/clouds"
    image = LANDSCAPE_IMAGE

    aug = iaa.Clouds()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [image for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_cloudlayer():
    pass


def chapter_augmenters_fog():
    fn_start = "weather/fog"
    image = LANDSCAPE_IMAGE

    aug = iaa.Fog()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [image for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_snowflakes():
    fn_start = "weather/snowflakes"
    image = LANDSCAPE_IMAGE

    aug = iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [image for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_snowflakeslayer():
    pass


def chapter_augmenters_rain():
    fn_start = "weather/rain"
    image = LANDSCAPE_IMAGE

    aug = iaa.Rain(speed=(0.1, 0.3))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [image for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_rainlayer():
    pass


if __name__ == "__main__":
    main()
