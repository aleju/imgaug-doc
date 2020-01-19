from __future__ import print_function, division

import os

import imageio

import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap

from . import utils
from .utils import run_and_save_augseq, DOCS_IMAGES_BASE_PATH


def main():
    chapter_augmenters_blendalpha()
    chapter_augmenters_blendalphamask()
    chapter_augmenters_blendalphaelementwise()
    chapter_augmenters_blendalphasimplexnoise()
    chapter_augmenters_blendalphafrequencynoise()
    chapter_augmenters_blendalphasomecolors()
    chapter_augmenters_blendalphahorizontallineargradient()
    chapter_augmenters_blendalphaverticallineargradient()
    chapter_augmenters_blendalpharegulargrid()
    chapter_augmenters_blendalphacheckerboard()
    chapter_augmenters_blendalphasegmapclassids()
    chapter_augmenters_blendalphaboundingboxes()


def chapter_augmenters_blendalpha():
    fn_start = "blend/blendalpha"
    aug_cls = iaa.Alpha

    aug = aug_cls(0.5, iaa.Grayscale(1.0))
    run_and_save_augseq(
        fn_start + "_050_grayscale.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4)], cols=4, rows=1)

    aug = aug_cls((0.0, 1.0), iaa.Grayscale(1.0))
    run_and_save_augseq(
        fn_start + "_uniform_factor.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls(
        (0.0, 1.0),
        iaa.Affine(rotate=(-20, 20)),
        per_channel=0.5)
    run_and_save_augseq(
        fn_start + "_affine_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls(
        (0.0, 1.0),
        first=iaa.Add(100),
        second=iaa.Multiply(0.2))
    run_and_save_augseq(
        fn_start + "_two_branches.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls([0.25, 0.75], iaa.MedianBlur(13))
    run_and_save_augseq(
        fn_start + "_with_choice.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_blendalphamask():
    fn_start = "blend/blendalphamask"

    aug = iaa.BlendAlphaMask(
        iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()),
        iaa.Sequential([
            iaa.Clouds(),
            iaa.WithChannels([1, 2], iaa.Multiply(0.5))
        ])
    )
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_blendalphaelementwise():
    fn_start = "blend/blendalphaelementwise"
    aug_cls = iaa.AlphaElementwise

    aug = aug_cls(0.5, iaa.Grayscale(1.0))
    run_and_save_augseq(
        fn_start + "_050_grayscale.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4)], cols=4, rows=1)

    aug = aug_cls((0.0, 1.0), iaa.AddToHue(100))
    run_and_save_augseq(
        fn_start + "_uniform_factor.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1)

    aug = aug_cls(
        (0.0, 1.0),
        iaa.Affine(rotate=(-20, 20)),
        per_channel=0.5)
    run_and_save_augseq(
        fn_start + "_affine_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls(
        (0.0, 1.0),
        first=iaa.Add(100),
        second=iaa.Multiply(0.2))
    run_and_save_augseq(
        fn_start + "_two_branches.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls([0.25, 0.75], iaa.MedianBlur(13))
    run_and_save_augseq(
        fn_start + "_with_choice.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_blendalphasimplexnoise():
    fn_start = "blend/blendalphasimplexnoise"

    aug = iaa.SimplexNoiseAlpha(iaa.EdgeDetect(1.0))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.SimplexNoiseAlpha(
        iaa.EdgeDetect(1.0),
        upscale_method="nearest")
    run_and_save_augseq(
        fn_start + "_nearest.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.SimplexNoiseAlpha(
        iaa.EdgeDetect(1.0),
        upscale_method="linear")
    run_and_save_augseq(
        fn_start + "_linear.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.SimplexNoiseAlpha(
        iaa.EdgeDetect(1.0),
        sigmoid_thresh=iap.Normal(10.0, 5.0))
    run_and_save_augseq(
        fn_start + "_sigmoid_thresh_normal.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_blendalphafrequencynoise():
    fn_start = "blend/blendalphafrequencynoise"

    aug = iaa.FrequencyNoiseAlpha(first=iaa.EdgeDetect(1.0))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.FrequencyNoiseAlpha(
        first=iaa.EdgeDetect(1.0),
        upscale_method="nearest")
    run_and_save_augseq(
        fn_start + "_nearest.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.FrequencyNoiseAlpha(
        first=iaa.EdgeDetect(1.0),
        upscale_method="linear")
    run_and_save_augseq(
        fn_start + "_linear.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.FrequencyNoiseAlpha(
        first=iaa.EdgeDetect(1.0),
        upscale_method="linear",
        exponent=-2,
        sigmoid=False)
    run_and_save_augseq(
        fn_start + "_clouds.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.FrequencyNoiseAlpha(
        first=iaa.EdgeDetect(1.0),
        sigmoid_thresh=iap.Normal(10.0, 5.0))
    run_and_save_augseq(
        fn_start + "_sigmoid_thresh_normal.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_blendalphasomecolors():
    fn_start = "blend/blendalphasomecolors"
    image = imageio.imread(
        os.path.join(DOCS_IMAGES_BASE_PATH,
                     "input_images",
                     "1280px-Vincent_Van_Gogh_-_Wheatfield_with_Crows.jpg"))

    # ca. 15% of original size
    image = ia.imresize_single_image(image, (92, 192))

    aug = iaa.BlendAlphaSomeColors(iaa.Grayscale(1.0))
    run_and_save_augseq(
        fn_start + "_grayscale.jpg", aug,
        [image for _ in range(4*3)], cols=4, rows=3)

    aug = iaa.BlendAlphaSomeColors(iaa.TotalDropout(1.0))
    run_and_save_augseq(
        fn_start + "_total_dropout.jpg", aug,
        [image for _ in range(4*3)], cols=4, rows=3)

    aug = iaa.BlendAlphaSomeColors(
        iaa.MultiplySaturation(0.5), iaa.MultiplySaturation(1.5))
    run_and_save_augseq(
        fn_start + "_saturation.jpg", aug,
        [image for _ in range(4*3)], cols=4, rows=3)

    aug = iaa.BlendAlphaSomeColors(
        iaa.AveragePooling(7), alpha=[0.0, 1.0], smoothness=0.0)
    run_and_save_augseq(
        fn_start + "_pooling.jpg", aug,
        [image for _ in range(4*3)], cols=4, rows=3)

    aug = iaa.BlendAlphaSomeColors(
        iaa.AveragePooling(7), nb_bins=2, smoothness=0.0)
    run_and_save_augseq(
        fn_start + "_pooling_2_bins.jpg", aug,
        [image for _ in range(4*4)], cols=4, rows=4)

    aug = iaa.BlendAlphaSomeColors(
        iaa.AveragePooling(7), from_colorspace="BGR")
    run_and_save_augseq(
        fn_start + "_pooling_bgr.jpg", aug,
        [image[:, :, ::-1] for _ in range(4*2)], cols=4, rows=2,
        image_colorspace=iaa.CSPACE_BGR)


def chapter_augmenters_blendalphahorizontallineargradient():
    fn_start = "blend/blendalphahorizontallineargradient"

    aug = iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToHue((-100, 100)))
    run_and_save_augseq(
        fn_start + "_hue.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.BlendAlphaHorizontalLinearGradient(
        iaa.TotalDropout(1.0),
        min_value=0.2, max_value=0.8)
    run_and_save_augseq(
        fn_start + "_total_dropout.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.BlendAlphaHorizontalLinearGradient(
        iaa.AveragePooling(11),
        start_at=(0.0, 1.0), end_at=(0.0, 1.0))
    run_and_save_augseq(
        fn_start + "_pooling.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_blendalphaverticallineargradient():
    fn_start = "blend/blendalphaverticallineargradient"

    aug = iaa.BlendAlphaVerticalLinearGradient(iaa.AddToHue((-100, 100)))
    run_and_save_augseq(
        fn_start + "_hue.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.BlendAlphaVerticalLinearGradient(
        iaa.TotalDropout(1.0),
        min_value=0.2, max_value=0.8)
    run_and_save_augseq(
        fn_start + "_total_dropout.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.BlendAlphaVerticalLinearGradient(
        iaa.AveragePooling(11),
        start_at=(0.0, 1.0), end_at=(0.0, 1.0))
    run_and_save_augseq(
        fn_start + "_pooling.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.BlendAlphaVerticalLinearGradient(
        iaa.Clouds(),
        start_at=(0.15, 0.35), end_at=0.0)
    run_and_save_augseq(
        fn_start + "_clouds.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_blendalpharegulargrid():
    fn_start = "blend/blendalpharegulargrid"

    aug = iaa.BlendAlphaRegularGrid(nb_rows=(4, 6), nb_cols=(1, 4),
                                    foreground=iaa.Multiply(0.0))
    run_and_save_augseq(
        fn_start + "_multiply.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.BlendAlphaRegularGrid(nb_rows=2, nb_cols=2,
                                    foreground=iaa.Multiply(0.0),
                                    background=iaa.AveragePooling(8),
                                    alpha=[0.0, 0.0, 1.0])
    run_and_save_augseq(
        fn_start + "_two_branches.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*3)], cols=4, rows=3)


def chapter_augmenters_blendalphacheckerboard():
    fn_start = "blend/blendalphacheckerboard"

    aug = iaa.BlendAlphaCheckerboard(nb_rows=2, nb_cols=(1, 4),
                                     foreground=iaa.AddToHue((-100, 100)))
    run_and_save_augseq(
        fn_start + "_hue.jpg", aug,
        [ia.quokka(size=(200, 200)) for _ in range(3*4)], cols=3, rows=4)


def chapter_augmenters_blendalphasegmapclassids():
    fn_start = "blend/blendalphasegmapclassids"

    aug = iaa.BlendAlphaSegMapClassIds(
        [1, 3], foreground=iaa.AddToHue((-100, 100)))
    batch = ia.Batch(
        images=[ia.quokka(size=(128, 128))] * (4*2),
        segmentation_maps=[ia.quokka_segmentation_map(size=(128, 128))] * (4*2)
    )
    run_and_save_augseq_batch(fn_start + "_hue.jpg", aug, batch,
                              cols=4, rows=2)


def chapter_augmenters_blendalphaboundingboxes():
    fn_start = "blend/blendalphaboundingboxes"

    aug = iaa.BlendAlphaBoundingBoxes(None,
                                      background=iaa.Multiply(0.0))
    batch = ia.Batch(
        images=[ia.quokka(size=(128, 128))] * (4*1),
        bounding_boxes=[ia.quokka_bounding_boxes(size=(128, 128))] * (4*1)
    )
    run_and_save_augseq_batch(fn_start + "_multiply_background.jpg", aug, batch,
                              cols=4, rows=1)


def run_and_save_augseq_batch(filename, augseq, batch, cols, rows, quality=95,
                              seed=1):
    ia.seed(seed)

    bia = batch.to_batch_in_augmentation()

    # calling N times augment_image() is here critical for random order in
    # Sequential
    images_aug = []
    nb_rows = bia.nb_rows
    for i in range(nb_rows):
        bia_i = bia.subselect_rows_by_indices([i])
        bia_i_aug = augseq.augment_batch(bia_i)
        images_aug.append(bia_i_aug.images[0])

    utils.save(
        "overview_of_augmenters",
        filename,
        utils.grid(images_aug, cols=cols, rows=rows),
        quality=quality
    )


if __name__ == "__main__":
    main()
