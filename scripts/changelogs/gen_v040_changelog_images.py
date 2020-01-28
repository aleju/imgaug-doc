import os
import imageio
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, "..", ".."))
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
OUT_DIR = os.path.join(IMAGES_DIR, "changelogs", "0.4.0")


def main():
    image, segmap = load_data()

    # color car lights
    ia.seed(10)  # 2 = blue lights, 8 = pink, 10 = green lights
    image_aug, _segmap_aug = iaa.BlendAlphaSegMapClassIds(
        1,
        foreground=iaa.BlendAlphaSomeColors(
            iaa.AddToHueAndSaturation(
                value_hue=(-200, 200), value_saturation=(-100, 100)
            )
        )
    )(image=image, segmentation_maps=segmap)

    imageio.imwrite(
        os.path.join(OUT_DIR, "cityscapes5-car-lights-changed.jpg"),
        ia.imresize_single_image(image_aug, 0.3)
    )

    # color train
    ia.seed(37)
    image_aug, _segmap_aug = iaa.BlendAlphaSegMapClassIds(
        2,
        foreground=iaa.AddToHueAndSaturation(
            value_hue=(-200, 200), value_saturation=(-100, 100)
        )
    )(image=image, segmentation_maps=segmap)

    imageio.imwrite(
        os.path.join(OUT_DIR, "cityscapes5-train-color.jpg"),
        ia.imresize_single_image(image_aug, 0.3)
    )

    # emboss street
    image_aug, _segmap_aug = iaa.BlendAlphaSegMapClassIds(
        4,
        foreground=iaa.Emboss(1.0, strength=1.0)
    )(image=image, segmentation_maps=segmap)

    imageio.imwrite(
        os.path.join(OUT_DIR, "cityscapes5-street-embossed.jpg"),
        ia.imresize_single_image(image_aug, 0.3)
    )

    # replace street with gaussian noise
    ia.seed(3)
    image_aug, _segmap_aug = iaa.BlendAlphaSegMapClassIds(
        4,
        foreground=iaa.Sequential([
            iaa.Multiply(0.0),
            iaa.AdditiveGaussianNoise(loc=128, scale=40, per_channel=True)
        ]),
    )(image=image, segmentation_maps=segmap)

    imageio.imwrite(
        os.path.join(OUT_DIR, "cityscapes5-street-gaussian-noise.jpg"),
        ia.imresize_single_image(image_aug, 0.3)
    )

    # regular grid dropout
    ia.seed(1)
    image_aug = iaa.BlendAlphaRegularGrid(
        nb_rows=(8, 12),
        nb_cols=(8, 12),
        foreground=iaa.Multiply(0.0)
    )(image=image)

    imageio.imwrite(
        os.path.join(OUT_DIR, "cityscapes5-regular-grid-dropout.jpg"),
        ia.imresize_single_image(image_aug, 0.3)
    )

    # checkerboard dropout
    ia.seed(1)
    image_aug = iaa.BlendAlphaCheckerboard(
        nb_rows=(8, 12),
        nb_cols=(8, 12),
        foreground=iaa.Multiply(0.0)
    )(image=image)

    imageio.imwrite(
        os.path.join(OUT_DIR, "cityscapes5-checkerboard-dropout.jpg"),
        ia.imresize_single_image(image_aug, 0.3)
    )


def load_data():
    image_fp = os.path.join(IMAGES_DIR, "annotated", "cityscapes5.png")
    image_fp_anno = os.path.join(IMAGES_DIR, "annotated",
                                 "cityscapes5-annotation.png")
    image = imageio.imread(image_fp)[:, :, 0:3]
    image_anno = imageio.imread(image_fp_anno)[:, :, 0:3]
    print(image.shape, image_anno.shape)
    color_cars = [0, 0, 142]
    color_train = [0, 80, 100]
    color_signs = [220, 220, 0]
    color_street = [128, 64, 128]
    colors = [color_cars, color_train, color_signs, color_street]
    segmap_arr = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.int32)
    for i, col in enumerate(colors):
        col_arr = np.int32(col).reshape((1, 1, -1))
        diff = np.sum(np.abs(image_anno.astype(np.int32) - col_arr), axis=2)
        mask = diff < 20
        print(i, np.sum(mask))
        segmap_arr[mask] = i+1
    segmap = ia.SegmentationMapsOnImage(segmap_arr, shape=image.shape)

    return image, segmap


if __name__ == "__main__":
    main()
