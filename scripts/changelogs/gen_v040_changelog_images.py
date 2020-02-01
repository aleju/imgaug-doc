import os
import tempfile

import imageio
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, "..", ".."))
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
OUT_DIR = os.path.join(IMAGES_DIR, "changelogs", "0.4.0")

INPUT_IMAGES_DIR = os.path.join(FILE_DIR, "..", "..", "images",
                                "input_images")


def main():
    # arithmetic
    generate_cutout()
    generate_dropout2d()
    generate_solarize()

    # cartoon
    generate_cartoon()

    # blend
    generate_blending()

    # mean shift blur
    generate_mean_shift_blur()

    # color
    generate_add_to_brightness()
    generate_change_color_temperature()
    generate_posterize()

    # randaugment
    generate_randaugment()

    # debug
    generate_debug()

    # geometric
    generate_with_polar_warping()
    generate_jigsaw()

    # imgcorruptlike
    generate_imgcorruptlike()

    # pillike
    generate_pillike()

    # rain
    generate_rain()


def _save(fname, image, size=None):
    if size is not None:
        image = ia.imresize_single_image(image, size)
    imageio.imwrite(
        os.path.join(OUT_DIR, fname),
        image
    )


def generate_cutout():
    ia.seed(1)

    image = ia.quokka((128, 128))
    images_aug = []

    images_aug.append(image)
    images_aug.extend(
        iaa.Cutout()(images=[image] * 7)
    )

    images_aug.append(image)
    images_aug.extend(
        iaa.Cutout(
            nb_iterations=2,
            cval=(0, 255),
            fill_mode=["constant", "gaussian"],
            fill_per_channel=0.8
        )(images=[image] * 7)
    )

    _save("cutout.jpg", ia.draw_grid(images_aug, cols=7, rows=2))


def generate_dropout2d():
    ia.seed(1)

    image = ia.quokka((128, 128))
    images_aug = [image]
    images_aug.extend(iaa.Dropout2d(p=0.5)(images=[image] * (2 * 8 - 1)))
    _save("dropout2d.jpg", ia.draw_grid(images_aug, cols=8, rows=2))


def generate_solarize():
    ia.seed(1)

    image = ia.quokka((128, 128))
    images_aug = [image]
    images_aug.extend(iaa.Solarize(p=1.0)(images=[image] * (2 * 8 - 1)))
    _save("solarize.jpg", ia.draw_grid(images_aug, cols=8, rows=2))


def generate_cartoon():
    ia.seed(1)

    image1 = imageio.imread(os.path.join(
        INPUT_IMAGES_DIR, "Pahalgam_Valley.jpg"))
    image2 = imageio.imread(os.path.join(
        INPUT_IMAGES_DIR, "1024px-Salad_platter.jpg"))
    image1 = iaa.Resize(
        {"width": 256, "height": "keep-aspect-ratio"}
    )(image=image1)
    image2 = iaa.Resize(
        {"width": 256, "height": "keep-aspect-ratio"}
    )(image=image2)

    images_aug = [image1]
    images_aug.extend(iaa.Cartoon()(images=[image1] * 3))
    images_aug.append(image2)
    images_aug.extend(iaa.Cartoon()(images=[image2] * 3))

    _save("cartoon.jpg", ia.draw_grid(images_aug, cols=4, rows=2))


def generate_mean_shift_blur():
    ia.seed(1)

    image = ia.quokka((128, 128))
    images_aug = [image]
    for radius in np.linspace(5.0, 40.0, 7):
        images_aug.append(
            iaa.MeanShiftBlur(
                spatial_radius=radius,
                color_radius=radius
            )(image=image)
        )
    _save("meanshiftblur.jpg", ia.draw_grid(images_aug, cols=8, rows=1))


def generate_add_to_brightness():
    ia.seed(1)

    image = ia.quokka((128, 128))
    images_aug = [image]
    for value in np.linspace(-100, 100, 7):
        images_aug.append(
            iaa.AddToBrightness(value)(image=image)
        )
    _save("addtobrightness.jpg", ia.draw_grid(images_aug, cols=8, rows=1))


def generate_change_color_temperature():
    ia.seed(1)

    image = imageio.imread(os.path.join(
        INPUT_IMAGES_DIR, "Pahalgam_Valley.jpg"))
    image = ia.imresize_single_image(image, 0.12)
    images_aug = [image]
    for kelvin in np.linspace(1000, 5000, 7):
        images_aug.append(
            iaa.ChangeColorTemperature(kelvin)(image=image)
        )
    _save(
        "changecolortemperature.jpg",
        ia.draw_grid(images_aug, cols=8, rows=1)
    )


def generate_posterize():
    ia.seed(1)

    image = ia.quokka((128, 128))
    images_aug = [image]
    for nbits in np.arange(7)[::-1]:
        images_aug.append(
            iaa.Posterize(1 + nbits)(image=image)
        )
    _save("posterize.jpg", ia.draw_grid(images_aug, cols=8, rows=1))


def generate_randaugment():
    ia.seed(1)

    image = ia.quokka((128, 128))
    images_aug = [image]
    images_aug.extend(iaa.RandAugment(m=20)(images=[image] * (2 * 8 - 1)))
    _save("randaugment.jpg", ia.draw_grid(images_aug, cols=8, rows=2))


def generate_debug():
    ia.seed(1)

    image = ia.quokka_square((128, 128))
    bbsoi = ia.quokka_bounding_boxes((128, 128), extract="square")
    segmaps = ia.quokka_segmentation_map((128, 128), extract="square")

    with tempfile.TemporaryDirectory() as folder_path:
        seq = iaa.Sequential([
            iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Crop(px=(0, 16))
            ], random_order=True),
            iaa.SaveDebugImageEveryNBatches(folder_path, 100)
        ])

        _ = seq(images=[image] * 4,
                segmentation_maps=[segmaps] * 4,
                bounding_boxes=[bbsoi] * 4)

        image_debug_path = os.path.join(folder_path, "batch_latest.png")
        image_debug = imageio.imread(image_debug_path)

        _save("savedebugimageeverynbatches.jpg", image_debug)


def generate_with_polar_warping():
    ia.seed(1)

    image = ia.quokka((128, 128))

    images_aug = [image]
    aug = iaa.WithPolarWarping(iaa.CropAndPad(percent=(-0.1, 0.1)))
    images_aug.extend(aug(images=[image] * 7))

    images_aug.append(image)
    aug = iaa.WithPolarWarping(
        iaa.Affine(
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-35, 35),
            scale=(0.8, 1.2),
            shear={"x": (-15, 15), "y": (-15, 15)}
        )
    )
    images_aug.extend(aug(images=[image] * 7))

    images_aug.append(image)
    aug = iaa.WithPolarWarping(iaa.AveragePooling((2, 8)))
    images_aug.extend(aug(images=[image] * 7))

    _save("withpolarwarping.jpg", ia.draw_grid(images_aug, cols=8, rows=3))


def generate_jigsaw():
    ia.seed(1)

    image = ia.quokka((128, 128))

    images_aug = [image]
    images_aug.extend(
        iaa.Jigsaw(nb_rows=5, nb_cols=5)(images=[image] * 7)
    )

    images_aug.append(image)
    images_aug.extend(
        iaa.Jigsaw(nb_rows=10, nb_cols=10)(images=[image] * 7)
    )

    _save("jigsaw.jpg", ia.draw_grid(images_aug, cols=8, rows=2))


def generate_imgcorruptlike():
    ia.seed(1)

    image = ia.quokka((128, 128))
    images_aug = []
    augnames = [
        "Original",
        "GaussianNoise",
        "ShotNoise",
        "ImpulseNoise",
        "SpeckleNoise",
        "GaussianBlur",
        "GlassBlur",
        "DefocusBlur",
        "MotionBlur",
        "ZoomBlur",
        "Fog",
        "Frost",
        "Snow",
        "Spatter",
        "Contrast",
        "Brightness",
        "Saturate",
        "JpegCompression",
        "Pixelate",
        "ElasticTransform"
    ]

    for augname in augnames:
        if augname == "Original":
            image_aug = np.copy(image)
        else:
            aug = getattr(iaa.imgcorruptlike, augname)
            image_aug = aug(severity=3)(image=image)
        image_aug = iaa.pad(image_aug, top=30, cval=255)
        image_aug = ia.draw_text(
            image_aug, y=6, x=2, text=augname, color=(0, 0, 0), size=15
        )
        images_aug.append(image_aug)

    _save("imgcorruptlike.jpg", ia.draw_grid(images_aug, cols=5, rows=4))


def generate_pillike():
    ia.seed(1)

    image = ia.quokka((128, 128))

    images_aug = [image]
    images_aug.extend(iaa.pillike.Autocontrast()(images=[image] * 7))

    images_aug.append(image)
    for factor in np.linspace(0.1, 1.9, 7):
        images_aug.append(iaa.pillike.EnhanceColor(factor)(image=image))

    images_aug.append(image)
    for factor in np.linspace(0.1, 1.9, 7):
        images_aug.append(iaa.pillike.EnhanceSharpness(factor)(image=image))

    images_aug.append(image)
    images_aug.extend([
        iaa.pillike.FilterBlur()(image=image),
        iaa.pillike.FilterSmooth()(image=image),
        iaa.pillike.FilterEdgeEnhance()(image=image),
        iaa.pillike.FilterFindEdges()(image=image),
        iaa.pillike.FilterContour()(image=image),
        iaa.pillike.FilterSharpen()(image=image),
        iaa.pillike.FilterDetail()(image=image)
    ])

    images_aug.append(image)
    images_aug.extend(
        iaa.pillike.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-10, 10),
            shear={"x": (-10, 10), "y": (-10, 10)},
            center=(0.0, 0.0)
        )(images=[image] * 7)
    )

    _save("pillike.jpg", ia.draw_grid(images_aug, cols=8, rows=5))


def generate_rain():
    ia.seed(2)

    image = imageio.imread(os.path.join(
        INPUT_IMAGES_DIR, "Pahalgam_Valley.jpg"))
    image = iaa.Resize(
        {"width": 256, "height": "keep-aspect-ratio"}
    )(image=image)

    images_aug = [image]
    images_aug.extend(iaa.Rain()(images=[image] * (2 * 8 - 1)))

    _save("rain.jpg", ia.draw_grid(images_aug, cols=4, rows=4))


def generate_blending():
    image, segmap = load_cityscapes_data()

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

    _save("cityscapes5-car-lights-changed.jpg", image_aug, size=0.3)

    # color train
    ia.seed(37)
    image_aug, _segmap_aug = iaa.BlendAlphaSegMapClassIds(
        2,
        foreground=iaa.AddToHueAndSaturation(
            value_hue=(-200, 200), value_saturation=(-100, 100)
        )
    )(image=image, segmentation_maps=segmap)

    _save("cityscapes5-train-color.jpg", image_aug, size=0.3)

    # emboss street
    image_aug, _segmap_aug = iaa.BlendAlphaSegMapClassIds(
        4,
        foreground=iaa.Emboss(1.0, strength=1.0)
    )(image=image, segmentation_maps=segmap)

    _save("cityscapes5-street-embossed.jpg", image_aug, size=0.3)

    # replace street with gaussian noise
    ia.seed(3)
    image_aug, _segmap_aug = iaa.BlendAlphaSegMapClassIds(
        4,
        foreground=iaa.Sequential([
            iaa.Multiply(0.0),
            iaa.AdditiveGaussianNoise(loc=128, scale=40, per_channel=True)
        ]),
    )(image=image, segmentation_maps=segmap)

    _save("cityscapes5-street-gaussian-noise.jpg", image_aug, size=0.3)

    # regular grid dropout
    ia.seed(1)
    image_aug = iaa.BlendAlphaRegularGrid(
        nb_rows=(8, 12),
        nb_cols=(8, 12),
        foreground=iaa.Multiply(0.0)
    )(image=image)

    _save("cityscapes5-regular-grid-dropout.jpg", image_aug, size=0.3)

    # checkerboard dropout
    ia.seed(1)
    image_aug = iaa.BlendAlphaCheckerboard(
        nb_rows=(8, 12),
        nb_cols=(8, 12),
        foreground=iaa.Multiply(0.0)
    )(image=image)

    _save("cityscapes5-checkerboard-dropout.jpg", image_aug, size=0.3)

    # somecolors + removesaturation
    ia.seed(1)
    image_gogh = imageio.imread(
        os.path.join(
            INPUT_IMAGES_DIR,
            "1280px-Vincent_Van_Gogh_-_Wheatfield_with_Crows.jpg"
        )
    )
    image_gogh = iaa.Resize(
        {"width": 256, "height": "keep-aspect-ratio"}
    )(image=image_gogh)
    images_aug = (
        [image_gogh]
        + iaa.BlendAlphaSomeColors(
            iaa.RemoveSaturation(1.0)
        )(images=[image_gogh] * (2*4-1))
    )
    _save(
        "blendalphasomecolors_removesaturation.jpg",
        ia.draw_grid(images_aug, cols=4, rows=2)
    )


def load_cityscapes_data():
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
