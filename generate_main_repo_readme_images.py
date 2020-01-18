from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import imageio
import tempfile
import six.moves as sm
import re
import os
from collections import defaultdict
import PIL.Image
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

np.random.seed(44)
ia.seed(44)

IMAGES_DIR = "readme_images"
INPUT_IMAGES_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                "images", "input_images")


def main():
    #draw_small_overview()
    #draw_single_sequential_images()
    draw_per_augmenter_videos()


def draw_small_overview():
    ia.seed(44)
    image = ia.quokka(size=0.2)
    heatmap = ia.quokka_heatmap(size=0.2)
    segmap = ia.quokka_segmentation_map(size=0.2)
    kps = ia.quokka_keypoints(size=0.2)
    bbs = ia.quokka_bounding_boxes(size=0.2)
    polys = ia.quokka_polygons(size=0.2)
    batch = ia.Batch(
        images=[image],
        heatmaps=[heatmap.invert()],
        segmentation_maps=[segmap],
        keypoints=[kps],
        bounding_boxes=[bbs],
        polygons=[polys]
    )

    augs = []
    augs.append(("noop", iaa.Noop()))
    augs.append(("non_geometric", iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=(0, 20)),
        iaa.ContrastNormalization(1.2),
        iaa.Sharpen(alpha=1.0, lightness=1.5)
    ])))
    augs.append(("affine", iaa.Affine(rotate=0, translate_percent={"x": 0.1}, scale=1.3, mode="constant", cval=25)))
    augs.append(("cropandpad", iaa.CropAndPad(percent=(-0.05, 0.2, -0.05, -0.2), pad_mode="maximum")))
    augs.append(("fliplr_perspective", iaa.Sequential([
        iaa.Fliplr(1.0),
        iaa.PerspectiveTransform(scale=0.15)
    ])))

    for name, aug in augs:
        result = list(aug.augment_batches([batch]))[0]
        image_aug = result.images_aug[0]
        image_aug_heatmap = result.heatmaps_aug[0].draw(cmap=None)[0]
        image_aug_segmap = result.segmentation_maps_aug[0].draw_on_image(image_aug, alpha=0.8)[0]
        image_aug_kps = result.keypoints_aug[0].draw_on_image(image_aug, color=[0, 255, 0], size=7)
        image_aug_bbs = result.bounding_boxes_aug[0].clip_out_of_image().draw_on_image(image_aug, size=3)
        # add polys for now to BBs image to save (screen) space
        image_aug_bbs = result.polygons_aug[0].clip_out_of_image().draw_on_image(
            image_aug_bbs, color=[0, 128, 0], color_points=[0, 128, 0], alpha=0.0,
            alpha_points=1.0, alpha_lines=0.5)
        imageio.imwrite(os.path.join(IMAGES_DIR, "small_overview", "%s_image.jpg" % (name,)), image_aug, quality=90)
        imageio.imwrite(os.path.join(IMAGES_DIR, "small_overview", "%s_heatmap.jpg" % (name,)), image_aug_heatmap, quality=90)
        imageio.imwrite(os.path.join(IMAGES_DIR, "small_overview", "%s_segmap.jpg" % (name,)), image_aug_segmap, quality=90)
        imageio.imwrite(os.path.join(IMAGES_DIR, "small_overview", "%s_kps.jpg" % (name,)), image_aug_kps, quality=90)
        imageio.imwrite(os.path.join(IMAGES_DIR, "small_overview", "%s_bbs.jpg" % (name,)), image_aug_bbs, quality=90)


def draw_single_sequential_images():
    ia.seed(44)

    #image = ia.imresize_single_image(imageio.imread("quokka.jpg", pilmode="RGB")[0:643, 0:643], (128, 128))
    image = ia.quokka_square(size=(128, 128))

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),  # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)),  # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)),  # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),  # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True),  # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.ContrastNormalization((0.5, 2.0))
                        )
                    ]),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),  # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),  # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )

    grid = seq.draw_grid(image, cols=8, rows=8)
    imageio.imwrite(os.path.join(IMAGES_DIR, "examples_grid.jpg"), grid)


def draw_per_augmenter_images():
    print("[draw_per_augmenter_images] Loading image...")
    #image = ia.imresize_single_image(imageio.imread("quokka.jpg", pilmode="RGB")[0:643, 0:643], (128, 128))
    image = ia.quokka_square(size=(128, 128))

    keypoints = [ia.Keypoint(x=34, y=15), ia.Keypoint(x=85, y=13), ia.Keypoint(x=63, y=73)] # left ear, right ear, mouth
    keypoints = [ia.KeypointsOnImage(keypoints, shape=image.shape)]

    print("[draw_per_augmenter_images] Initializing...")
    rows_augmenters = [
        (0, "Noop", [("", iaa.Noop()) for _ in sm.xrange(5)]),
        (0, "Crop\n(top, right,\nbottom, left)", [(str(vals), iaa.Crop(px=vals)) for vals in [(2, 0, 0, 0), (0, 8, 8, 0), (4, 0, 16, 4), (8, 0, 0, 32), (32, 64, 0, 0)]]),
        (0, "Pad\n(top, right,\nbottom, left)", [(str(vals), iaa.Pad(px=vals)) for vals in [(2, 0, 0, 0), (0, 8, 8, 0), (4, 0, 16, 4), (8, 0, 0, 32), (32, 64, 0, 0)]]),
        (0, "Fliplr", [(str(p), iaa.Fliplr(p)) for p in [0, 0, 1, 1, 1]]),
        (0, "Flipud", [(str(p), iaa.Flipud(p)) for p in [0, 0, 1, 1, 1]]),
        (0, "Superpixels\np_replace=1", [("n_segments=%d" % (n_segments,), iaa.Superpixels(p_replace=1.0, n_segments=n_segments)) for n_segments in [25, 50, 75, 100, 125]]),
        (0, "Superpixels\nn_segments=100", [("p_replace=%.2f" % (p_replace,), iaa.Superpixels(p_replace=p_replace, n_segments=100)) for p_replace in [0, 0.25, 0.5, 0.75, 1.0]]),
        (0, "Invert", [("p=%d" % (p,), iaa.Invert(p=p)) for p in [0, 0, 1, 1, 1]]),
        (0, "Invert\n(per_channel)", [("p=%.2f" % (p,), iaa.Invert(p=p, per_channel=True)) for p in [0.5, 0.5, 0.5, 0.5, 0.5]]),
        (0, "Add", [("value=%d" % (val,), iaa.Add(val)) for val in [-45, -25, 0, 25, 45]]),
        (0, "Add\n(per channel)", [("value=(%d, %d)" % (vals[0], vals[1],), iaa.Add(vals, per_channel=True)) for vals in [(-55, -35), (-35, -15), (-10, 10), (15, 35), (35, 55)]]),
        (0, "AddToHueAndSaturation", [("value=%d" % (val,), iaa.AddToHueAndSaturation(val)) for val in [-45, -25, 0, 25, 45]]),
        (0, "Multiply", [("value=%.2f" % (val,), iaa.Multiply(val)) for val in [0.25, 0.5, 1.0, 1.25, 1.5]]),
        (1, "Multiply\n(per channel)", [("value=(%.2f, %.2f)" % (vals[0], vals[1],), iaa.Multiply(vals, per_channel=True)) for vals in [(0.15, 0.35), (0.4, 0.6), (0.9, 1.1), (1.15, 1.35), (1.4, 1.6)]]),
        (0, "GaussianBlur", [("sigma=%.2f" % (sigma,), iaa.GaussianBlur(sigma=sigma)) for sigma in [0.25, 0.50, 1.0, 2.0, 4.0]]),
        (0, "AverageBlur", [("k=%d" % (k,), iaa.AverageBlur(k=k)) for k in [1, 3, 5, 7, 9]]),
        (0, "MedianBlur", [("k=%d" % (k,), iaa.MedianBlur(k=k)) for k in [1, 3, 5, 7, 9]]),
        (0, "BilateralBlur\nsigma_color=250,\nsigma_space=250", [("d=%d" % (d,), iaa.BilateralBlur(d=d, sigma_color=250, sigma_space=250)) for d in [1, 3, 5, 7, 9]]),
        (0, "Sharpen\n(alpha=1)", [("lightness=%.2f" % (lightness,), iaa.Sharpen(alpha=1, lightness=lightness)) for lightness in [0, 0.5, 1.0, 1.5, 2.0]]),
        (0, "Emboss\n(alpha=1)", [("strength=%.2f" % (strength,), iaa.Emboss(alpha=1, strength=strength)) for strength in [0, 0.5, 1.0, 1.5, 2.0]]),
        (0, "EdgeDetect", [("alpha=%.2f" % (alpha,), iaa.EdgeDetect(alpha=alpha)) for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]]),
        (0, "DirectedEdgeDetect\n(alpha=1)", [("direction=%.2f" % (direction,), iaa.DirectedEdgeDetect(alpha=1, direction=direction)) for direction in [0.0, 1*(360/5)/360, 2*(360/5)/360, 3*(360/5)/360, 4*(360/5)/360]]),
        (0, "AdditiveGaussianNoise", [("scale=%.2f*255" % (scale,), iaa.AdditiveGaussianNoise(scale=scale * 255)) for scale in [0.025, 0.05, 0.1, 0.2, 0.3]]),
        (0, "AdditiveGaussianNoise\n(per channel)", [("scale=%.2f*255" % (scale,), iaa.AdditiveGaussianNoise(scale=scale * 255, per_channel=True)) for scale in [0.025, 0.05, 0.1, 0.2, 0.3]]),
        (0, "Dropout", [("p=%.2f" % (p,), iaa.Dropout(p=p)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]),
        (0, "Dropout\n(per channel)", [("p=%.2f" % (p,), iaa.Dropout(p=p, per_channel=True)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]),
        (3, "CoarseDropout\n(p=0.2)", [("size_percent=%.2f" % (size_percent,), iaa.CoarseDropout(p=0.2, size_percent=size_percent, min_size=2)) for size_percent in [0.3, 0.2, 0.1, 0.05, 0.02]]),
        (0, "CoarseDropout\n(p=0.2, per channel)", [("size_percent=%.2f" % (size_percent,), iaa.CoarseDropout(p=0.2, size_percent=size_percent, per_channel=True, min_size=2)) for size_percent in [0.3, 0.2, 0.1, 0.05, 0.02]]),
        (0, "SaltAndPepper", [("p=%.2f" % (p,), iaa.SaltAndPepper(p=p)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]),
        (0, "Salt", [("p=%.2f" % (p,), iaa.Salt(p=p)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]),
        (0, "Pepper", [("p=%.2f" % (p,), iaa.Pepper(p=p)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]),
        (0, "CoarseSaltAndPepper\n(p=0.2)", [("size_percent=%.2f" % (size_percent,), iaa.CoarseSaltAndPepper(p=0.2, size_percent=size_percent, min_size=2)) for size_percent in [0.3, 0.2, 0.1, 0.05, 0.02]]),
        (0, "CoarseSalt\n(p=0.2)", [("size_percent=%.2f" % (size_percent,), iaa.CoarseSalt(p=0.2, size_percent=size_percent, min_size=2)) for size_percent in [0.3, 0.2, 0.1, 0.05, 0.02]]),
        (0, "CoarsePepper\n(p=0.2)", [("size_percent=%.2f" % (size_percent,), iaa.CoarsePepper(p=0.2, size_percent=size_percent, min_size=2)) for size_percent in [0.3, 0.2, 0.1, 0.05, 0.02]]),
        (0, "ContrastNormalization", [("alpha=%.1f" % (alpha,), iaa.ContrastNormalization(alpha=alpha)) for alpha in [0.5, 0.75, 1.0, 1.25, 1.50]]),
        (0, "ContrastNormalization\n(per channel)", [("alpha=(%.2f, %.2f)" % (alphas[0], alphas[1],), iaa.ContrastNormalization(alpha=alphas, per_channel=True)) for alphas in [(0.4, 0.6), (0.65, 0.85), (0.9, 1.1), (1.15, 1.35), (1.4, 1.6)]]),
        (0, "Grayscale", [("alpha=%.1f" % (alpha,), iaa.Grayscale(alpha=alpha)) for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]]),
        (6, "PerspectiveTransform", [("scale=%.3f" % (scale,), iaa.PerspectiveTransform(scale=scale)) for scale in [0.025, 0.05, 0.075, 0.10, 0.125]]),
        (0, "PiecewiseAffine", [("scale=%.3f" % (scale,), iaa.PiecewiseAffine(scale=scale)) for scale in [0.015, 0.03, 0.045, 0.06, 0.075]]),
        (0, "Affine: Scale", [("%.1fx" % (scale,), iaa.Affine(scale=scale)) for scale in [0.1, 0.5, 1.0, 1.5, 1.9]]),
        (0, "Affine: Translate", [("x=%d y=%d" % (x, y), iaa.Affine(translate_px={"x": x, "y": y})) for x, y in [(-32, -16), (-16, -32), (-16, -8), (16, 8), (16, 32)]]),
        (0, "Affine: Rotate", [("%d deg" % (rotate,), iaa.Affine(rotate=rotate)) for rotate in [-90, -45, 0, 45, 90]]),
        (0, "Affine: Shear", [("%d deg" % (shear,), iaa.Affine(shear=shear)) for shear in [-45, -25, 0, 25, 45]]),
        (0, "Affine: Modes", [(mode, iaa.Affine(translate_px=-32, mode=mode)) for mode in ["constant", "edge", "symmetric", "reflect", "wrap"]]),
        (0, "Affine: cval", [("%d" % (int(cval*255),), iaa.Affine(translate_px=-32, cval=int(cval*255), mode="constant")) for cval in [0.0, 0.25, 0.5, 0.75, 1.0]]),
        (
            2, "Affine: all", [
                (
                    "",
                    iaa.Affine(
                        scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                        translate_px={"x": (-32, 32), "y": (-32, 32)},
                        rotate=(-45, 45),
                        shear=(-32, 32),
                        mode=ia.ALL,
                        cval=(0.0, 1.0)
                    )
                )
                for _ in sm.xrange(5)
            ]
        ),
        (1, "ElasticTransformation\n(sigma=0.2)", [("alpha=%.1f" % (alpha,), iaa.ElasticTransformation(alpha=alpha, sigma=0.2)) for alpha in [0.1, 0.5, 1.0, 3.0, 9.0]]),
        (0, "Alpha\nwith EdgeDetect(1.0)", [("factor=%.1f" % (factor,), iaa.Alpha(factor=factor, first=iaa.EdgeDetect(1.0))) for factor in [0.0, 0.25, 0.5, 0.75, 1.0]]),
        (4, "Alpha\nwith EdgeDetect(1.0)\n(per channel)", [("factor=(%.2f, %.2f)" % (factor[0], factor[1]), iaa.Alpha(factor=factor, first=iaa.EdgeDetect(1.0), per_channel=0.5)) for factor in [(0.0, 0.2), (0.15, 0.35), (0.4, 0.6), (0.65, 0.85), (0.8, 1.0)]]),
        (15, "SimplexNoiseAlpha\nwith EdgeDetect(1.0)", [("", iaa.SimplexNoiseAlpha(first=iaa.EdgeDetect(1.0))) for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]]),
        (9, "FrequencyNoiseAlpha\nwith EdgeDetect(1.0)", [("exponent=%.1f" % (exponent,), iaa.FrequencyNoiseAlpha(exponent=exponent, first=iaa.EdgeDetect(1.0), size_px_max=16, upscale_method="linear", sigmoid=False)) for exponent in [-4, -2, 0, 2, 4]])
    ]

    print("[draw_per_augmenter_images] Augmenting...")
    rows = []
    for (row_seed, row_name, augmenters) in rows_augmenters:
        ia.seed(row_seed)
        #for img_title, augmenter in augmenters:
        #    #aug.reseed(1000)
        #    pass

        row_images = []
        row_keypoints = []
        row_titles = []
        for img_title, augmenter in augmenters:
            aug_det = augmenter.to_deterministic()
            row_images.append(aug_det.augment_image(image))
            row_keypoints.append(aug_det.augment_keypoints(keypoints)[0])
            row_titles.append(img_title)
        rows.append((row_name, row_images, row_keypoints, row_titles))

    # matplotlib drawin routine
    """
    print("[draw_per_augmenter_images] Plotting...")
    width = 8
    height = int(1.5 * len(rows_augmenters))
    fig = plt.figure(figsize=(width, height))
    grid_rows = len(rows)
    grid_cols = 1 + 5
    gs = gridspec.GridSpec(grid_rows, grid_cols, width_ratios=[2, 1, 1, 1, 1, 1])
    axes = []
    for i in sm.xrange(grid_rows):
        axes.append([plt.subplot(gs[i, col_idx]) for col_idx in sm.xrange(grid_cols)])
    fig.tight_layout()
    #fig.subplots_adjust(bottom=0.2 / grid_rows, hspace=0.22)
    #fig.subplots_adjust(wspace=0.005, hspace=0.425, bottom=0.02)
    fig.subplots_adjust(wspace=0.005, hspace=0.005, bottom=0.02)

    for row_idx, (row_name, row_images, row_keypoints, row_titles) in enumerate(rows):
        axes_row = axes[row_idx]

        for col_idx in sm.xrange(grid_cols):
            ax = axes_row[col_idx]

            ax.cla()
            ax.axis("off")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if col_idx == 0:
                ax.text(0, 0.5, row_name, color="black")
            else:
                cell_image = row_images[col_idx-1]
                cell_keypoints = row_keypoints[col_idx-1]
                cell_image_kp = cell_keypoints.draw_on_image(cell_image, size=5)
                ax.imshow(cell_image_kp)
                x = 0
                y = 145
                #ax.text(x, y, row_titles[col_idx-1], color="black", backgroundcolor="white", fontsize=6)
                ax.text(x, y, row_titles[col_idx-1], color="black", fontsize=7)


    fig.savefig("examples.jpg", bbox_inches="tight")
    #plt.show()
    """

    # simpler and faster drawing routine
    """
    output_image = ExamplesImage(128, 128, 128+64, 32)
    for (row_name, row_images, row_keypoints, row_titles) in rows:
        row_images_kps = []
        for image, keypoints in zip(row_images, row_keypoints):
            row_images_kps.append(keypoints.draw_on_image(image, size=5))
        output_image.add_row(row_name, row_images_kps, row_titles)
    imageio.imwrite("examples.jpg", output_image.draw())
    """

    # routine to draw many single files
    seen = defaultdict(lambda: 0)
    markups = []
    for (row_name, row_images, row_keypoints, row_titles) in rows:
        output_image = ExamplesImage(128, 128, 128+64, 32)
        row_images_kps = []
        for image, keypoints in zip(row_images, row_keypoints):
            row_images_kps.append(keypoints.draw_on_image(image, size=5))
        output_image.add_row(row_name, row_images_kps, row_titles)
        if "\n" in row_name:
            row_name_clean = row_name[0:row_name.find("\n")+1]
        else:
            row_name_clean = row_name
        row_name_clean = re.sub(r"[^a-z0-9]+", "_", row_name_clean.lower())
        row_name_clean = row_name_clean.strip("_")
        if seen[row_name_clean] > 0:
            row_name_clean = "%s_%d" % (row_name_clean, seen[row_name_clean] + 1)
        fp = os.path.join(IMAGES_DIR, "examples_%s.jpg" % (row_name_clean,))
        #imageio.imwrite(fp, output_image.draw())
        save(fp, output_image.draw())
        seen[row_name_clean] += 1

        markup_descr = row_name.replace('"', '') \
                               .replace("\n", " ") \
                               .replace("(", "") \
                               .replace(")", "")
        markup = '![%s](%s?raw=true "%s")' % (markup_descr, fp, markup_descr)
        markups.append(markup)

    for markup in markups:
        print(markup)


class ExamplesImage(object):
    def __init__(self, image_height, image_width, title_cell_width, subtitle_height):
        self.rows = []
        self.image_height = image_height
        self.image_width = image_width
        self.title_cell_width = title_cell_width
        self.cell_height = image_height + subtitle_height
        self.cell_width = image_width

    def add_row(self, title, images, subtitles):
        assert len(images) == len(subtitles)
        images_rs = []
        for image in images:
            images_rs.append(ia.imresize_single_image(image, (self.image_height, self.image_width)))
        self.rows.append((title, images_rs, subtitles))

    def draw(self):
        rows_drawn = [self.draw_row(title, images, subtitles) for title, images, subtitles in self.rows]
        grid = np.vstack(rows_drawn)
        return grid

    def draw_row(self, title, images, subtitles):
        title_cell = np.zeros((self.cell_height, self.title_cell_width, 3), dtype=np.uint8) + 255
        title_cell = ia.draw_text(title_cell, x=2, y=12, text=title, color=[0, 0, 0], size=16)

        image_cells = []
        for image, subtitle in zip(images, subtitles):
            image_cell = np.zeros((self.cell_height, self.cell_width, 3), dtype=np.uint8) + 255
            image_cell[0:image.shape[0], 0:image.shape[1], :] = image
            image_cell = ia.draw_text(image_cell, x=2, y=image.shape[0]+2, text=subtitle, color=[0, 0, 0], size=11)
            image_cells.append(image_cell)

        row = np.hstack([title_cell] + image_cells)
        return row


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def generate_augmenter_url(module, name):
    module = module.lower()
    name = name.lower()
    return (
        "https://imgaug.readthedocs.io/en/latest/source/"
        "overview/%s.html#%s" % (module, name)
    )


def draw_per_augmenter_videos():
    class _Descriptor(object):
        def __init__(self, module, title, augmenters, subtitles, seed=0, affects_geometry=False, comment=None, url=None):
            self.module = module
            self.title = title
            self.augmenters = augmenters
            self.subtitles = subtitles
            self.seed = seed
            self.affects_geometry = affects_geometry
            self.comment = comment
            self._url = url

        @property
        def title_markup(self):
            return self.title.replace('"', '') \
                             .replace("\n", " ") \
                             .replace("(", "") \
                             .replace(")", "")

        @classmethod
        def from_augsubs(cls, module, title, augsubs, seed=0, affects_geometry=False, comment=None, url=None):
            return _Descriptor(module=module,
                               title=title,
                               augmenters=[el[1] for el in augsubs],
                               subtitles=[el[0] for el in augsubs],
                               seed=seed,
                               affects_geometry=affects_geometry,
                               comment=comment,
                               url=url)

        @property
        def url(self):
            if self._url is None:
                module = self.module
                name = self.title.replace(":", "\n").split("\n")[0]
                return generate_augmenter_url(module, name)
            return self._url

        def generate_frames(self, image, keypoints, bounding_boxes, polygons, heatmap, segmap, subtitle_height):
            frames_images = []
            frames_kps = []
            frames_bbs = []
            frames_heatmap = []
            frames_segmap = []
            any_subtitle = any([len(subtitle) > 0 for subtitle in self.subtitles])
            for i, (augmenter, subtitle) in enumerate(zip(self.augmenters, self.subtitles)):
                # print("seeding", augmenter.name, self.seed+i)
                augmenter.localize_random_state_(recursive=True)
                augmenter.seed_(self.seed+i)

                def _subt(img, toptitle):
                    if self.affects_geometry:
                        #return self._draw_cell(img, subtitle, subtitle_height if any_subtitle else 0, toptitle, 16)
                        return self._draw_cell(img, subtitle, subtitle_height, toptitle, 16)
                    else:
                        #return self._draw_cell(img, subtitle, subtitle_height if any_subtitle else 0, "", 0)
                        return self._draw_cell(img, subtitle, subtitle_height, "", 16)
                aug_det = augmenter.to_deterministic()
                image_aug = aug_det.augment_image(image)

                if self.affects_geometry is not False:
                    affects_geometry = self.affects_geometry
                    if affects_geometry is True:
                        affects_geometry = ["keypoints", "bounding_boxes",
                                            "polygons", "heatmaps",
                                            "segmentation_maps"]
                    kps_aug = aug_det.augment_keypoints([keypoints])[0] if "keypoints" in affects_geometry else None
                    bbs_aug = aug_det.augment_bounding_boxes([bounding_boxes])[0] if "bounding_boxes" in affects_geometry else None
                    polys_aug = aug_det.augment_polygons([polygons])[0] if "polygons" in affects_geometry else None
                    heatmap_aug = aug_det.augment_heatmaps([heatmap])[0] if "heatmaps" in affects_geometry else None
                    segmap_aug = aug_det.augment_segmentation_maps([segmap])[0] if "segmentation_maps" in affects_geometry else None

                    coords_subt = ["IMG"]
                    image_with_coords = image_aug
                    if kps_aug is not None:
                        image_with_coords = kps_aug.draw_on_image(image_aug, size=5)
                        coords_subt.append("KPs")
                    if bbs_aug is not None:
                        image_with_coords = bbs_aug.draw_on_image(image_with_coords)
                        coords_subt.append("BBs")
                    if polys_aug is not None:
                        image_with_coords = polys_aug.draw_on_image(
                            image_with_coords,
                            color_lines=(0, 128, 0),
                            color_points=(0, 128, 0),
                            alpha=0,
                            alpha_lines=0.5,
                            alpha_points=1.0)
                        coords_subt.append("Polys")

                    image_with_coordsaug = _subt(
                        image_with_coords,
                        ", ".join(coords_subt)
                    )
                    frames_images.append(image_with_coordsaug)
                    #frames_kps.append(_subt(kps_aug.draw_on_image(image_aug, size=5), "keypoints"))
                    #frames_bbs.append(_subt(bbs_aug.draw_on_image(image_aug), "bounding boxes"))
                    #frames_kps.append(_subt(
                    #    bbs_aug.draw_on_image(kps_aug.draw_on_image(image_aug, size=5)),
                    #    "Keypoints + BBs"
                    #))
                    frames_heatmap.append(_subt(heatmap_aug.draw_on_image(image_aug)[0], "Heatmaps"))
                    frames_segmap.append(_subt(segmap_aug.draw_on_image(image_aug)[0], "Segmentation Maps"))
                else:
                    frames_images.append(_subt(image_aug, "Images"))
            return frames_images, frames_kps, frames_bbs, frames_heatmap, frames_segmap

        @classmethod
        def _draw_cell(cls, image, subtitle, subtitle_height, toptitle, toptitle_height):
            cell_height, cell_width = image.shape[0:2]
            image_cell = np.zeros((toptitle_height + cell_height + subtitle_height, cell_width, 3), dtype=np.uint8) + 255
            image_cell[toptitle_height:toptitle_height+image.shape[0], 0:image.shape[1], :] = image
            image_cell = ia.draw_text(image_cell, x=2, y=toptitle_height + image.shape[0]+2, text=subtitle, color=[0, 0, 0], size=9)
            if toptitle != "":
                image_cell = ia.draw_text(image_cell, x=2, y=2, text=toptitle, color=[0, 0, 0], size=9)

            return image_cell

    class _MarkdownTableCell(object):
        def __init__(self, descriptor, markup_images, markup_kps, markup_bbs, markup_hm, markup_segmap):
            self.descriptor = descriptor
            self.markup_images = markup_images
            self.markup_kps = markup_kps
            self.markup_bbs = markup_bbs
            self.markup_hm = markup_hm
            self.markup_segmap = markup_segmap

        @property
        def colspan(self):
            #only_images = len(self.markup_kps) == 0 and len(self.markup_bbs) == 0 and len(self.markup_hm) == 0 and len(self.markup_segmap) == 0
            only_images = not self.descriptor.affects_geometry
            return 1 if only_images else 2

        def render_title(self):
            url = self.descriptor.url
            if url is None:
                return '<td colspan="%d"><sub>%s</sub></td>' % (self.colspan, self.descriptor.title.replace("\n", "<br/>"))
            else:
                title1 = self.descriptor.title
                title2 = ""
                if "\n" in title1:
                    title1 = self.descriptor.title.split("\n")[0]
                    title2 = "<br/>" + "<br/>".join(self.descriptor.title.split("\n")[1:])

                return '<td colspan="%d"><sub><a href=\"%s\">%s</a>%s</sub></td>' % (
                    self.colspan, url, title1, title2)

        def render_main(self):
            #return '<td colspan="%d">\n\n%s%s%s%s%s\n\n</td>' % (self.colspan, self.markup_images, self.markup_kps, self.markup_bbs, self.markup_hm, self.markup_segmap)
            return '<td colspan="%d">%s%s%s%s%s</td>' % (self.colspan, self.markup_images, self.markup_kps, self.markup_bbs, self.markup_hm, self.markup_segmap)

        def render_comment(self):
            if self.descriptor.comment is not None:
                #return '<td colspan="%d">\n<small>\n\n%s\n\n</small>\n</td>' % (self.colspan, self.descriptor.comment,)
                return '<td colspan="%d"><sub>%s</sub></td>' % (self.colspan, self.descriptor.comment,)
            else:
                return '<td colspan="%d">&nbsp;</td>' % (self.colspan,)

    class _MarkdownTableSeeAlsoUrl(object):
        def __init__(self, url, linktext):
            self.url = url
            self.linktext = linktext

        def render(self):
            return "<a href=\"%s\">%s</a>" % (self.url, self.linktext)

        @classmethod
        def from_augmenter(cls, module, name):
            return _MarkdownTableSeeAlsoUrl(
                generate_augmenter_url(module, name), name)

    class _MarkdownTableSeeAlsoUrlList(object):
        def __init__(self, urls):
            self.urls = urls

        @property
        def colspan(self):
            return 5

        def render_title(self):
            return ""

        def render_main(self):
            urls_str = ", ".join([url.render() for url in self.urls])
            return "See also: %s" % (urls_str,)

        def render_comment(self):
            return ""

    class _MarkdownTable(object):
        ROW_SIZE = 5  # in columns

        def __init__(self):
            self.cells = []

        def render(self):
            current_module = None
            first_row_in_module = True
            markup = []
            cells = self.cells
            while len(cells) > 0:
                current_row_size = 0
                row_title = []
                row_main = []
                row_comment = []
                any_comment = False
                while current_row_size < self.ROW_SIZE and len(cells) > 0:
                    cell = cells[0]
                    if isinstance(cell, _MarkdownTableSeeAlsoUrlList):
                        pass  # should always be at the end of a module
                    elif current_module is None:
                        current_module = cell.descriptor.module
                    elif current_module != cell.descriptor.module:
                        if current_row_size == 0:
                            current_module = cell.descriptor.module
                            first_row_in_module = True
                        else:
                            break
                    if cell.colspan > (self.ROW_SIZE - current_row_size):
                        break
                    row_title.append(cell.render_title())
                    row_main.append(cell.render_main())
                    row_comment.append(cell.render_comment())
                    if (not isinstance(cell, _MarkdownTableSeeAlsoUrlList)
                            and cell.descriptor.comment is not None
                            and len(cell.descriptor.comment) > 0):
                        any_comment = True
                    current_row_size += cell.colspan
                    cells = cells[1:]

                while current_row_size < self.ROW_SIZE:
                    row_title.append("<td>&nbsp;</td>")
                    row_main.append("<td>&nbsp;</td>")
                    row_comment.append("<td>&nbsp;</td>")
                    current_row_size += 1

                if first_row_in_module:
                    #markup.append('<tr>\n<td colspan="3">\n\n**%s**\n\n</td>\n</tr>' % (current_module,))
                    markup.append('<tr><td colspan="%d"><strong>%s</strong></td></tr>' % (self.ROW_SIZE, current_module,))
                    first_row_in_module = False
                markup.append("<tr>\n%s\n</tr>\n<tr>\n%s\n</tr>%s" % (
                    "\n".join(row_title),
                    "\n".join(row_main),
                    "" if not any_comment else "\n<tr>\n%s\n</tr>" % ("\n".join(row_comment),)
                ))

            return "<table>\n\n%s\n\n</table>" % ("\n".join(markup),)

        def append(self, cell):
            self.cells.append(cell)

    print("[draw_per_augmenter_videos] Loading image...")
    # image = ia.imresize_single_image(imageio.imread("quokka.jpg", pilmode="RGB")[0:643, 0:643], (128, 128))
    h, w = 100, 100
    h_subtitle = 32
    image = ia.quokka_square(size=(h, w))
    keypoints = ia.quokka_keypoints(size=(h, w), extract="square")
    bbs = ia.quokka_bounding_boxes(size=(h, w), extract="square")
    polygons = ia.quokka_polygons(size=(h, w), extract="square")
    heatmap = ia.quokka_heatmap(size=(h, w), extract="square")
    segmap = ia.quokka_segmentation_map(size=(h, w), extract="square")

    image_landscape = imageio.imread("https://upload.wikimedia.org/wikipedia/commons/8/89/Kukle%2CCzech_Republic..jpg", format="jpg")
    # os.path.join(os.path.dirname(os.path.abspath(__file__)), "landscape.jpg")
    image_landscape = ia.imresize_single_image(image_landscape, (96, 128))
    image_valley = imageio.imread(os.path.join(INPUT_IMAGES_DIR, "Pahalgam_Valley.jpg"))
    image_valley = ia.imresize_single_image(image_valley, (96, 128))
    image_vangogh = imageio.imread(os.path.join(INPUT_IMAGES_DIR, "1280px-Vincent_Van_Gogh_-_Wheatfield_with_Crows.jpg"))
    image_vangogh = ia.imresize_single_image(image_vangogh, (96, 128))

    print("[draw_per_augmenter_videos] Initializing...")
    descriptors = []
    # ###
    # meta
    # ###
    descriptors.extend([
        _Descriptor.from_augsubs(
            "meta",
            "Identity",
            [("", iaa.Identity()) for _ in sm.xrange(1)]),
        _Descriptor.from_augsubs(
            "meta",
            "ChannelShuffle",
            [("p=1.0", iaa.ChannelShuffle(p=1.0)) for _ in sm.xrange(5)]
        ),
        _MarkdownTableSeeAlsoUrlList([
            _MarkdownTableSeeAlsoUrl.from_augmenter("meta", "Sequential"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("meta", "SomeOf"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("meta", "OneOf"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("meta", "Sometimes"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("meta", "WithChannels"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("meta", "Lambda"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("meta", "AssertLambda"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("meta", "AssertShape"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("meta", "RemoveCBAsByOutOfImageFraction"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("meta", "ClipCBAsToImagePlanes"),
        ])
    ])

    # ###
    # arithmetic
    # ###
    descriptors.extend([
        _Descriptor.from_augsubs(
            "arithmetic",
            "Add",
            [("value=%d" % (val,), iaa.Add(val)) for val in [-45, -25, 0, 25, 45]]
        ),
        _Descriptor.from_augsubs(
            "arithmetic",
            "Add\n(per_channel=True)",
            [("value=(%d, %d)" % (vals[0], vals[1],), iaa.Add(vals, per_channel=True))
             for vals in [(-55, -35), (-35, -15), (-10, 10), (15, 35), (35, 55)]]
        ),
        _Descriptor.from_augsubs(
            "arithmetic",
            "AdditiveGaussianNoise",
            [("scale=%.2f*255" % (scale,), iaa.AdditiveGaussianNoise(scale=scale * 255))
             for scale in [0.025, 0.05, 0.1, 0.2, 0.3]]
        ),
        _Descriptor.from_augsubs(
            "arithmetic",
            "AdditiveGaussianNoise\n(per_channel=True)",
            [("scale=%.2f*255" % (scale,), iaa.AdditiveGaussianNoise(scale=scale * 255, per_channel=True))
             for scale in [0.025, 0.05, 0.1, 0.2, 0.3]]
        ),
        # _Descriptor.from_augsubs(
        #     "arithmetic",
        #     "AdditiveLaplaceNoise",
        #     [("scale=%.2f*255" % (scale,), iaa.AdditiveLaplaceNoise(scale=scale * 255))
        #      for scale in [0.025, 0.05, 0.1, 0.2, 0.3]]
        # ),
        # _Descriptor.from_augsubs(
        #     "arithmetic",
        #     "AdditiveLaplaceNoise\n(per_channel=True)",
        #     [("scale=%.2f*255" % (scale,), iaa.AdditiveLaplaceNoise(scale=scale * 255, per_channel=True))
        #      for scale in [0.025, 0.05, 0.1, 0.2, 0.3]]
        # ),
        # _Descriptor.from_augsubs(
        #     "arithmetic",
        #     "AdditivePoissonNoise",
        #     [("lam=%.2f" % (lam,), iaa.AdditivePoissonNoise(lam=lam))
        #      for lam in [4.0, 8.0, 16.0, 32.0, 64.0]]
        # ),
        # _Descriptor.from_augsubs(
        #     "arithmetic",
        #     "AdditivePoissonNoise\n(per_channel=True)",
        #     [("lam=%.2f" % (lam,), iaa.AdditivePoissonNoise(lam=lam, per_channel=True))
        #      for lam in [4.0, 8.0, 16.0, 32.0, 64.0]]
        # ),
        _Descriptor.from_augsubs(
            "arithmetic",
            "Multiply",
            [("value=%.2f" % (val,), iaa.Multiply(val))
             for val in [0.25, 0.5, 1.0, 1.25, 1.5]]
        ),
        # _Descriptor.from_augsubs(
        #     "arithmetic",
        #     "Multiply\n(per_channel=True)",
        #     [("value=(%.2f, %.2f)" % (vals[0], vals[1],), iaa.Multiply(vals, per_channel=True))
        #      for vals in [(0.15, 0.35), (0.4, 0.6), (0.9, 1.1), (1.15, 1.35), (1.4, 1.6)]]
        # ),
        # MultiplyElementwise
        _Descriptor.from_augsubs(
            "arithmetic",
            "Cutout",
            [
                ("nb_iterations=1", iaa.Cutout(nb_iterations=1, size=(0.1, 0.3), fill_mode="constant")),
                ("nb_iterations=1", iaa.Cutout(nb_iterations=1, size=(0.1, 0.3), fill_mode="constant")),
                ("nb_iterations=2", iaa.Cutout(nb_iterations=2, size=(0.1, 0.3), fill_mode="constant")),
                ("nb_iterations=2", iaa.Cutout(nb_iterations=2, size=(0.1, 0.3), fill_mode="constant")),
                ("non-squared", iaa.Cutout(nb_iterations=2, size=(0.1, 0.3), fill_mode="constant", squared=False)),
                ("non-squared", iaa.Cutout(nb_iterations=2, size=(0.1, 0.3), fill_mode="constant", squared=False)),
                ("RGB colors", iaa.Cutout(nb_iterations=2, size=(0.1, 0.3), cval=(0, 255), fill_mode="constant", fill_per_channel=True)),
                ("RGB colors", iaa.Cutout(nb_iterations=2, size=(0.1, 0.3), cval=(0, 255), fill_mode="constant", fill_per_channel=True)),
                ("gaussian", iaa.Cutout(nb_iterations=2, size=(0.1, 0.3), fill_mode="gaussian")),
                ("gaussian", iaa.Cutout(nb_iterations=2, size=(0.1, 0.3), fill_mode="gaussian"))
            ]
        ),
        _Descriptor.from_augsubs(
            "arithmetic",
            "Dropout",
            [("p=%.2f" % (p,), iaa.Dropout(p=p))
             for p in [0.025, 0.05, 0.1, 0.2, 0.4]]
        ),
        # _Descriptor.from_augsubs(
        #     "arithmetic",
        #     "Dropout\n(per_channel=True)",
        #     [("p=%.2f" % (p,), iaa.Dropout(p=p, per_channel=True))
        #      for p in [0.025, 0.05, 0.1, 0.2, 0.4]]
        # ),
        _Descriptor.from_augsubs(
            "arithmetic",
            "CoarseDropout\n(p=0.2)",
            [("size_percent=%.2f" % (size_percent,), iaa.CoarseDropout(p=0.2, size_percent=size_percent, min_size=2))
             for size_percent in [0.3, 0.2, 0.1, 0.05, 0.02]]
        ),
        _Descriptor.from_augsubs(
            "arithmetic",
            "CoarseDropout\n(p=0.2, per_channel=True)",
            [("size_percent=%.2f" % (size_percent,), iaa.CoarseDropout(p=0.2, size_percent=size_percent, per_channel=True, min_size=2))
             for size_percent in [0.3, 0.2, 0.1, 0.05, 0.02]]
        ),
        _Descriptor.from_augsubs(
            "arithmetic",
            "Dropout2d",
            [("p=%.2f" % (p,), iaa.Dropout2d(p=p))
             for p in [0.5, 0.5, 0.5, 0.5, 0.5]]
        ),
        # TotalDropout -> see also
        # ReplaceElementwise -> see also
        # _Descriptor.from_augsubs(
        #     "arithmetic",
        #     "ImpulseNoise",
        #     [("p=%.2f" % (p,), iaa.ImpulseNoise(p=p)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]
        # ),
        _Descriptor.from_augsubs(
            "arithmetic",
            "SaltAndPepper",
            [("p=%.2f" % (p,), iaa.SaltAndPepper(p=p)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]
        ),
        _Descriptor.from_augsubs(
            "arithmetic",
            "CoarseSaltAndPepper\n(p=0.2)",
            [("size_percent=%.2f" % (size_percent,), iaa.CoarseSaltAndPepper(p=0.2, size_percent=size_percent, min_size=2))
             for size_percent in [0.3, 0.2, 0.1, 0.05, 0.02]]
        ),
        # _Descriptor.from_augsubs(
        #     "arithmetic",
        #     "Salt",
        #     [("p=%.2f" % (p,), iaa.Salt(p=p)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]
        # ),
        # _Descriptor.from_augsubs(
        #     "arithmetic",
        #     "Pepper",
        #     [("p=%.2f" % (p,), iaa.Pepper(p=p)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]
        # ),
        # _Descriptor.from_augsubs(
        #     "arithmetic",
        #     "CoarseSalt\n(p=0.2)",
        #     [("size_percent=%.2f" % (size_percent,), iaa.CoarseSalt(p=0.2, size_percent=size_percent, min_size=2))
        #      for size_percent in [0.3, 0.2, 0.1, 0.05, 0.02]]
        # ),
        # _Descriptor.from_augsubs(
        #     "arithmetic",
        #     "CoarsePepper\n(p=0.2)",
        #     [("size_percent=%.2f" % (size_percent,), iaa.CoarsePepper(p=0.2, size_percent=size_percent, min_size=2))
        #      for size_percent in [0.3, 0.2, 0.1, 0.05, 0.02]]
        # ),
        _Descriptor.from_augsubs(
            "arithmetic",
            "Invert",
            [("p=%d" % (p,), iaa.Invert(p=p)) for p in [0, 1]]
        ),
        # _Descriptor.from_augsubs(
        #     "arithmetic",
        #     "Invert\n(per_channel=True)",
        #     [("p=%.2f" % (p,), iaa.Invert(p=p, per_channel=True)) for p in [0.5, 0.5, 0.5, 0.5, 0.5]]
        # ),
        _Descriptor.from_augsubs(
            "arithmetic",
            "Solarize",
            [("p=%d" % (p,), iaa.Solarize(p=p)) for p in [0, 1]]
        ),
        #_Descriptor.from_augsubs(
        #    "arithmetic",
        #    "ContrastNormalization",
        #    [("alpha=%.1f" % (alpha,), iaa.ContrastNormalization(alpha=alpha)) for alpha in [0.5, 0.75, 1.0, 1.25, 1.50]]
        #),
        #_Descriptor.from_augsubs(
        #    "arithmetic",
        #    "ContrastNormalization\n(per channel)",
        #    [("alpha=(%.2f, %.2f)" % (alphas[0], alphas[1],), iaa.ContrastNormalization(alpha=alphas, per_channel=True))
        #     for alphas in [(0.4, 0.6), (0.65, 0.85), (0.9, 1.1), (1.15, 1.35), (1.4, 1.6)]]
        #),
        _Descriptor.from_augsubs(
            "arithmetic",
            "JpegCompression",
            [("compression=%d" % (compression,), iaa.JpegCompression(compression=compression))
             for compression in np.linspace(50, 100, num=5)]
        ),
        _MarkdownTableSeeAlsoUrlList([
            _MarkdownTableSeeAlsoUrl.from_augmenter("arithmetic", "AddElementwise"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("arithmetic", "AdditiveLaplaceNoise"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("arithmetic", "AdditivePoissonNoise"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("arithmetic", "MultiplyElementwise"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("arithmetic", "TotalDropout"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("arithmetic", "ReplaceElementwise"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("arithmetic", "ImpulseNoise"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("arithmetic", "Salt"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("arithmetic", "Pepper"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("arithmetic", "CoarseSalt"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("arithmetic", "CoarsePepper"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("arithmetic", "Solarize"),
        ])
    ])

    # ###
    # artistic
    # ###
    descriptors.extend([
        _Descriptor.from_augsubs(
            "artistic",
            "Cartoon",
            [("off", iaa.Identity()),
             ("on", iaa.Cartoon())]
        ),
    ])

    # ###
    # blend
    # ###
    descriptors.extend([
        _Descriptor.from_augsubs(
            "blend",
            "BlendAlpha\nwith EdgeDetect(1.0)",
            [("factor=%.1f" % (factor,), iaa.BlendAlpha(factor=factor, foreground=iaa.EdgeDetect(1.0)))
             for factor in [0.0, 0.25, 0.5, 0.75, 1.0]]
        ),
        # _Descriptor.from_augsubs(
        #     "blend",
        #     "Alpha\nwith EdgeDetect(1.0)\n(per_channel=True)",
        #     [("factor=(%.2f, %.2f)" % (factor[0], factor[1]), iaa.Alpha(factor=factor, first=iaa.EdgeDetect(1.0), per_channel=0.5))
        #      for factor in [(0.0, 0.2), (0.15, 0.35), (0.4, 0.6), (0.65, 0.85), (0.8, 1.0)]],
        #     seed=4
        # ),
        # AlphaElementwise
        _Descriptor.from_augsubs(
            "blend",
            "BlendAlphaSimplexNoise\nwith EdgeDetect(1.0)",
            [("", iaa.BlendAlphaSimplexNoise(foreground=iaa.EdgeDetect(1.0))) for _ in range(7)],
            seed=15
        ),
        _Descriptor.from_augsubs(
            "blend",
            "BlendAlphaFrequencyNoise\nwith EdgeDetect(1.0)",
            [("exponent=%.1f" % (exponent,), iaa.BlendAlphaFrequencyNoise(exponent=exponent, foreground=iaa.EdgeDetect(1.0), size_px_max=16, upscale_method="linear", sigmoid=False))
             for exponent in [-4, -2, 0, 2, 4]]
        ),
        _Descriptor.from_augsubs(
            "blend",
            "BlendAlphaSomeColors\nwith RemoveSaturation(1.0)",
            [("", iaa.BlendAlphaSomeColors(iaa.RemoveSaturation(1.0), seed=2))
             for _ in range(8)]
        ),
        # _Descriptor.from_augsubs(
        #     "blend",
        #     "BlendAlphaVerticalLinearGradient\nwith Clouds()",
        #     [("", iaa.BlendAlphaVerticalLinearGradient(iaa.Clouds()))
        #      for _ in range(5)]
        # ),
        _Descriptor.from_augsubs(
            "blend",
            "BlendAlphaRegularGrid\nwith Multiply((0.0, 0.5))",
            [("", iaa.BlendAlphaRegularGrid(
                nb_rows=(2, 8),
                nb_cols=(2, 8),
                foreground=iaa.Multiply((0.0, 0.5)),
                alpha=(0.0, 1.0)))
             for _ in range(5)]
        ),
        _MarkdownTableSeeAlsoUrlList([
            _MarkdownTableSeeAlsoUrl.from_augmenter("blend", "BlendAlphaMask"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("blend", "BlendAlphaElementwise"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("blend", "BlendAlphaVerticalLinearGradient"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("blend", "BlendAlphaHorizontalLinearGradient"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("blend", "BlendAlphaSegMapClassIds"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("blend", "BlendAlphaBoundingBoxes"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("blend", "BlendAlphaCheckerboard"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("blend", "SomeColorsMaskGen"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("blend", "HorizontalLinearGradientMaskGen"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("blend", "VerticalLinearGradientMaskGen"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("blend", "RegularGridMaskGen"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("blend", "CheckerboardMaskGen"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("blend", "SegMapClassIdsMaskGen"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("blend", "BoundingBoxesMaskGen"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("blend", "InvertMaskGen")
        ])
    ])

    # ###
    # blur
    # ###
    descriptors.extend([
        _Descriptor.from_augsubs(
            "blur",
            "GaussianBlur",
            [("sigma=%.2f" % (sigma,), iaa.GaussianBlur(sigma=sigma))
             for sigma in [0.25, 0.50, 1.0, 2.0, 4.0]]
        ),
        _Descriptor.from_augsubs(
            "blur",
            "AverageBlur",
            [("k=%d" % (k,), iaa.AverageBlur(k=k))
             for k in [1, 3, 5, 7, 9]]
        ),
        _Descriptor.from_augsubs(
            "blur",
            "MedianBlur",
            [("k=%d" % (k,), iaa.MedianBlur(k=k))
             for k in [1, 3, 5, 7, 9]]
        ),
        _Descriptor.from_augsubs(
            "blur",
            "BilateralBlur\n(sigma_color=250,\nsigma_space=250)",
            [("d=%d" % (d,), iaa.BilateralBlur(d=d, sigma_color=250, sigma_space=250))
             for d in [1, 3, 5, 7, 9]]
        ),
        _Descriptor.from_augsubs(
            "blur",
            "MotionBlur\n(angle=0)",
            [("k=%d" % (k,), iaa.MotionBlur(k=k, angle=0)) for k in [3, 5, 7, 11, 13]]
        ),
        _Descriptor.from_augsubs(
            "blur",
            "MotionBlur\n(k=5)",
            [("angle=%d" % (angle,), iaa.MotionBlur(k=5, angle=angle))
             for angle in np.linspace(0, 360-360/5, num=5)]
        ),
        _Descriptor.from_augsubs(
            "blur",
            "MeanShiftBlur",
            [("",
              iaa.MeanShiftBlur())
             for _ in range(5)]
        ),
    ])

    # ####
    # collections
    # ####
    descriptors.extend([
        _Descriptor.from_augsubs(
            "collections",
            "RandAugment",
            [("n=2, m=(6, 12)",
              iaa.RandAugment(n=2, m=(6, 12)))
             for _ in range(5)]
        )
    ])

    # ####
    # color
    # ####
    descriptors.extend([
        # WithColorspace -> see also
        # WithBrightnessChannels -> see also
        _Descriptor.from_augsubs(
            "color",
            "MultiplyAndAddToBrightness",
            [("mul=%.1f, add=%d" % (mul, add),
              iaa.MultiplyAndAddToBrightness(mul=mul, add=add))
             for mul, add in [(1.0, 0), (1.0, -30), (1.0, 30), (0.5, 0), (1.5, 0)]]
        ),
        # MultiplyBrightness -> see also
        # AddToBrightness -> see also
        # WithHueAndSaturation -> see also
        _Descriptor.from_augsubs(
            "color",
            "MultiplyHueAndSaturation",
            [("mul=%.2f" % (mul,), iaa.MultiplyHueAndSaturation(mul=mul)) for mul in [0.5, 0.75, 1.0, 1.25, 1.5]]
        ),
        _Descriptor.from_augsubs(
            "color",
            "MultiplyHue",
            [("mul=%.2f" % (mul,), iaa.MultiplyHue(mul=mul)) for mul in [-1.0, -0.5, 0.0, 0.5, 1.0]]
        ),
        _Descriptor.from_augsubs(
            "color",
            "MultiplySaturation",
            [("mul=%.2f" % (mul,), iaa.MultiplySaturation(mul=mul)) for mul in [0.0, 0.5, 1.0, 1.5, 2.0]]
        ),
        # RemoveSaturation -> further below
        _Descriptor.from_augsubs(
            "color",
            "AddToHueAndSaturation",
            [("hue=%d, sat=%d" % (hue, sat), iaa.AddToHueAndSaturation(value_hue=hue, value_saturation=sat))
             for hue, sat in [(0, 0), (-45, 0), (45, 0), (0, -45), (0, 45)]]
        ),
        # _Descriptor.from_augsubs(
        #     "color",
        #     "AddToHue",
        #     [("value=%d" % (val,), iaa.AddToHue(val)) for val in [-45, -25, 0, 25, 45]]
        # ),
        # _Descriptor.from_augsubs(
        #     "color",
        #     "AddToSaturation",
        #     [("value=%d" % (val,), iaa.AddToSaturation(val)) for val in [-45, -25, 0, 25, 45]]
        # ),
        # ChangeColorspace -> see also
        _Descriptor.from_augsubs(
            "color",
            "Grayscale",
            [("alpha=%.2f" % (alpha,), iaa.Grayscale(alpha=alpha)) for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]]
        ),
        _Descriptor.from_augsubs(
            "color",
            "RemoveSaturation",
            [("mul=%.2f" % (mul,), iaa.RemoveSaturation(mul=mul)) for mul in [0.0, 0.25, 0.5, 0.75, 1.0]]
        ),
        _Descriptor.from_augsubs(
            "color",
            "ChangeColorTemperature",
            [("kelvin=%d" % (kelvin,), iaa.ChangeColorTemperature(kelvin)) for kelvin in [1000, 2000, 4000, 8000, 16000]]
        ),
        _Descriptor.from_augsubs(
            "color",
            "KMeansColorQuantization\n(to_colorspace=RGB)",
            [("n_colors=%d\n" % (n_colors,), iaa.KMeansColorQuantization(n_colors=n_colors, to_colorspace=iaa.CSPACE_RGB)) for n_colors in [2, 4, 8, 16, 32]]
        ),
        _Descriptor.from_augsubs(
            "color",
            "UniformColorQuantization\n(to_colorspace=RGB)",
            [("n_colors=%d" % (n_colors,), iaa.UniformColorQuantization(n_colors=n_colors, to_colorspace=iaa.CSPACE_RGB)) for n_colors in [2, 4, 8, 16, 32]]
        ),
        # Posterize -> see also
        _MarkdownTableSeeAlsoUrlList([
            _MarkdownTableSeeAlsoUrl.from_augmenter("color", "WithColorspace"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("color", "WithBrightnessChannels"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("color", "MultiplyBrightness"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("color", "AddToBrightness"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("color", "WithHueAndSaturation"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("color", "AddToHue"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("color", "AddToSaturation"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("color", "ChangeColorspace"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("color", "Posterize"),
        ])
    ])

    # ####
    # contrast
    # ####
    descriptors.extend([
        _Descriptor.from_augsubs(
            "contrast",
            "GammaContrast",
            [("gamma=%.2f" % (gamma,), iaa.GammaContrast(gamma=gamma)) for gamma in np.linspace(0.5, 1.75, num=5)]
        ),
        _Descriptor.from_augsubs(
            "contrast",
            "GammaContrast\n(per_channel=True)",
            [("gamma=(0.5, 1.75)", iaa.GammaContrast(gamma=(0.5, 1.75), per_channel=True)) for _ in range(5)]
        ),
        _Descriptor.from_augsubs(
            "contrast",
            "SigmoidContrast\n(cutoff=0.5)",
            [("gain=%.1f" % (gain,), iaa.SigmoidContrast(gain=gain, cutoff=0.5))
             for gain in np.linspace(5, 17.5, num=5)]
        ),
        _Descriptor.from_augsubs(
            "contrast",
            "SigmoidContrast\n(gain=10)",
            [("cutoff=%.2f" % (cutoff,), iaa.SigmoidContrast(gain=10, cutoff=cutoff))
             for cutoff in np.linspace(0.0, 1.0, num=5)]
        ),
        # _Descriptor.from_augsubs(
        #     "contrast",
        #     "SigmoidContrast\n(per_channel=True)",
        #     [("gain=(5, 15),\ncutoff=(0.0, 1.0)", iaa.SigmoidContrast(gain=(5, 15), cutoff=(0.0, 1.0), per_channel=True))
        #      for _ in range(5)]
        # ),
        _Descriptor.from_augsubs(
            "contrast",
            "LogContrast",
            [("gain=%.2f" % (gain,), iaa.LogContrast(gain=gain)) for gain in np.linspace(0.5, 1.0, num=5)]
        ),
        # _Descriptor.from_augsubs(
        #     "contrast",
        #     "LogContrast\n(per_channel=True)",
        #     [("gain=(0.5, 1.0)", iaa.LogContrast(gain=(0.5, 1.0), per_channel=True)) for _ in range(5)]
        # ),
        _Descriptor.from_augsubs(
            "contrast",
            "LinearContrast",
            [("alpha=%.2f" % (alpha,), iaa.LinearContrast(alpha=alpha)) for alpha in np.linspace(0.25, 1.75, num=5)]
        ),
        # _Descriptor.from_augsubs(
        #     "contrast",
        #     "LinearContrast\n(per_channel=True)",
        #     [("alpha=(0.25, 1.75)", iaa.LinearContrast(alpha=(0.25, 1.75), per_channel=True)) for _ in range(5)]
        # ),
        _Descriptor.from_augsubs(
            "contrast",
            "AllChannels-\nHistogramEqualization",
            [("", iaa.AllChannelsHistogramEqualization()) for _ in range(1)],
            url=generate_augmenter_url("contrast", "AllChannelsHistogramEqualization")
        ),
        _Descriptor.from_augsubs(
            "contrast",
            "HistogramEqualization",
            [("to_colorspace=%s" % (to_colorspace,), iaa.HistogramEqualization(to_colorspace=to_colorspace))
             for to_colorspace
             in [iaa.HistogramEqualization.Lab, iaa.HistogramEqualization.HSV, iaa.HistogramEqualization.HLS]]
        ),
        _Descriptor.from_augsubs(
            "contrast",
            "AllChannelsCLAHE",
            [("clip_limit=%.1f" % (clip_limit,), iaa.AllChannelsCLAHE(clip_limit=clip_limit))
             for clip_limit
             in np.linspace(0.1, 8.0, num=5)]
        ),
        # _Descriptor.from_augsubs(
        #     "contrast",
        #     "AllChannelsCLAHE\n(per_channel=True)",
        #     [("clip_limit=(1, 20)", iaa.AllChannelsCLAHE(clip_limit=(1, 20), per_channel=True)) for _ in range(5)],
        #     seed=4
        # ),
        _Descriptor.from_augsubs(
            "contrast",
            "CLAHE",
            [("clip_limit=%.1f,\nto_colorspace=%s" % (clip_limit, to_colorspace),
              iaa.CLAHE(clip_limit=clip_limit, to_colorspace=to_colorspace))
             for to_colorspace, clip_limit
             in zip([iaa.CLAHE.Lab] * 5, np.linspace(0.1, 8.0, num=5))]
        ),
        _MarkdownTableSeeAlsoUrlList([
            _MarkdownTableSeeAlsoUrl.from_augmenter("contrast", "Equalize"),
        ])
    ])

    # ###
    # convolutional
    # ###
    descriptors.extend([
        # Convolve
        _Descriptor.from_augsubs(
            "convolutional",
            "Sharpen\n(alpha=1)",
            [("lightness=%.2f" % (lightness,), iaa.Sharpen(alpha=1, lightness=lightness))
             for lightness in [0, 0.5, 1.0, 1.5, 2.0]]
        ),
        _Descriptor.from_augsubs(
            "convolutional",
            "Emboss\n(alpha=1)",
            [("strength=%.2f" % (strength,), iaa.Emboss(alpha=1, strength=strength))
             for strength in [0, 0.5, 1.0, 1.5, 2.0]]
        ),
        _Descriptor.from_augsubs(
            "convolutional",
            "EdgeDetect",
            [("alpha=%.2f" % (alpha,), iaa.EdgeDetect(alpha=alpha))
             for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]]
        ),
        _Descriptor.from_augsubs(
            "convolutional",
            "DirectedEdgeDetect\n(alpha=1)",
            [("direction=%.2f" % (direction,), iaa.DirectedEdgeDetect(alpha=1, direction=direction))
             for direction in [0.0, 1*(360/5)/360, 2*(360/5)/360, 3*(360/5)/360, 4*(360/5)/360]]
        ),
        _MarkdownTableSeeAlsoUrlList([
            _MarkdownTableSeeAlsoUrl.from_augmenter("convolutional", "Convolve"),
        ])
    ])

    # ###
    # debug
    # ###
    descriptors.extend([
        _MarkdownTableSeeAlsoUrlList([
            _MarkdownTableSeeAlsoUrl.from_augmenter("debug", "SaveDebugImageEveryNBatches"),
        ])
    ])

    # ###
    # edges
    # ###
    descriptors.extend([
        _Descriptor.from_augsubs(
            "edges",
            "Canny",
            [("alpha=%.2f" % (alpha,), iaa.Canny(alpha=alpha))
             for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]]),
    ])

    # ###
    # flip
    # ###
    descriptors.extend([
        _Descriptor.from_augsubs(
            "flip",
            "Fliplr",
            [("p=%.1f" % (p,), iaa.Fliplr(p)) for p in [0, 1]],
            affects_geometry=True),
        _Descriptor.from_augsubs(
            "flip",
            "Flipud",
            [("p=%.1f" % (p,), iaa.Flipud(p)) for p in [0, 1]],
            affects_geometry=True),
        _MarkdownTableSeeAlsoUrlList([
            _MarkdownTableSeeAlsoUrl.from_augmenter("color", "HorizontalFlip"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("color", "VerticalFlip"),
        ])
    ])

    # ###
    # geometric
    # ###
    descriptors.extend([
        _Descriptor.from_augsubs(
            "geometric",
            "Affine",
            [("", iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}, translate_px={"x": (-32, 32), "y": (-32, 32)}, rotate=(-45, 45), shear=(-32, 32), mode=["constant", "edge"], cval=(0.0, 1.0)))
             for _ in sm.xrange(5)],
            affects_geometry=True
        ),
        #_Descriptor.from_augsubs(
        #    "geometric",
        #    "Affine: Scale",
        #    [("%.1fx" % (scale,), iaa.Affine(scale=scale)) for scale in [0.1, 0.5, 1.0, 1.5, 1.9]],
        #    affects_geometry=True),
        #_Descriptor.from_augsubs(
        #    "geometric",
        #    "Affine: Translate",
        #    [("x=%d y=%d" % (x, y), iaa.Affine(translate_px={"x": x, "y": y}))
        #     for x, y in [(-32, -16), (-16, -32), (-16, -8), (16, 8), (16, 32)]],
        #    affects_geometry=True),
        #_Descriptor.from_augsubs(
        #    "geometric",
        #    "Affine: Rotate",
        #    [("%d deg" % (rotate,), iaa.Affine(rotate=rotate)) for rotate in [-90, -45, 0, 45, 90]],
        #    affects_geometry=True),
        #_Descriptor.from_augsubs(
        #    "geometric",
        #    "Affine: Shear",
        #    [("%d deg" % (shear,), iaa.Affine(shear=shear)) for shear in [-45, -25, 0, 25, 45]],
        #    affects_geometry=True),
        _Descriptor.from_augsubs(
            "geometric",
            "Affine: Modes",
            [("mode=%s" % (mode,), iaa.Affine(translate_px=-32, mode=mode)) for mode in ["constant", "edge", "symmetric", "reflect", "wrap"]],
            affects_geometry=True,
            #comment='Augmentation of heatmaps and segmentation maps is currently always done with mode="constant" '
            #        + 'for consistency with keypoint and bounding box augmentation. It may be resonable to use '
            #        + 'mode="constant" for images too when augmenting heatmaps or segmentation maps.'
            ),
        _Descriptor.from_augsubs(
            "geometric",
            "Affine: cval",
            [("cval=%d" % (int(cval*255),), iaa.Affine(translate_px=-32, cval=int(cval*255), mode="constant"))
             for cval in [0.0, 0.25, 0.5, 0.75, 1.0]],
            affects_geometry=True),

        # ScaleX -> see also
        # ScaleY -> see also
        # TranslateX -> see also
        # TranslateY -> see also
        # Rotate -> see also
        # ShearX -> see also
        # ShearY -> see also
        # AffineCV2 -> deprecated

        _Descriptor.from_augsubs(
            "geometric",
            "PiecewiseAffine",
            [("scale=%.3f" % (scale,), iaa.PiecewiseAffine(scale=scale))
             for scale in [0.015, 0.03, 0.045, 0.06, 0.075]],
            affects_geometry=True
        ),
        _Descriptor.from_augsubs(
            "geometric",
            "PerspectiveTransform",
            [("scale=%.3f" % (scale,), iaa.PerspectiveTransform(scale=scale))
             for scale in [0.025, 0.05, 0.075, 0.10, 0.125]],
            affects_geometry=True,
            seed=6
        ),
        _Descriptor.from_augsubs(
            "geometric",
            "ElasticTransformation\n(sigma=1.0)",
            [("alpha=%.1f" % (alpha,), iaa.ElasticTransformation(alpha=alpha, sigma=1.0))
             for alpha in np.linspace(1.0, 20.0, num=5)],
            affects_geometry=True,
            seed=1
        ),
        _Descriptor.from_augsubs(
            "geometric",
            "ElasticTransformation\n(sigma=5.0)",
            [("alpha=%.1f" % (alpha,), iaa.ElasticTransformation(alpha=alpha, sigma=5.0))
             for alpha in np.linspace(1.0, 60.0, num=5)],
            affects_geometry=True
        ),
        _Descriptor.from_augsubs(
            "geometric",
            "Rot90",
            [("k=%d" % (k,), iaa.Rot90(k=k)) for k in [0, 1, 2, 3]],
            affects_geometry=True
        ),
        _Descriptor.from_augsubs(
            "geometric",
            "WithPolarWarping\n+Affine",
            [("",
              iaa.WithPolarWarping(
                  iaa.Affine(
                      rotate=(-15, 15),
                      translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                      scale=(0.9, 1.1),
                      shear=(-5, 5)
                  )
              )) for _ in np.arange(5)],
            affects_geometry=True
        ),
        _Descriptor.from_augsubs(
            "geometric",
            "Jigsaw\n(5x5 grid)",
            [("", iaa.Jigsaw(nb_rows=5, nb_cols=5, max_steps=1)) for _ in np.arange(5)],
            affects_geometry=["heatmaps", "segmentation_maps", "keypoints"]
        ),
        _MarkdownTableSeeAlsoUrlList([
            _MarkdownTableSeeAlsoUrl.from_augmenter("geometric", "ScaleX"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("geometric", "ScaleY"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("geometric", "TranslateX"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("geometric", "TranslateY"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("geometric", "Rotate")
        ])
    ])

    # ###
    # imgcorruptlike
    # ###
    descriptors.extend([
        _Descriptor.from_augsubs(
            "imgcorruptlike",
            "GlassBlur",
            [("severity=%d" % (severity,),
              iaa.imgcorruptlike.GlassBlur(severity=severity))
             for severity in [1, 2, 3, 4, 5]]),
        _Descriptor.from_augsubs(
            "imgcorruptlike",
            "DefocusBlur",
            [("severity=%d" % (severity,),
              iaa.imgcorruptlike.DefocusBlur(severity=severity))
             for severity in [1, 2, 3, 4, 5]]),
        _Descriptor.from_augsubs(
            "imgcorruptlike",
            "ZoomBlur",
            [("severity=%d" % (severity,),
              iaa.imgcorruptlike.ZoomBlur(severity=severity))
             for severity in [1, 2, 3, 4, 5]]),
        _Descriptor.from_augsubs(
            "imgcorruptlike",
            "Snow",
            [("severity=%d" % (severity,),
              iaa.imgcorruptlike.Snow(severity=severity))
             for severity in [1, 2, 3, 4, 5]]),
        _Descriptor.from_augsubs(
            "imgcorruptlike",
            "Spatter",
            [("severity=%d" % (severity,),
              iaa.imgcorruptlike.Spatter(severity=severity))
             for severity in [1, 2, 3, 4, 5]]),
        _MarkdownTableSeeAlsoUrlList([
            _MarkdownTableSeeAlsoUrl.from_augmenter("imgcorruptlike", "GaussianNoise"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("imgcorruptlike", "ShotNoise"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("imgcorruptlike", "ImpulseNoise"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("imgcorruptlike", "SpeckleNoise"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("imgcorruptlike", "GaussianBlur"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("imgcorruptlike", "MotionBlur"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("imgcorruptlike", "Fog"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("imgcorruptlike", "Frost"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("imgcorruptlike", "Contrast"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("imgcorruptlike", "Brightness"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("imgcorruptlike", "Saturate"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("imgcorruptlike", "JpegCompression"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("imgcorruptlike", "Pixelate"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("imgcorruptlike", "ElasticTransform")
        ])
    ])

    # ###
    # pillike
    # ###
    descriptors.extend([
        _Descriptor.from_augsubs(
            "pillike",
            "Autocontrast",
            [("cutoff=%d" % (cutoff,),
              iaa.pillike.Autocontrast(cutoff)) for cutoff in [0, 5, 10, 15, 20]]),
        _Descriptor.from_augsubs(
            "pillike",
            "EnhanceColor",
            [("factor=%.1f" % (factor,),
              iaa.pillike.EnhanceColor(factor)) for factor in np.linspace(0.0, 3.0, 5)]),
        _Descriptor.from_augsubs(
            "pillike",
            "EnhanceSharpness",
            [("factor=%.1f" % (factor,),
              iaa.pillike.EnhanceSharpness(factor)) for factor in np.linspace(0.0, 3.0, 5)]),
        _Descriptor.from_augsubs(
            "pillike",
            "FilterEdgeEnhanceMore",
            [("off", iaa.Identity()),
             ("on", iaa.pillike.FilterEdgeEnhanceMore())]),
        _Descriptor.from_augsubs(
            "pillike",
            "FilterContour",
            [("off", iaa.Identity()),
             ("on", iaa.pillike.FilterContour())]),
        _MarkdownTableSeeAlsoUrlList([
            _MarkdownTableSeeAlsoUrl.from_augmenter("pillike", "Solarize"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("pillike", "Posterize"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("pillike", "Equalize"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("pillike", "EnhanceContrast"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("pillike", "EnhanceBrightness"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("pillike", "FilterBlur"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("pillike", "FilterSmooth"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("pillike", "FilterSmoothMore"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("pillike", "FilterEdgeEnhance"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("pillike", "FilterFindEdges"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("pillike", "FilterEmboss"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("pillike", "FilterSharpen"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("pillike", "FilterDetail"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("pillike", "Affine"),
        ])
    ])

    # ###
    # pooling
    # ###
    descriptors.extend([
        _Descriptor.from_augsubs(
            "pooling",
            "AveragePooling",
            [("kernel_size=%d" % (k,), iaa.AveragePooling(k))
             for k in [1, 2, 4, 8, 16]]),
        _Descriptor.from_augsubs(
            "pooling",
            "MaxPooling",
            [("kernel_size=%d" % (k,), iaa.MaxPooling(k))
             for k in [1, 2, 4, 8, 16]]),
        _Descriptor.from_augsubs(
            "pooling",
            "MinPooling",
            [("kernel_size=%d" % (k,), iaa.MinPooling(k))
             for k in [1, 2, 4, 8, 16]]),
        _Descriptor.from_augsubs(
            "pooling",
            "MedianPooling",
            [("kernel_size=%d" % (k,), iaa.MedianPooling(k))
             for k in [1, 2, 4, 8, 16]]),
    ])

    # ###
    # segmentation
    # ###
    descriptors.extend([
        _Descriptor.from_augsubs(
            "segmentation",
            "Superpixels\n(p_replace=1)",
            [("n_segments=%d" % (n_segments,), iaa.Superpixels(p_replace=1.0, n_segments=n_segments))
             for n_segments in [25, 50, 75, 100, 125]]),
        _Descriptor.from_augsubs(
            "segmentation",
            "Superpixels\n(n_segments=100)",
            [("p_replace=%.2f" % (p_replace,), iaa.Superpixels(p_replace=p_replace, n_segments=100))
             for p_replace in [0, 0.25, 0.5, 0.75, 1.0]]),
        _Descriptor.from_augsubs(
            "segmentation",
            "UniformVoronoi",
            [("n_points=%d" % (n_points,), iaa.UniformVoronoi(n_points))
             for n_points in [50, 100, 200, 400, 800]]),
        _Descriptor.from_augsubs(
            "segmentation",
            "RegularGridVoronoi: rows/cols\n(p_drop_points=0)",
            [("n_rows=n_cols=%d" % (n_rows,),
              iaa.RegularGridVoronoi(n_rows=n_rows, n_cols=n_rows, p_drop_points=0))
             for n_rows in [4, 8, 16, 32, 64]]),
        _Descriptor.from_augsubs(
            "segmentation",
            "RegularGridVoronoi: p_drop_points\n(n_rows=n_cols=30)",
            [("p_drop_points=%.2f" % (p_drop_points,),
              iaa.RegularGridVoronoi(n_rows=30, n_cols=30, p_drop_points=p_drop_points))
             for p_drop_points in [0.0, 0.2, 0.4, 0.6, 0.8]]),
        _Descriptor.from_augsubs(
            "segmentation",
            "RegularGridVoronoi: p_replace\n(n_rows=n_cols=16)",
            [("p_replace=%.2f" % (p_replace,),
              iaa.RegularGridVoronoi(n_rows=16, n_cols=16, p_drop_points=0, p_replace=p_replace))
             for p_replace in [0, 0.25, 0.5, 0.75, 1.0]]),
        _MarkdownTableSeeAlsoUrlList([
            _MarkdownTableSeeAlsoUrl.from_augmenter("segmentation", "Voronoi"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("segmentation", "RelativeRegularGridVoronoi"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("segmentation", "RegularGridPointsSampler"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("segmentation", "RelativeRegularGridPointsSampler"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("segmentation", "DropoutPointsSampler"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("segmentation", "UniformPointsSampler"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("segmentation", "SubsamplingPointsSampler")
        ])
    ])

    # ###
    # size
    # ###
    descriptors.extend([
        _Descriptor.from_augsubs(
            "size",
            "CropAndPad",
            [("px=%s" % (str(vals),), iaa.CropAndPad(px=vals))
             for vals in [(-2, 0, 0, 0), (2, 0, 0, 0), (0, 2, 0, -2), (0, -2, 0, 2), (-2, -1, 0, 1), (2, 1, 0, -1)]],
            affects_geometry=True),
        _Descriptor.from_augsubs(
            "size",
            "Crop",
            [("px=%s" % (str(vals),), iaa.Crop(px=vals))
             for vals in [(2, 0, 0, 0), (0, 8, 8, 0), (4, 0, 16, 4), (8, 0, 0, 32), (32, 64, 0, 0)]],
            affects_geometry=True),
        _Descriptor.from_augsubs(
            "size",
            "Pad",
            [("px=%s" % (str(vals),), iaa.Pad(px=vals))
             for vals in [(2, 0, 0, 0), (0, 8, 8, 0), (4, 0, 16, 4), (8, 0, 0, 32), (32, 64, 0, 0)]],
            affects_geometry=True),
        _Descriptor.from_augsubs(
            "size",
            "PadToFixedSize\n(height'=height+32,\nwidth'=width+32)",
            [("position=%s" % (str(position),), iaa.KeepSizeByResize(iaa.PadToFixedSize(height=h+32, width=w+32, position=position)))
             for position in ["left-top", "center-top", "right-top", "left-center", "center", "right-center",
                              "left-bottom", "center-bottom", "right-bottom",
                              "uniform", "uniform", "uniform", "uniform",
                              "normal", "normal", "normal", "normal"]],
            affects_geometry=True,
            seed=1),
        _Descriptor.from_augsubs(
            "size",
            "CropToFixedSize\n(height'=height-32,\nwidth'=width-32)",
            [("position=%s" % (str(position),), iaa.KeepSizeByResize(iaa.CropToFixedSize(height=h-32, width=w-32, position=position)))
             for position in ["left-top", "center-top", "right-top", "left-center", "center", "right-center",
                              "left-bottom", "center-bottom", "right-bottom",
                              "uniform", "uniform", "uniform", "uniform",
                              "normal", "normal", "normal", "normal"]],
            affects_geometry=True),
        _MarkdownTableSeeAlsoUrlList([
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "Resize"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "CropToMultiplesOf"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "PadToMultiplesOf"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "CropToPowersOf"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "PadToPowersOf"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "CropToAspectRatio"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "PadToAspectRatio"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "CropToSquare"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "PadToSquare"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "CenterCropToFixedSize"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "CenterPadToFixedSize"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "CenterCropToMultiplesOf"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "CenterPadToMultiplesOf"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "CenterCropToPowersOf"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "CenterPadToPowersOf"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "CenterCropToAspectRatio"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "CenterPadToAspectRatio"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "CenterCropToSquare"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "CenterPadToSquare"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("size", "KeepSizeByResize"),
        ])
    ])

    # ###
    # weather
    # ###
    descriptors.extend([
        _Descriptor.from_augsubs(
            "weather",
            "FastSnowyLandscape\n(lightness_multiplier=2.0)",
            [("lightness_threshold=%d" % (lthresh,), iaa.FastSnowyLandscape(lightness_threshold=lthresh, lightness_multiplier=2.0))
             for lthresh in [0, 50, 100, 150, 200]]
        ),
        # CloudLayer -> see also
        _Descriptor.from_augsubs(
            "weather",
            "Clouds",
            [("", iaa.Clouds()) for _ in range(5)]
        ),
        _Descriptor.from_augsubs(
            "weather",
            "Fog",
            [("", iaa.Fog()) for _ in range(5)]
        ),
        # SnowflakesLayer -> see also
        _Descriptor.from_augsubs(
            "weather",
            "Snowflakes",
            [("", iaa.Snowflakes()) for _ in range(5)]
        ),
        # RainLayer -> see also
        _Descriptor.from_augsubs(
            "weather",
            "Rain",
            [("", iaa.Rain()) for _ in range(5)]
        ),
        _MarkdownTableSeeAlsoUrlList([
            _MarkdownTableSeeAlsoUrl.from_augmenter("weather", "CloudLayer"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("weather", "SnowflakesLayer"),
            _MarkdownTableSeeAlsoUrl.from_augmenter("weather", "RainLayer")
        ])
    ])

    def mimwrite_if_changed(fp, frames, duration):
        # we first save and then load the frames that are supposed to be saved here
        # this is done to compare with the already saved frames, because there is a bit of compression involved when
        # saving
        with tempfile.NamedTemporaryFile(mode="wb") as tempf:
            imageio.mimwrite(tempf.name, frames, duration=duration, format="gif")
            frames_to_save = imageio.mimread(tempf.name)

        save = False
        if not os.path.isfile(fp):
            save = True
        else:
            frames_saved = imageio.mimread(fp)
            if len(frames_to_save) != len(frames_saved):
                save = True
            elif any([frame_i.shape[0:2] != frame_j.shape[0:2] for frame_i, frame_j in zip(frames_to_save, frames_saved)]):  # note that loaded frames have 4 channels, even if saved with 3
                save = True
            else:
                for frame_i, frame_j in zip(frames_to_save, frames_saved):
                    diff = np.abs(frame_i.astype(np.float64) - frame_j.astype(np.float64))
                    if np.average(diff) > 1.0:
                        save = True
                        break
        if save:
            imageio.mimwrite(fp, frames, duration=duration)
        else:
            print("[draw_per_augmenter_videos] [mimwrite_if_changed] skipped '%s' (not changed)" % (fp,))

    def _makedirs(fp):
        fp_dir = os.path.dirname(fp)
        if not os.path.exists(fp_dir):
            os.makedirs(fp_dir)

    # routine to generate gifs and produce markup
    DOC_BASE = "https://raw.githubusercontent.com/aleju/imgaug-doc/master/"
    table = _MarkdownTable()

    for descriptor in descriptors:
        if isinstance(descriptor, _MarkdownTableSeeAlsoUrlList):
            table.append(descriptor)
            continue

        print(descriptor.title)

        image_to_use = image
        if descriptor.module in ["weather"]:
            image_to_use = image_landscape
        elif descriptor.module in ["artistic"]:
            image_to_use = image_valley
        elif "SomeColors" in descriptor.title:
            image_to_use = image_vangogh

        frames_images, frames_kps, frames_bbs, frames_hm, frames_segmap = \
            descriptor.generate_frames(
                image_to_use,
                keypoints, bbs, polygons, heatmap, segmap, h_subtitle)

        if descriptor.affects_geometry:
            frames_images = [np.hstack([frame_image, frame_hm, frame_segmap])
                             for frame_image, frame_hm, frame_segmap in zip(frames_images, frames_hm, frames_segmap)]

        aug_name = slugify(descriptor.title)
        fp_all = os.path.join(IMAGES_DIR, "augmenter_videos/%s/%s.gif" % (descriptor.module, aug_name,))

        _makedirs(fp_all)

        #fp_images = os.path.join(IMAGES_DIR, "augmenter_videos/augment_images_with_coordsaug/%s.gif" % (aug_name,))
        #fp_kps = os.path.join(IMAGES_DIR, "augmenter_videos/augment_keypoints/%s.gif" % (aug_name,))
        #fp_bbs = os.path.join(IMAGES_DIR, "augmenter_videos/augment_bounding_boxes/%s.gif" % (aug_name,))
        #fp_kps_bbs = os.path.join(IMAGES_DIR, "augmenter_videos/augment_coordinate_based/%s.gif" % (aug_name,))
        #fp_hm = os.path.join(IMAGES_DIR, "augmenter_videos/augment_heatmaps/%s.gif" % (aug_name,))
        #fp_segmap = os.path.join(IMAGES_DIR, "augmenter_videos/augment_segmentation_maps/%s.gif" % (aug_name,))

        #_makedirs(fp_images)
        #_makedirs(fp_hm)
        #_makedirs(fp_segmap)

        mimwrite_if_changed(fp_all, frames_images, duration=1.25)
        #if descriptor.affects_geometry:
        #    markup_images = '![%s (+Keypoints, +BBs)](%s%s?raw=true "%s (+Keypoints, +BBs")' % (descriptor.title_markup, DOC_BASE, fp_images, descriptor.title_markup)
        #else:
        #    markup_images = '![%s](%s%s?raw=true "%s")' % (descriptor.title_markup, DOC_BASE, fp_images, descriptor.title_markup)
        #markup_images = '![%s](%s%s?raw=true "%s")' % (descriptor.title_markup, DOC_BASE, fp_images, descriptor.title_markup)
        height = frames_images[0].shape[0]
        width = frames_images[0].shape[1]
        markup_images = '<img src="%s%s" height="%d" width="%d" alt="%s">' % (DOC_BASE, fp_all, height, width, descriptor.title_markup)

        #markup_kps_bbs = ""
        #markup_kps = ""
        #markup_bbs = ""
        markup_hm = ""
        markup_segmap = ""
        """
        if descriptor.affects_geometry > 0:
            #imageio.mimsave(fp_kps, frames_kps, duration=1.0)
            #imageio.mimsave(fp_bbs, frames_bbs, duration=1.0)
            #imageio.mimsave(fp_kps_bbs, frames_kps, duration=1.0)
            mimwrite_if_changed(fp_hm, frames_hm, duration=1.25)
            mimwrite_if_changed(fp_segmap, frames_segmap, duration=1.25)
            #markup_kps = '![%s (keypoint augmentation)](%s?raw=true "%s (keypoint augmentation)")' % (descriptor.title_markup, fp_kps, descriptor.title_markup)
            #markup_bbs = '![%s (bounding box augmentation)](%s?raw=true "%s (bounding box augmentation)")' % (descriptor.title_markup, fp_bbs, descriptor.title_markup)
            #markup_kps_bbs = '![%s (keypoint and BB augmentation)](%s?raw=true "%s (keypoint and BB augmentation)")' % (descriptor.title_markup, fp_kps_bbs, descriptor.title_markup)
            markup_hm = '![%s (heatmap augmentation)](%s%s?raw=true "%s (heatmap augmentation)")' % (descriptor.title_markup, DOC_BASE, fp_hm, descriptor.title_markup)
            markup_segmap = '![%s (segmentation map augmentation)](%s%s?raw=true "%s (segmentation map augmentation)")' % (descriptor.title_markup, DOC_BASE, fp_segmap, descriptor.title_markup)
        """

        #table.append(descriptor, markup_images, markup_kps, markup_bbs, markup_hm, markup_segmap)
        #table.append(descriptor, markup_images, markup_kps_bbs, "", markup_hm, markup_segmap)
        cell = _MarkdownTableCell(
            descriptor, markup_images, "", "", markup_hm, markup_segmap)
        table.append(cell)

    fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), "readme_example_images_code.txt")
    with open(fp, "w") as f:
        f.write(table.render())
        print("Wrote table code to: %s" % (fp,))


"""
class ExampleFrameCell(object):
    def __init__(self, image_height, image_width, subtitle_height):
        self.image_height = image_height
        self.image_width = image_width
        self.cell_height = image_height + subtitle_height
        self.cell_width = image_width

    def draw(self, image, subtitle):
        image_cell = np.zeros((self.cell_height, self.cell_width, 3), dtype=np.uint8) + 255
        image_cell[0:image.shape[0], 0:image.shape[1], :] = image
        image_cell = ia.draw_text(image_cell, x=2, y=image.shape[0]+2, text=subtitle, color=[0, 0, 0], size=9)

        return image_cell
"""

#
# TODO this part is largely copied from generate_documentation_images, make DRY
#


def compress_to_jpg(image, quality=75):
    quality = quality if quality is not None else 75
    im = PIL.Image.fromarray(image)
    out = BytesIO()
    im.save(out, format="JPEG", quality=quality)
    jpg_string = out.getvalue()
    out.close()
    return jpg_string


def decompress_jpg(image_compressed):
    img_compressed_buffer = BytesIO()
    img_compressed_buffer.write(image_compressed)
    img = imageio.imread(img_compressed_buffer, pilmode="RGB")
    img_compressed_buffer.close()
    return img


def arrdiff(arr1, arr2):
    nb_cells = np.prod(arr2.shape)
    d_avg = np.sum(np.power(np.abs(arr1.astype(np.float64) - arr2.astype(np.float64)), 2)) / nb_cells
    return d_avg


def save(fp, image, quality=75):
    image_jpg = compress_to_jpg(image, quality=quality)
    image_jpg_decompressed = decompress_jpg(image_jpg)

    # If the image file already exists and is (practically) identical,
    # then don't save it again to avoid polluting the repository with tons
    # of image updates.
    # Not that we have to compare here the results AFTER jpg compression
    # and then decompression. Otherwise we compare two images of which
    # image (1) has never been compressed while image (2) was compressed and
    # then decompressed.
    if os.path.isfile(fp):
        image_saved = imageio.imread(fp, pilmode="RGB")
        #print("arrdiff", arrdiff(image_jpg_decompressed, image_saved))
        same_shape = (image_jpg_decompressed.shape == image_saved.shape)
        d_avg = arrdiff(image_jpg_decompressed, image_saved) if same_shape else -1
        if same_shape and d_avg <= 1.0:
            print("[INFO] Did not save image '%s', because the already saved image is basically identical (d_avg=%.8f)" % (fp, d_avg,))
            return
        else:
            print("[INFO] Saving image '%s'..." % (fp,))

    with open(fp, "w") as f:
        f.write(image_jpg)


if __name__ == "__main__":
    main()
