from __future__ import print_function, division

import time
import pickle
import os
import gc
import argparse
import re

import numpy as np
import six.moves as sm
import skimage.data

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_augmenters", type=str,
                        help="Names of augmenters to measure, regexes, delimiter is ','.")
    parser.add_argument("--nosave", action="store_true", help="Whether not to save any results")
    args = parser.parse_args()
    if args.only_augmenters is not None:
        args.only_augmenters = [name.strip() for name in args.only_augmenters.split(",")]
    args.save = (args.nosave is not True)

    if not args.save:
        print("[NOTE] will not save data")

    iterations = 100
    batch_sizes = [1, 128]
    backgrounds = [False]

    print("---------------------------")
    print("Images")
    print("---------------------------")
    results_images = []
    base_image = skimage.data.astronaut()
    images = [ia.imresize_single_image(base_image, (64, 64)),
              ia.imresize_single_image(base_image, (224, 224))]

    for image in images:
        print("")
        print("image size: %s" % (image.shape,))
        augmenters = create_augmenters(height=image.shape[0], width=image.shape[1],
                                       height_augmentable=image.shape[0], width_augmentable=image.shape[1],
                                       only_augmenters=args.only_augmenters)
        for batch_size in batch_sizes:
            if batch_size != batch_sizes[0]:
                print("")
            print("batch_size: %d" % (batch_size,))

            for background in backgrounds:
                for augmenter in augmenters:
                    images_batch = np.uint8([image] * batch_size)

                    ia.seed(1)
                    times = []
                    gc.disable()  # as done in timeit
                    if not background:
                        for _ in sm.xrange(iterations):
                            time_start = time.time()
                            _img_aug = augmenter.augment_images(images_batch)
                            time_end = time.time()
                            times.append(time_end - time_start)
                    else:
                        batches = [ia.Batch(images=images_batch) for _ in sm.xrange(iterations)]
                        for _ in sm.xrange(iterations):
                            time_start = time.time()
                            gen = augmenter.augment_batches(batches, background=True)
                            for _batch_aug in gen:
                                pass
                            time_end = time.time()
                            times.append(time_end - time_start)
                    gc.enable()

                    results_images.append({
                        "augmentable": "images",
                        "background": background,
                        "image.shape": image.shape,
                        "batch_size": batch_size,
                        "augmenter.name": augmenter.name,
                        "times": times
                    })

                    items_per_sec = (1/np.average(times)) * batch_size
                    mbit_per_img = (image.size * image.dtype.itemsize * 8) / 1024 / 1024
                    mbit_per_sec = items_per_sec * mbit_per_img
                    print("IMG | HxW=%s B=%d %s "
                          "| SUM %10.5fs "
                          "| ITER avg %10.5fs, min %10.5fs, max %10.5fs "
                          "| img/s %11.3f "
                          "| mbit/s %9.3f, mbyte/s %9.3f "
                          "| %s" % (
                              image.shape[0:2], batch_size, "BG" if background else "FG",
                              float(np.sum(times)), np.average(times), np.min(times), np.max(times),
                              items_per_sec,
                              mbit_per_sec, mbit_per_sec / 8,
                              augmenter.name))

    if args.save:
        current_dir = os.path.dirname(__file__)
        target_dir = os.path.join(current_dir, "measure_performance_results")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with open(os.path.join(target_dir, "results_images.pickle"), "wb") as f:
            pickle.dump(results_images, f, protocol=-1)

    print("---------------------------")
    print("Heatmaps")
    print("---------------------------")
    results_heatmaps = []
    for nb_heatmaps in [1, 5]:  # per image
        base_image = skimage.data.astronaut()
        images = [ia.imresize_single_image(base_image, (64, 64)),
                  ia.imresize_single_image(base_image, (224, 224))]
        heatmaps = [np.tile(heatmap[..., 0:1], (1, 1, nb_heatmaps))
                    for heatmap in iaa.Grayscale(1.0).augment_images(images)]
        heatmaps_ois = [ia.HeatmapsOnImage(heatmap.astype(np.float32)/255.0, shape=(224, 224, 3))
                        for heatmap in heatmaps]

        for heatmaps_oi in heatmaps_ois:
            print("")
            print("heatmap size: %s (on image: %s)" % (heatmaps_oi.arr_0to1.shape, heatmaps_oi.shape,))
            augmenters = create_augmenters(height=heatmaps_oi.shape[0], width=heatmaps_oi.shape[1],
                                           height_augmentable=heatmaps_oi.arr_0to1.shape[0],
                                           width_augmentable=heatmaps_oi.arr_0to1.shape[1],
                                           only_augmenters=args.only_augmenters)
            for batch_size in batch_sizes:
                if batch_size != batch_sizes[0]:
                    print("")
                print("batch_size: %d" % (batch_size,))
                for background in backgrounds:
                    for augmenter in augmenters:
                        heatmaps_oi_batch = [heatmaps_oi] * batch_size

                        ia.seed(1)
                        times = []
                        gc.disable()  # as done in timeit
                        if not background:
                            for _ in sm.xrange(iterations):
                                time_start = time.time()
                                _hms_aug = augmenter.augment_heatmaps(heatmaps_oi_batch)
                                time_end = time.time()
                                times.append(time_end - time_start)
                                gc.collect()
                        else:
                            batches = [ia.Batch(heatmaps=heatmaps_oi_batch) for _ in sm.xrange(iterations)]
                            for _ in sm.xrange(iterations):
                                time_start = time.time()
                                gen = augmenter.augment_batches(batches, background=True)
                                for _batch_aug in gen:
                                    pass
                                time_end = time.time()
                                times.append(time_end - time_start)
                                gc.collect()
                        gc.disable()

                        results_heatmaps.append({
                            "augmentable": "heatmaps",
                            "background": background,
                            "nb_heatmaps": nb_heatmaps,
                            "heatmaps_oi.arr_0to1.shape": heatmaps_oi.arr_0to1.shape,
                            "heatmaps_oi.shape": heatmaps_oi.shape,
                            "batch_size": batch_size,
                            "augmenter.name": augmenter.name,
                            "times": times
                        })

                        h, w, c = heatmaps_oi.arr_0to1.shape
                        items_per_sec = (1/np.average(times)) * batch_size * c
                        mbit_per_img = (h * w * heatmaps_oi.arr_0to1.dtype.itemsize * 8) / 1024 / 1024
                        mbit_per_sec = items_per_sec * mbit_per_img
                        print("HMs | HxWxN=%s (on %s) B=%d %s "
                              "| SUM %10.5fs "
                              "| ITER avg %10.5fs, min %10.5fs, max %10.5fs "
                              "| hms/s %11.3f "
                              "| mbit/s %9.3f, mbyte/s %9.3f "
                              "| %s" % (
                                  heatmaps_oi.arr_0to1.shape[0:3], heatmaps_oi.shape[0:2], batch_size,
                                  "BG" if background else "FG",
                                  float(np.sum(times)), np.average(times), np.min(times), np.max(times),
                                  items_per_sec,
                                  mbit_per_sec, mbit_per_sec / 8,
                                  augmenter.name))

    if args.save:
        current_dir = os.path.dirname(__file__)
        target_dir = os.path.join(current_dir, "measure_performance_results")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with open(os.path.join(target_dir, "results_heatmaps.pickle"), "wb") as f:
            pickle.dump(results_heatmaps, f, protocol=-1)

    print("---------------------------")
    print("Keypoints")
    print("---------------------------")
    results_keypoints = []
    for nb_points in [1, 10]:  # per image
        base_image = skimage.data.astronaut()
        h, w = base_image.shape[0:2]
        if nb_points == 1:
            keypoints = [ia.Keypoint(x=x*w, y=y*h)
                         for y, x in [(0.4, 0.4)]]
        else:
            keypoints = [ia.Keypoint(x=x*w, y=y*h)
                         for y, x in [(0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.6, 0.6), (0.7, 0.7), (0.8, 0.8),
                                      (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5)]]
        base_image_kpoi = ia.KeypointsOnImage(keypoints, shape=(224, 224, 3))
        images = [ia.imresize_single_image(base_image, (64, 64)),
                  ia.imresize_single_image(base_image, (224, 224))]
        keypoints_on_images = [base_image_kpoi.on(image.shape) for image in images]

        for keypoints_on_image in keypoints_on_images:
            print("")
            print("#points: %d (on image: %s)" % (len(keypoints_on_image.keypoints), keypoints_on_image.shape,))
            augmenters = create_augmenters(height=keypoints_on_image.shape[0], width=keypoints_on_image.shape[1],
                                           height_augmentable=keypoints_on_image.shape[0],
                                           width_augmentable=keypoints_on_image.shape[1],
                                           only_augmenters=args.only_augmenters)
            for batch_size in batch_sizes:
                if batch_size != batch_sizes[0]:
                    print("")
                print("batch_size: %d" % (batch_size,))
                for background in backgrounds:
                    for augmenter in augmenters:
                        keypoints_on_image_batch = [keypoints_on_image] * batch_size

                        ia.seed(1)
                        times = []
                        gc.disable()  # as done in timeit
                        if not background:
                            for _ in sm.xrange(iterations):
                                time_start = time.time()
                                _kps_aug = augmenter.augment_keypoints(keypoints_on_image_batch)
                                time_end = time.time()
                                times.append(time_end - time_start)
                                gc.collect()
                        else:
                            batches = [ia.Batch(keypoints=keypoints_on_image_batch) for _ in sm.xrange(iterations)]
                            for _ in sm.xrange(iterations):
                                time_start = time.time()
                                gen = augmenter.augment_batches(batches, background=True)
                                for _batch_aug in gen:
                                    pass
                                time_end = time.time()
                                times.append(time_end - time_start)
                        gc.enable()

                        results_keypoints.append({
                            "augmentable": "keypoints",
                            "background": background,
                            "nb_points": len(keypoints_on_image.keypoints),
                            "keypoints_on_image.shape": keypoints_on_image.shape,
                            "batch_size": batch_size,
                            "augmenter.name": augmenter.name,
                            "times": times
                        })

                        items_per_sec = (1/np.average(times)) * batch_size * len(keypoints_on_image.keypoints)
                        mbit_per_img = (len(keypoints_on_image.keypoints) * 2 * 32) / 1024 / 1024
                        mbit_per_sec = items_per_sec * mbit_per_img
                        print("KPs | #points=%d (on %s) B=%d %s "
                              "| SUM %10.5fs "
                              "| ITER avg %10.5fs, min %10.5fs, max %10.5fs "
                              "| kps/s %11.3f "
                              "| mbit/s %9.3f, mbyte/s %9.3f "
                              "| %s" % (
                                  len(keypoints_on_image.keypoints), keypoints_on_image.shape[0:2], batch_size,
                                  "BG" if background else "FG",
                                  float(np.sum(times)), np.average(times), np.min(times), np.max(times),
                                  items_per_sec,
                                  mbit_per_sec, mbit_per_sec / 8,
                                  augmenter.name))

    if args.save:
        current_dir = os.path.dirname(__file__)
        target_dir = os.path.join(current_dir, "measure_performance_results")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with open(os.path.join(target_dir, "results_keypoints.pickle"), "wb") as f:
            pickle.dump(results_keypoints, f, protocol=-1)


def create_augmenters(height, width, height_augmentable, width_augmentable, only_augmenters):
    def lambda_func_images(images, random_state, parents, hooks):
        return images

    def lambda_func_heatmaps(heatmaps, random_state, parents, hooks):
        return heatmaps

    def lambda_func_keypoints(keypoints, random_state, parents, hooks):
        return keypoints

    def assertlambda_func_images(images, random_state, parents, hooks):
        return True

    def assertlambda_func_heatmaps(heatmaps, random_state, parents, hooks):
        return True

    def assertlambda_func_keypoints(keypoints, random_state, parents, hooks):
        return True

    augmenters_meta = [
        iaa.Sequential([iaa.Noop(), iaa.Noop()], random_order=False, name="Sequential_2xNoop"),
        iaa.Sequential([iaa.Noop(), iaa.Noop()], random_order=True, name="Sequential_2xNoop_random_order"),
        iaa.SomeOf((1, 3), [iaa.Noop(), iaa.Noop(), iaa.Noop()], random_order=False, name="SomeOf_3xNoop"),
        iaa.SomeOf((1, 3), [iaa.Noop(), iaa.Noop(), iaa.Noop()], random_order=True, name="SomeOf_3xNoop_random_order"),
        iaa.OneOf([iaa.Noop(), iaa.Noop(), iaa.Noop()], name="OneOf_3xNoop"),
        iaa.Sometimes(0.5, iaa.Noop(), name="Sometimes_Noop"),
        iaa.WithChannels([1, 2], iaa.Noop(), name="WithChannels_1_and_2_Noop"),
        iaa.Noop(name="Noop"),
        iaa.Lambda(func_images=lambda_func_images, func_heatmaps=lambda_func_heatmaps, func_keypoints=lambda_func_keypoints,
                   name="Lambda"),
        iaa.AssertLambda(func_images=assertlambda_func_images, func_heatmaps=assertlambda_func_heatmaps,
                         func_keypoints=assertlambda_func_keypoints, name="AssertLambda"),
        iaa.AssertShape((None, height_augmentable, width_augmentable, None), name="AssertShape"),
        iaa.ChannelShuffle(0.5, name="ChannelShuffle")
    ]
    augmenters_arithmetic = [
        iaa.Add((-10, 10), name="Add"),
        iaa.AddElementwise((-10, 10), name="AddElementwise"),
        #iaa.AddElementwise((-500, 500), name="AddElementwise"),
        iaa.AdditiveGaussianNoise(scale=(5, 10), name="AdditiveGaussianNoise"),
        iaa.AdditiveLaplaceNoise(scale=(5, 10), name="AdditiveLaplaceNoise"),
        iaa.AdditivePoissonNoise(lam=(1, 5), name="AdditivePoissonNoise"),
        iaa.Multiply((0.5, 1.5), name="Multiply"),
        iaa.MultiplyElementwise((0.5, 1.5), name="MultiplyElementwise"),
        iaa.Dropout((0.01, 0.05), name="Dropout"),
        iaa.CoarseDropout((0.01, 0.05), size_percent=(0.01, 0.1), name="CoarseDropout"),
        iaa.ReplaceElementwise((0.01, 0.05), (0, 255), name="ReplaceElementwise"),
        #iaa.ReplaceElementwise((0.95, 0.99), (0, 255), name="ReplaceElementwise"),
        iaa.ImpulseNoise((0.01, 0.05), name="ImpulseNoise"),
        iaa.SaltAndPepper((0.01, 0.05), name="SaltAndPepper"),
        iaa.CoarseSaltAndPepper((0.01, 0.05), size_percent=(0.01, 0.1), name="CoarseSaltAndPepper"),
        iaa.Salt((0.01, 0.05), name="Salt"),
        iaa.CoarseSalt((0.01, 0.05), size_percent=(0.01, 0.1), name="CoarseSalt"),
        iaa.Pepper((0.01, 0.05), name="Pepper"),
        iaa.CoarsePepper((0.01, 0.05), size_percent=(0.01, 0.1), name="CoarsePepper"),
        iaa.Invert(0.1, name="Invert"),
        # ContrastNormalization
        iaa.JpegCompression((50, 99), name="JpegCompression")
    ]
    augmenters_blur = [
        iaa.GaussianBlur(sigma=(1.0, 5.0), name="GaussianBlur"),
        iaa.AverageBlur(k=(3, 11), name="AverageBlur"),
        iaa.MedianBlur(k=(3, 11), name="MedianBlur"),
        iaa.BilateralBlur(d=(3, 11), name="BilateralBlur"),
        iaa.MotionBlur(k=(3, 11), name="MotionBlur")
    ]
    augmenters_color = [
        # InColorspace (deprecated)
        iaa.WithColorspace(to_colorspace="HSV", children=iaa.Noop(), name="WithColorspace"),
        iaa.AddToHueAndSaturation((-10, 10), name="AddToHueAndSaturation"),
        iaa.ChangeColorspace(to_colorspace="HSV", name="ChangeColorspace"),
        iaa.Grayscale((0.01, 0.99), name="Grayscale")
    ]
    augmenters_contrast = [
        iaa.GammaContrast(gamma=(0.5, 2.0), name="GammaContrast"),
        iaa.SigmoidContrast(gain=(5, 20), cutoff=(0.25, 0.75), name="SigmoidContrast"),
        iaa.LogContrast(gain=(0.7, 1.0), name="LogContrast"),
        iaa.LinearContrast((0.5, 1.5), name="LinearContrast"),
        iaa.AllChannelsHistogramEqualization(name="AllChannelsHistogramEqualization"),
        iaa.HistogramEqualization(to_colorspace="HSV", name="HistogramEqualization"),
        iaa.AllChannelsCLAHE(clip_limit=(2, 10), tile_grid_size_px=(3, 11), name="AllChannelsCLAHE"),
        iaa.CLAHE(clip_limit=(2, 10), tile_grid_size_px=(3, 11), to_colorspace="HSV", name="CLAHE"),
    ]
    augmenters_convolutional = [
        iaa.Convolve(np.float32([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), name="Convolve_3x3"),
        iaa.Sharpen(alpha=(0.01, 0.99), lightness=(0.5, 2), name="Sharpen"),
        iaa.Emboss(alpha=(0.01, 0.99), strength=(0, 2), name="Emboss"),
        iaa.EdgeDetect(alpha=(0.01, 0.99), name="EdgeDetect"),
        iaa.DirectedEdgeDetect(alpha=(0.01, 0.99), name="DirectedEdgeDetect")
    ]
    augmenters_flip = [
        iaa.Fliplr(1.0, name="Fliplr"),
        iaa.Flipud(1.0, name="Flipud")
    ]
    augmenters_geometric = [
        iaa.Affine(scale=(0.9, 1.1), translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, rotate=(-10, 10),
                   shear=(-10, 10), order=0, mode="constant", cval=(0, 255), name="Affine_order_0_constant"),
        iaa.Affine(scale=(0.9, 1.1), translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, rotate=(-10, 10),
                   shear=(-10, 10), order=1, mode="constant", cval=(0, 255), name="Affine_order_1_constant"),
        iaa.Affine(scale=(0.9, 1.1), translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, rotate=(-10, 10),
                   shear=(-10, 10), order=3, mode="constant", cval=(0, 255), name="Affine_order_3_constant"),
        iaa.Affine(scale=(0.9, 1.1), translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, rotate=(-10, 10),
                   shear=(-10, 10), order=1, mode="edge", cval=(0, 255), name="Affine_order_1_edge"),
        iaa.Affine(scale=(0.9, 1.1), translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, rotate=(-10, 10),
                   shear=(-10, 10), order=1, mode="constant", cval=(0, 255), backend="skimage",
                   name="Affine_order_1_constant_skimage"),
        # TODO AffineCv2
        iaa.PiecewiseAffine(scale=(0.01, 0.05), nb_rows=4, nb_cols=4, order=1, mode="constant",
                            name="PiecewiseAffine_4x4_order_1_constant"),
        iaa.PiecewiseAffine(scale=(0.01, 0.05), nb_rows=4, nb_cols=4, order=0, mode="constant",
                            name="PiecewiseAffine_4x4_order_0_constant"),
        iaa.PiecewiseAffine(scale=(0.01, 0.05), nb_rows=4, nb_cols=4, order=1, mode="edge",
                            name="PiecewiseAffine_4x4_order_1_edge"),
        iaa.PiecewiseAffine(scale=(0.01, 0.05), nb_rows=8, nb_cols=8, order=1, mode="constant",
                            name="PiecewiseAffine_8x8_order_1_constant"),
        iaa.PerspectiveTransform(scale=(0.01, 0.05), keep_size=False, name="PerspectiveTransform"),
        iaa.PerspectiveTransform(scale=(0.01, 0.05), keep_size=True, name="PerspectiveTransform_keep_size"),
        iaa.ElasticTransformation(alpha=(1, 10), sigma=(0.5, 1.5), order=0, mode="constant", cval=0,
                                  name="ElasticTransformation_order_0_constant"),
        iaa.ElasticTransformation(alpha=(1, 10), sigma=(0.5, 1.5), order=1, mode="constant", cval=0,
                                  name="ElasticTransformation_order_1_constant"),
        iaa.ElasticTransformation(alpha=(1, 10), sigma=(0.5, 1.5), order=1, mode="nearest", cval=0,
                                  name="ElasticTransformation_order_1_nearest"),
        iaa.ElasticTransformation(alpha=(1, 10), sigma=(0.5, 1.5), order=1, mode="reflect", cval=0,
                                  name="ElasticTransformation_order_1_reflect"),
        iaa.Rot90((1, 3), keep_size=False, name="Rot90"),
        iaa.Rot90((1, 3), keep_size=True, name="Rot90_keep_size")
    ]
    augmenters_segmentation = [
        iaa.Superpixels(p_replace=(0.05, 1.0), n_segments=(10, 100), max_size=64, interpolation="cubic",
                        name="Superpixels_max_size_64_cubic"),
        iaa.Superpixels(p_replace=(0.05, 1.0), n_segments=(10, 100), max_size=64, interpolation="linear",
                        name="Superpixels_max_size_64_linear"),
        iaa.Superpixels(p_replace=(0.05, 1.0), n_segments=(10, 100), max_size=128, interpolation="linear",
                        name="Superpixels_max_size_128_linear"),
        iaa.Superpixels(p_replace=(0.05, 1.0), n_segments=(10, 100), max_size=224, interpolation="linear",
                        name="Superpixels_max_size_224_linear"),
    ]
    augmenters_size = [
        iaa.Scale((0.8, 1.2), interpolation="nearest", name="Scale_nearest"),
        iaa.Scale((0.8, 1.2), interpolation="linear", name="Scale_linear"),
        iaa.Scale((0.8, 1.2), interpolation="cubic", name="Scale_cubic"),
        iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="constant", pad_cval=(0, 255), keep_size=False,
                       name="CropAndPad"),
        iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge", pad_cval=(0, 255), keep_size=False,
                       name="CropAndPad_edge"),
        iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="constant", pad_cval=(0, 255), name="CropAndPad_keep_size"),
        iaa.Crop(percent=(0.05, 0.2), keep_size=False, name="Crop"),
        iaa.Crop(percent=(0.05, 0.2), name="Crop_keep_size"),
        iaa.Pad(percent=(0.05, 0.2), pad_mode="constant", pad_cval=(0, 255), keep_size=False, name="Pad"),
        iaa.Pad(percent=(0.05, 0.2), pad_mode="edge", pad_cval=(0, 255), keep_size=False, name="Pad_edge"),
        iaa.Pad(percent=(0.05, 0.2), pad_mode="constant", pad_cval=(0, 255), name="Pad_keep_size"),
        iaa.PadToFixedSize(width=width+10, height=height+10, pad_mode="constant", pad_cval=(0, 255),
                           name="PadToFixedSize"),
        iaa.CropToFixedSize(width=width-10, height=height-10, name="CropToFixedSize"),
        iaa.KeepSizeByResize(iaa.CropToFixedSize(height=height-10, width=width-10), interpolation="nearest",
                             name="KeepSizeByResize_CropToFixedSize_nearest"),
        iaa.KeepSizeByResize(iaa.CropToFixedSize(height=height-10, width=width-10), interpolation="linear",
                             name="KeepSizeByResize_CropToFixedSize_linear"),
        iaa.KeepSizeByResize(iaa.CropToFixedSize(height=height-10, width=width-10), interpolation="cubic",
                             name="KeepSizeByResize_CropToFixedSize_cubic"),
    ]
    augmenters_weather = [
        iaa.FastSnowyLandscape(lightness_threshold=(100, 255), lightness_multiplier=(1.0, 4.0),
                               name="FastSnowyLandscape"),
        iaa.Clouds(name="Clouds"),
        iaa.Fog(name="Fog"),
        iaa.CloudLayer(intensity_mean=(196, 255), intensity_freq_exponent=(-2.5, -2.0), intensity_coarse_scale=10,
                       alpha_min=0, alpha_multiplier=(0.25, 0.75), alpha_size_px_max=(2, 8),
                       alpha_freq_exponent=(-2.5, -2.0), sparsity=(0.8, 1.0), density_multiplier=(0.5, 1.0),
                       name="CloudLayer"),
        iaa.Snowflakes(name="Snowflakes"),
        iaa.SnowflakesLayer(density=(0.005, 0.075), density_uniformity=(0.3, 0.9),
                            flake_size=(0.2, 0.7), flake_size_uniformity=(0.4, 0.8),
                            angle=(-30, 30), speed=(0.007, 0.03),
                            blur_sigma_fraction=(0.0001, 0.001), name="SnowflakesLayer")
    ]

    augmenters = augmenters_meta + augmenters_arithmetic + augmenters_blur + augmenters_color + augmenters_contrast \
        + augmenters_convolutional + augmenters_flip + augmenters_geometric + augmenters_segmentation \
        + augmenters_size + augmenters_weather

    if only_augmenters is not None:
        augmenters_reduced = []
        for augmenter in augmenters:
            if any([re.search(pattern, augmenter.name) for pattern in only_augmenters]):
                augmenters_reduced.append(augmenter)
        augmenters = augmenters_reduced

    return augmenters


if __name__ == "__main__":
    main()
