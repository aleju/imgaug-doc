import os
import pickle
from collections import OrderedDict, defaultdict
import numpy as np
import dashtable

EXP_NAME_MAPPING = OrderedDict([
    ("Sequential_2xNoop", "Sequential (2xNoop)"),
    ("Sequential_2xNoop_random_order", "Sequential (2xNoop, random_order)"),
    ("SomeOf_3xNoop", "SomeOf (1-3, 3xNoop)"),
    ("SomeOf_3xNoop_random_order", "SomeOf (1-3, 3xNoop, random_order)"),
    ("OneOf_3xNoop", "OneOf (3xNoop)"),
    ("Sometimes_Noop", "Sometimes (Noop)"),
    ("WithChannels_1_and_2_Noop", "WithChannels ([1,2], Noop)"),
    ("Noop", "Noop"),
    ("Lambda", "Lambda (return input)"),
    ("AssertLambda", "AssertLambda (return True)"),
    ("AssertShape", "AssertShape (None, H, W, None)"),
    ("ChannelShuffle", "ChannelShuffle (0.5)"),
    ("Add", "Add"),
    ("AddElementwise", "AddElementwise"),
    ("AdditiveGaussianNoise", "AdditiveGaussianNoise"),
    ("AdditiveLaplaceNoise", "AdditiveLaplaceNoise"),
    ("AdditivePoissonNoise", "AdditivePoissonNoise"),
    ("Multiply", "Multiply"),
    ("MultiplyElementwise", "MultiplyElementwise"),
    ("Dropout", "Dropout (1-5%)"),
    ("CoarseDropout", "CoarseDropout (1-5%, size=1-10%)"),
    ("ReplaceElementwise", "ReplaceElementwise"),
    ("ImpulseNoise", "ImpulseNoise"),
    ("SaltAndPepper", "SaltAndPepper"),
    ("CoarseSaltAndPepper", "CoarseSaltAndPepper"),
    ("Salt", "Salt"),
    ("CoarseSalt", "CoarseSalt"),
    ("Pepper", "Pepper"),
    ("CoarsePepper", "CoarsePepper"),
    ("Invert", "Invert (10%)"),
    ("JpegCompression", "JpegCompression (50-99%)"),
    ("GaussianBlur", "GaussianBlur (sigma=(1,5))"),
    ("AverageBlur", "AverageBlur"),
    ("MedianBlur", "MedianBlur"),
    ("BilateralBlur", "BilateralBlur"),
    ("MotionBlur", "MotionBlur"),
    ("WithColorspace", "WithColorspace (HSV, Noop)"),
    ("AddToHueAndSaturation", "AddToHueAndSaturation"),
    ("ChangeColorspace", "ChangeColorspace (HSV)"),
    ("Grayscale", "Grayscale"),
    ("GammaContrast", "GammaContrast"),
    ("SigmoidContrast", "SigmoidContrast"),
    ("LogContrast", "LogContrast"),
    ("LinearContrast", "LinearContrast"),
    ("AllChannelsHistogramEqualization", "AllChannelsHistogramEqualization"),
    ("HistogramEqualization", "HistogramEqualization"),
    ("AllChannelsCLAHE", "AllChannelsCLAHE"),
    ("CLAHE", "CLAHE"),
    ("Convolve_3x3", "Convolve (3x3)"),
    ("Sharpen", "Sharpen"),
    ("Emboss", "Emboss"),
    ("EdgeDetect", "EdgeDetect"),
    ("DirectedEdgeDetect", "DirectedEdgeDetect"),
    ("Fliplr", "Fliplr (p=100%)"),
    ("Flipud", "Flipud (p=100%)"),
    ("Affine_order_0_constant", "Affine (order=0, constant)"),
    ("Affine_order_1_constant", "Affine (order=1, constant)"),
    ("Affine_order_3_constant", "Affine (order=3, constant)"),
    ("Affine_order_1_edge", "Affine (order=1, edge)"),
    ("Affine_order_1_constant_skimage", "Affine (order=1, constant, skimage)"),
    ("PiecewiseAffine_4x4_order_1_constant", "PiecewiseAffine (4x4, order=1, constant)"),
    ("PiecewiseAffine_4x4_order_0_constant", "PiecewiseAffine (4x4, order=0, constant)"),
    ("PiecewiseAffine_4x4_order_1_edge", "PiecewiseAffine (4x4, order=1, edge)"),
    ("PiecewiseAffine_8x8_order_1_constant", "PiecewiseAffine (8x8, order=1, constant)"),
    ("PerspectiveTransform", "PerspectiveTransform"),
    ("PerspectiveTransform_keep_size", "PerspectiveTransform (keep_size)"),
    ("ElasticTransformation_order_0_constant", "ElasticTransformation (order=0, constant)"),
    ("ElasticTransformation_order_1_constant", "ElasticTransformation (order=1, constant)"),
    ("ElasticTransformation_order_1_nearest", "ElasticTransformation (order=1, nearest)"),
    ("ElasticTransformation_order_1_reflect", "ElasticTransformation (order=1, reflect)"),
    ("Rot90", "Rot90"),
    ("Rot90_keep_size", "Rot90 (keep_size)"),
    ("Superpixels_max_size_64_cubic", "Superpixels (max_size=64, cubic)"),
    ("Superpixels_max_size_64_linear", "Superpixels (max_size=64, linear)"),
    ("Superpixels_max_size_128_linear", "Superpixels (max_size=128, linear)"),
    ("Superpixels_max_size_224_linear", "Superpixels (max_size=224, linear)"),
    ("Scale_nearest", "Resize (nearest)"),  # legacy support
    ("Scale_linear", "Resize (linear)"),  # legacy support
    ("Scale_cubic", "Resize (cubic)"),  # legacy support
    ("Resize_nearest", "Resize (nearest)"),
    ("Resize_linear", "Resize (linear)"),
    ("Resize_cubic", "Resize (cubic)"),
    ("CropAndPad", "CropAndPad"),
    ("CropAndPad_edge", "CropAndPad (edge)"),
    ("CropAndPad_keep_size", "CropAndPad (keep_size)"),
    ("Crop", "Crop"),
    ("Crop_keep_size", "Crop (keep_size)"),
    ("Pad", "Pad"),
    ("Pad_edge", "Pad (edge)"),
    ("Pad_keep_size", "Pad (keep_size)"),
    ("PadToFixedSize", "PadToFixedSize"),
    ("CropToFixedSize", "CropToFixedSize"),
    ("KeepSizeByResize_CropToFixedSize_nearest", "KeepSizeByResize<br>(CropToFixedSize(nearest))"),
    ("KeepSizeByResize_CropToFixedSize_linear", "KeepSizeByResize<br>(CropToFixedSize(linear))"),
    ("KeepSizeByResize_CropToFixedSize_cubic", "KeepSizeByResize<br>(CropToFixedSize(cubic))"),
    ("FastSnowyLandscape", "FastSnowyLandscape"),
    ("Clouds", "Clouds"),
    ("Fog", "Fog"),
    ("CloudLayer", "CloudLayer"),
    ("Snowflakes", "Snowflakes"),
    ("SnowflakesLayer", "SnowflakesLayer")
])


def main():
    dir_path = os.path.join("measure_performance_results", "027_after_performance_improvements")
    with open(os.path.join(dir_path, "results_images.pickle"), "rb") as f:
        measurements_images = pickle.load(f)

    with open(os.path.join(dir_path, "results_heatmaps.pickle"), "rb") as f:
        measurements_hms = pickle.load(f)

    with open(os.path.join(dir_path, "results_keypoints.pickle"), "rb") as f:
        measurements_kps = pickle.load(f)

    # images
    print("--------------------")
    print("IMAGES")
    print("--------------------")
    image_table_imgs_per_sec = TableForImageData(".1f")
    image_table_mbits_per_sec = TableForImageData(".1f")
    rows = defaultdict(dict)
    for subdict in measurements_images:
        augmentable = subdict["augmentable"]
        background = subdict["background"]
        image_shape = subdict["image.shape"]
        batch_size = subdict["batch_size"]
        augmenter_name = subdict["augmenter.name"]
        times = subdict["times"]

        assert augmentable == "images"
        assert background is False
        assert image_shape[0] in [64, 224]
        assert batch_size in [1, 128]

        # exp_name = EXP_NAME_MAPPING[augmenter_name]
        imgs_per_sec = batch_size / np.average(times)
        mbit_per_sec = (batch_size * np.prod(image_shape) * 8 / 1024 / 1024) / np.average(times)
        res_type = "a" if image_shape[0] == 64 else "b"
        bsize_type = "a" if batch_size == 1 else "b"
        rows[augmenter_name][(res_type, bsize_type)] = (imgs_per_sec, mbit_per_sec)

    for augmenter_name, exp_name in EXP_NAME_MAPPING.items():
        for table, index in zip([image_table_imgs_per_sec, image_table_mbits_per_sec], [0, 1]):
            table.add_row(
                exp_name,
                res_a_bsize_a=rows[augmenter_name][("a", "a")][index],
                res_a_bsize_b=rows[augmenter_name][("a", "b")][index],
                res_b_bsize_a=rows[augmenter_name][("b", "a")][index],
                res_b_bsize_b=rows[augmenter_name][("b", "b")][index]
            )

    # Without adding one dummy line, the last augmenter is skipped. If only adding one dummy line, the last augmenter
    # is broken (0s appear in cells).
    image_table_imgs_per_sec.add_row("", 0, 0, 0, 0)
    image_table_mbits_per_sec.add_row("", 0, 0, 0, 0)
    image_table_imgs_per_sec.add_row("", 0, 0, 0, 0)
    image_table_mbits_per_sec.add_row("", 0, 0, 0, 0)

    print("imgs mbit/sec:")
    print(image_table_mbits_per_sec.render_rst())
    # print(image_table_imgs_per_sec.render_html())
    print("imgs items/sec:")
    print(image_table_imgs_per_sec.render_rst())

    # heatmaps
    print("--------------------")
    print("HEATMAPS")
    print("--------------------")
    heatmaps_table_imgs_per_sec = TableForHeatmapsData(".1f")
    heatmaps_table_mbits_per_sec = TableForHeatmapsData(".1f")
    rows = defaultdict(dict)
    for subdict in measurements_hms:
        augmentable = subdict["augmentable"]
        background = subdict["background"]
        hms_shape = subdict["heatmaps_oi.arr_0to1.shape"]
        image_shape = subdict["heatmaps_oi.shape"]
        batch_size = subdict["batch_size"]
        augmenter_name = subdict["augmenter.name"]
        times = subdict["times"]

        assert augmentable == "heatmaps"
        assert background is False
        assert hms_shape[0] in [64, 224]
        assert hms_shape[2] in [1, 5]
        assert image_shape[0] in [64, 224]
        assert batch_size in [1, 128]

        if image_shape[0] == 224 and hms_shape[2] == 5:
            h, w, c = hms_shape
            items_per_sec = (1/np.average(times)) * batch_size * c
            mbit_per_img = (h * w * np.dtype("float32").itemsize * 8) / 1024 / 1024
            mbit_per_sec = items_per_sec * mbit_per_img

            res_type = "a" if hms_shape[0] == 64 else "b"
            bsize_type = "a" if batch_size == 1 else "b"
            rows[augmenter_name][(res_type, bsize_type)] = (items_per_sec, mbit_per_sec)

    for augmenter_name, exp_name in EXP_NAME_MAPPING.items():
        for table, index in zip([heatmaps_table_imgs_per_sec, heatmaps_table_mbits_per_sec], [0, 1]):
            table.add_row(
                exp_name,
                res_a_bsize_a=rows[augmenter_name][("a", "a")][index],
                res_a_bsize_b=rows[augmenter_name][("a", "b")][index],
                res_b_bsize_a=rows[augmenter_name][("b", "a")][index],
                res_b_bsize_b=rows[augmenter_name][("b", "b")][index]
            )

    # Without adding one dummy line, the last augmenter is skipped. If only adding one dummy line, the last augmenter
    # is broken (0s appear in cells).
    heatmaps_table_mbits_per_sec.add_row("", 0, 0, 0, 0)
    heatmaps_table_imgs_per_sec.add_row("", 0, 0, 0, 0)
    heatmaps_table_mbits_per_sec.add_row("", 0, 0, 0, 0)
    heatmaps_table_imgs_per_sec.add_row("", 0, 0, 0, 0)

    print("hms mbit/sec:")
    print(heatmaps_table_mbits_per_sec.render_rst())
    # print(image_table_imgs_per_sec.render_html())
    print("hms items/sec:")
    print(heatmaps_table_imgs_per_sec.render_rst())

    # keypoints
    print("--------------------")
    print("KEYPOINTS")
    print("--------------------")
    keypoints_table_imgs_per_sec = TableForKeypointsData(".1f")
    keypoints_table_mbits_per_sec = TableForKeypointsData(".1f")
    rows = defaultdict(dict)
    for subdict in measurements_kps:
        augmentable = subdict["augmentable"]
        background = subdict["background"]
        nb_points = subdict["nb_points"]
        keypoints_on_image_shape = subdict["keypoints_on_image.shape"]
        batch_size = subdict["batch_size"]
        augmenter_name = subdict["augmenter.name"]
        times = subdict["times"]

        assert augmentable == "keypoints"
        assert background is False
        assert nb_points in [1, 10]
        assert keypoints_on_image_shape[0] in [64, 224]
        assert batch_size in [1, 128]

        if keypoints_on_image_shape[0] in [64, 224] and nb_points == 10:
            items_per_sec = (1/np.average(times)) * batch_size * nb_points
            mbit_per_img = (nb_points * 2 * 32) / 1024 / 1024
            mbit_per_sec = items_per_sec * mbit_per_img

            res_type = "a" if keypoints_on_image_shape[0] == 64 else "b"
            bsize_type = "a" if batch_size == 1 else "b"
            rows[augmenter_name][(res_type, bsize_type)] = (items_per_sec, mbit_per_sec)

    for augmenter_name, exp_name in EXP_NAME_MAPPING.items():
        for table, index in zip([keypoints_table_imgs_per_sec, keypoints_table_mbits_per_sec], [0, 1]):
            table.add_row(
                exp_name,
                res_a_bsize_a=rows[augmenter_name][("a", "a")][index],
                res_a_bsize_b=rows[augmenter_name][("a", "b")][index],
                res_b_bsize_a=rows[augmenter_name][("b", "a")][index],
                res_b_bsize_b=rows[augmenter_name][("b", "b")][index]
            )

    # Without adding one dummy line, the last augmenter is skipped. If only adding one dummy line, the last augmenter
    # is broken (0s appear in cells).
    keypoints_table_mbits_per_sec.add_row("", 0, 0, 0, 0)
    keypoints_table_imgs_per_sec.add_row("", 0, 0, 0, 0)
    keypoints_table_mbits_per_sec.add_row("", 0, 0, 0, 0)
    keypoints_table_imgs_per_sec.add_row("", 0, 0, 0, 0)

    #print(keypoints_table_mbits_per_sec.render_rst())
    # print(image_table_imgs_per_sec.render_html())
    print("kps items/sec:")
    print(keypoints_table_imgs_per_sec.render_rst())


class TableForImageData(object):
    def __init__(self, cell_format):
        self.cell_format = cell_format
        self.rows = []

    def add_row(self, exp_name, res_a_bsize_a, res_a_bsize_b, res_b_bsize_a, res_b_bsize_b):
        self.rows.append({
            "augmenter": exp_name,
            "res_a_bsize_a": res_a_bsize_a,
            "res_a_bsize_b": res_a_bsize_b,
            "res_b_bsize_a": res_b_bsize_a,
            "res_b_bsize_b": res_b_bsize_b
        })

    def render_html(self):
        fmt = tuple([self.cell_format] * 4)
        rows_html = ""
        for row in self.rows:
            rows_html += ("""
            <tr>
                <td>{augmenter:s}</td>
                <td>{res_a_bsize_a:%s}</td>
                <td>{res_a_bsize_b:%s}</td>
                <td>{res_b_bsize_a:%s}</td>
                <td>{res_b_bsize_b:%s}</td>
            </tr>""" % fmt).format(**row)

        out = """
        <table>
            <tr>
                <th></th>
                <th colspan=2>64x64x3, uint8</th>
                <th colspan=2>224x224x3, uint8</th>
            </tr>
            <tr>
                <th>Augmenter</th>
                <th>B=1</th>
                <th>B=128</th>
                <th>B=1</th>
                <th>B=128</th>
            </tr>
            %s
        </table>""" % (rows_html,)
        return out

    def render_rst(self):
        return dashtable.html2rst(self.render_html())


class TableForHeatmapsData(object):
    def __init__(self, cell_format):
        self.cell_format = cell_format
        self.rows = []

    def add_row(self, exp_name, res_a_bsize_a, res_a_bsize_b, res_b_bsize_a, res_b_bsize_b):
        self.rows.append({
            "augmenter": exp_name,
            "res_a_bsize_a": res_a_bsize_a,
            "res_a_bsize_b": res_a_bsize_b,
            "res_b_bsize_a": res_b_bsize_a,
            "res_b_bsize_b": res_b_bsize_b
        })

    def render_html(self):
        fmt = tuple([self.cell_format] * 4)
        rows_html = ""
        for row in self.rows:
            rows_html += ("""
            <tr>
                <td>{augmenter:s}</td>
                <td>{res_a_bsize_a:%s}</td>
                <td>{res_a_bsize_b:%s}</td>
                <td>{res_b_bsize_a:%s}</td>
                <td>{res_b_bsize_b:%s}</td>
            </tr>""" % fmt).format(**row)

        out = """
        <table>
            <tr>
                <th></th>
                <th colspan=2>64x64x5 on 224x224x3</th>
                <th colspan=2>224x224x5 on 224x224x3</th>
            </tr>
            <tr>
                <th>Augmenter</th>
                <th>B=1</th>
                <th>B=128</th>
                <th>B=1</th>
                <th>B=128</th>
            </tr>
            %s
        </table>""" % (rows_html,)
        return out

    def render_rst(self):
        return dashtable.html2rst(self.render_html())


class TableForKeypointsData(object):
    def __init__(self, cell_format):
        self.cell_format = cell_format
        self.rows = []

    def add_row(self, exp_name, res_a_bsize_a, res_a_bsize_b, res_b_bsize_a, res_b_bsize_b):
        self.rows.append({
            "augmenter": exp_name,
            "res_a_bsize_a": res_a_bsize_a,
            "res_a_bsize_b": res_a_bsize_b,
            "res_b_bsize_a": res_b_bsize_a,
            "res_b_bsize_b": res_b_bsize_b
        })

    def render_html(self):
        fmt = tuple([self.cell_format] * 4)
        rows_html = ""
        for row in self.rows:
            rows_html += ("""
            <tr>
                <td>{augmenter:s}</td>
                <td>{res_a_bsize_a:%s}</td>
                <td>{res_a_bsize_b:%s}</td>
                <td>{res_b_bsize_a:%s}</td>
                <td>{res_b_bsize_b:%s}</td>
            </tr>""" % fmt).format(**row)

        out = """
        <table>
            <tr>
                <th></th>
                <th colspan=2>10 KPs on 64x64x3</th>
                <th colspan=2>10 KPs on 224x224x3</th>
            </tr>
            <tr>
                <th>Augmenter</th>
                <th>B=1</th>
                <th>B=128</th>
                <th>B=1</th>
                <th>B=128</th>
            </tr>
            %s
        </table>""" % (rows_html,)
        return out

    def render_rst(self):
        return dashtable.html2rst(self.render_html())


if __name__ == "__main__":
    main()
