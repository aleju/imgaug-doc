from __future__ import print_function, division

import os
import tempfile

import imageio

import imgaug as ia
import imgaug.augmenters as iaa

from . import utils


def main():
    chapter_augmenters_savedebugimageeverynbatches()


def chapter_augmenters_savedebugimageeverynbatches():
    fn_start = "debug/savedebugimageeverynbatches"

    image = ia.quokka_square((128, 128))
    bbsoi = ia.quokka_bounding_boxes((128, 128), extract="square")
    segmaps = ia.quokka_segmentation_map((128, 128), extract="square")

    folder_path = tempfile.mkdtemp()
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

    utils.save(
        "overview_of_augmenters",
        fn_start + ".jpg",
        image_debug,
        quality=95
    )


if __name__ == "__main__":
    main()
