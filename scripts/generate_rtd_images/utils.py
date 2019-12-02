from __future__ import print_function, division
import os
import math

from skimage import data
import numpy as np
import imageio
import PIL.Image

import imgaug as ia
import imgaug.augmenters as iaa

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

DOCS_IMAGES_BASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",  # scripts/
    "..",  # repo root
    "images"
)


def save(chapter_dir, filename, image, quality=None):
    file_fp = os.path.join(DOCS_IMAGES_BASE_PATH, chapter_dir, filename)
    dir_path = os.path.dirname(file_fp)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    image_jpg = compress_to_jpg(image, quality=quality)
    image_jpg_decompressed = decompress_jpg(image_jpg)

    # If the image file already exists and is (practically) identical,
    # then don't save it again to avoid polluting the repository with tons
    # of image updates.
    # Not that we have to compare here the results AFTER jpg compression
    # and then decompression. Otherwise we compare two images of which
    # image (1) has never been compressed while image (2) was compressed and
    # then decompressed.
    if os.path.isfile(file_fp):
        print(file_fp)
        image_saved = imageio.imread(file_fp)
        #print("arrdiff", arrdiff(image_jpg_decompressed, image_saved))
        same_shape = (image_jpg_decompressed.shape == image_saved.shape)
        d_avg = arrdiff(image_jpg_decompressed, image_saved) if same_shape else -1
        if same_shape and d_avg <= 1.0:
            print("[INFO] Did not save image '%s/%s', because the already saved image is basically identical (d_avg=%.4f)" % (chapter_dir, filename, d_avg,))
            return

    with open(file_fp, "wb") as f:
        f.write(image_jpg)


def arrdiff(arr1, arr2):
    nb_cells = np.prod(arr2.shape)
    d_avg = np.sum(np.power(np.abs(arr1 - arr2), 2)) / nb_cells
    return d_avg


def compress_to_jpg(image, quality=90):
    quality = quality if quality is not None else 90
    im = PIL.Image.fromarray(image)
    out = BytesIO()
    im.save(out, format="JPEG", quality=quality)
    jpg_string = out.getvalue()
    out.close()
    return jpg_string


def decompress_jpg(image_compressed):
    img_compressed_buffer = BytesIO()
    img_compressed_buffer.write(image_compressed)
    img = imageio.imread(img_compressed_buffer.getvalue(), pilmode="RGB", format="jpg")
    img_compressed_buffer.close()
    return img


def grid(images, rows, cols, border=1, border_color=255):
    nb_images = len(images)
    cell_height = max([image.shape[0] for image in images])
    cell_width = max([image.shape[1] for image in images])
    channels = set([image.shape[2] for image in images])
    assert len(channels) == 1
    nb_channels = list(channels)[0]
    if rows is None and cols is None:
        rows = cols = int(math.ceil(math.sqrt(nb_images)))
    elif rows is not None:
        cols = int(math.ceil(nb_images / rows))
    elif cols is not None:
        rows = int(math.ceil(nb_images / cols))
    assert rows * cols >= nb_images

    cell_height = cell_height + 2 * border
    cell_width = cell_width + 2 * border

    width = cell_width * cols
    height = cell_height * rows
    grid = np.zeros((height, width, nb_channels), dtype=np.uint8)
    cell_idx = 0
    for row_idx in range(rows):
        for col_idx in range(cols):
            if cell_idx < nb_images:
                image = images[cell_idx]
                border_top = border_right = border_bottom = border_left = border
                #if row_idx > 1:
                #border_top = 0
                #if col_idx > 1:
                #border_left = 0
                #image = np.pad(image, ((border_top, border_bottom), (border_left, border_right), (0, 0)), mode="constant", constant_values=border_color)
                image = ia.pad(
                    image,
                    top=border_top,
                    right=border_right,
                    bottom=border_bottom,
                    left=border_left,
                    mode="constant",
                    cval=border_color)

                image = iaa.PadToFixedSize(
                    height=cell_height, width=cell_width, position="center"
                )(image=image)

                cell_y1 = cell_height * row_idx
                cell_y2 = cell_y1 + image.shape[0]
                cell_x1 = cell_width * col_idx
                cell_x2 = cell_x1 + image.shape[1]
                grid[cell_y1:cell_y2, cell_x1:cell_x2, :] = image
            cell_idx += 1

    #grid = np.pad(grid, ((border, 0), (border, 0), (0, 0)), mode="constant", constant_values=border_color)

    return grid


def checkerboard(size):
    img = data.checkerboard()
    img3d = np.tile(img[..., np.newaxis], (1, 1, 3))
    return ia.imresize_single_image(img3d, size)


def run_and_save_augseq(filename, augseq, images, cols, rows, quality=95,
                        seed=1, image_colorspace="RGB"):
    ia.seed(seed)
    # augseq may be a single seq (applied to all images) or a list (one seq per
    # image).
    # use type() here instead of isinstance, because otherwise Sequential is
    # also interpreted as a list
    if type(augseq) == list:
        # one augmenter per image specified
        assert len(augseq) == len(images)
        images_aug = [augseq[i].augment_image(images[i]) for i in range(len(images))]
    else:
        # calling N times augment_image() is here critical for random order in
        # Sequential
        images_aug = [augseq.augment_image(images[i]) for i in range(len(images))]

    if image_colorspace != "RGB":
        images_aug = iaa.ChangeColorspace(from_colorspace=image_colorspace,
                                          to_colorspace="RGB")(images=images_aug)

    save(
        "overview_of_augmenters",
        filename,
        grid(images_aug, cols=cols, rows=rows),
        quality=quality
    )
