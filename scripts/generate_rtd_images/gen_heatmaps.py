from __future__ import print_function, division

from .utils import save


def main():
    """Generate all example images for the chapter `Examples: Heatmaps`
    in the documentation."""
    chapter_examples_heatmaps_simple()
    chapter_examples_heatmaps_multiple_small()
    chapter_examples_heatmaps_arr_small()
    chapter_examples_heatmaps_resizing()
    chapter_examples_heatmaps_padding()


def chapter_examples_heatmaps_simple():
    import imageio
    import numpy as np
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug.augmentables.heatmaps import HeatmapsOnImage

    ia.seed(1)

    # Load an example image (uint8, 128x128x3).
    image = ia.quokka(size=(128, 128), extract="square")

    # Create an example depth map (float32, 128x128).
    # Here, we use a simple gradient that has low values (around 0.0)
    # towards the left of the image and high values (around 50.0)
    # towards the right. This is obviously a very unrealistic depth
    # map, but makes the example easier.
    depth = np.linspace(0, 50, 128).astype(np.float32)  # 128 values from 0.0 to 50.0
    depth = np.tile(depth.reshape(1, 128), (128, 1))    # change to a horizontal gradient

    # We add a cross to the center of the depth map, so that we can more
    # easily see the effects of augmentations.
    depth[64-2:64+2, 16:128-16] = 0.75 * 50.0  # line from left to right
    depth[16:128-16, 64-2:64+2] = 1.0 * 50.0   # line from top to bottom

    # Convert our numpy array depth map to a heatmap object.
    # We have to add the shape of the underlying image, as that is necessary
    # for some augmentations.
    depth = HeatmapsOnImage(
        depth, shape=image.shape, min_value=0.0, max_value=50.0)

    # To save some computation time, we want our models to perform downscaling
    # and hence need the ground truth depth maps to be at a resolution of
    # 64x64 instead of the 128x128 of the input image.
    # Here, we use simple average pooling to perform the downscaling.
    depth = depth.avg_pool(2)

    # Define our augmentation pipeline.
    seq = iaa.Sequential([
        iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
        iaa.Sharpen((0.0, 1.0)),       # sharpen the image
        iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects heatmaps)
        iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects heatmaps)
    ], random_order=True)

    # Augment images and heatmaps.
    images_aug = []
    heatmaps_aug = []
    for _ in range(5):
        images_aug_i, heatmaps_aug_i = seq(image=image, heatmaps=depth)
        images_aug.append(images_aug_i)
        heatmaps_aug.append(heatmaps_aug_i)

    # We want to generate an image of original input images and heatmaps
    # before/after augmentation.
    # It is supposed to have five columns:
    # (1) original image,
    # (2) augmented image,
    # (3) augmented heatmap on top of augmented image,
    # (4) augmented heatmap on its own in jet color map,
    # (5) augmented heatmap on its own in intensity colormap.
    # We now generate the cells of these columns.
    #
    # Note that we add a [0] after each heatmap draw command. That's because
    # the heatmaps object can contain many sub-heatmaps and hence we draw
    # command returns a list of drawn sub-heatmaps.
    # We only used one sub-heatmap, so our lists always have one entry.
    cells = []
    for image_aug, heatmap_aug in zip(images_aug, heatmaps_aug):
        cells.append(image)                                                     # column 1
        cells.append(image_aug)                                                 # column 2
        cells.append(heatmap_aug.draw_on_image(image_aug)[0])                   # column 3
        cells.append(heatmap_aug.draw(size=image_aug.shape[:2])[0])             # column 4
        cells.append(heatmap_aug.draw(size=image_aug.shape[:2], cmap=None)[0])  # column 5

    # Convert cells to grid image and save.
    grid_image = ia.draw_grid(cells, cols=5)
    # imageio.imwrite("example_heatmaps.jpg", grid_image)

    save(
        "examples_heatmaps",
        "simple.jpg",
        grid_image,
        quality=90
    )


def chapter_examples_heatmaps_multiple_full():
    import imgaug as ia
    from imgaug import augmenters as iaa
    import imageio
    import numpy as np

    ia.seed(1)

    # Load an image and generate a heatmap array with three sub-heatmaps.
    # Each sub-heatmap contains just three horizontal lines, with one of them having a higher
    # value (1.0) than the other two (0.2).
    image = ia.quokka(size=(128, 128), extract="square")
    heatmap = np.zeros((128, 128, 3), dtype=np.float32)
    for i in range(3):
        heatmap[1*30-5:1*30+5, 10:-10, i] = 1.0 if i == 0 else 0.5
        heatmap[2*30-5:2*30+5, 10:-10, i] = 1.0 if i == 1 else 0.5
        heatmap[3*30-5:3*30+5, 10:-10, i] = 1.0 if i == 2 else 0.5

    # Convert heatmap array to heatmap object.
    heatmap = ia.HeatmapsOnImage(heatmap, shape=image.shape)

    # Define our augmentation pipeline.
    seq = iaa.Sequential([
        iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
        iaa.Sharpen((0.0, 1.0)),       # sharpen the image
        iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects heatmaps)
        iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects heatmaps)
    ], random_order=True)

    # Augment images and heatmaps.
    images_aug = []
    heatmaps_aug = []
    for _ in range(5):
        seq_det = seq.to_deterministic()
        images_aug.append(seq_det.augment_image(image))
        heatmaps_aug.append(seq_det.augment_heatmaps([heatmap])[0])

    # We want to generate an image of inputs before/after augmentation.
    # It is supposed to have five columns: (1) original image, (2) augmented image,
    # (3) augmented heatmap on top of augmented image, (4) augmented heatmap on its own in jet
    # color map, (5) augmented heatmap on its own in intensity colormap,
    # We now generate the cells of these columns.
    cells = []
    for image_aug, heatmap_aug in zip(images_aug, heatmaps_aug):
        subheatmaps_drawn = heatmap_aug.draw_on_image(image_aug)
        cells.append(image)                 # column 1
        cells.append(image_aug)             # column 2
        cells.append(subheatmaps_drawn[0])  # column 3
        cells.append(subheatmaps_drawn[1])  # column 4
        cells.append(subheatmaps_drawn[2])  # column 5

    # Convert cells to grid image and save.
    grid_image = ia.draw_grid(cells, cols=5)
    # imageio.imwrite("example_multiple_heatmaps.jpg", grid_image)

    save(
        "examples_heatmaps",
        "multiple_full.jpg",
        grid_image,
        quality=90
    )


def chapter_examples_heatmaps_multiple_small():
    import imageio
    import numpy as np
    import imgaug as ia
    from imgaug.augmentables.heatmaps import HeatmapsOnImage

    # Load an image and generate a heatmap array with three sub-heatmaps.
    # Each sub-heatmap contains just three horizontal lines, with one of them
    # having a higher value (1.0) than the other two (0.2).
    image = ia.quokka(size=(128, 128), extract="square")
    heatmap = np.zeros((128, 128, 3), dtype=np.float32)
    for i in range(3):
        heatmap[1*30-5:1*30+5, 10:-10, i] = 1.0 if i == 0 else 0.5
        heatmap[2*30-5:2*30+5, 10:-10, i] = 1.0 if i == 1 else 0.5
        heatmap[3*30-5:3*30+5, 10:-10, i] = 1.0 if i == 2 else 0.5
    heatmap = HeatmapsOnImage(heatmap, shape=image.shape)

    # Draw image and the three sub-heatmaps on it.
    # We draw four columns: (1) image, (2-4) heatmaps one to three drawn on
    # top of the image.
    subheatmaps_drawn = heatmap.draw_on_image(image)
    cells = [image, subheatmaps_drawn[0], subheatmaps_drawn[1],
             subheatmaps_drawn[2]]
    grid_image = np.hstack(cells)  # Horizontally stack the images
    # imageio.imwrite("example_multiple_heatmaps.jpg", grid_image)

    save(
        "examples_heatmaps",
        "multiple_small.jpg",
        grid_image,
        quality=90
    )


def chapter_examples_heatmaps_arr_full():
    import imgaug as ia
    from imgaug import augmenters as iaa
    import imageio
    import numpy as np

    ia.seed(1)

    # Load an image and generate a heatmap array with three sub-heatmaps.
    # Each sub-heatmap contains just three horizontal lines, with one of them having a higher
    # value (1.0) than the other two (0.2).
    image = ia.quokka(size=(128, 128), extract="square")
    heatmap = np.zeros((128, 128, 1), dtype=np.float32)
    heatmap[64-4:64+4, 10:-10, 0] = 1.0

    # Convert heatmap array to heatmap object.
    heatmap = ia.HeatmapsOnImage(heatmap, shape=image.shape)

    # Define our augmentation pipeline.
    seq = iaa.Sequential([
        iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
        iaa.Sharpen((0.0, 1.0)),       # sharpen the image
        iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects heatmaps)
        iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects heatmaps)
    ], random_order=True)

    # Augment images and heatmaps.
    images_aug = []
    heatmaps_aug = []
    for _ in range(5):
        seq_det = seq.to_deterministic()
        images_aug.append(seq_det.augment_image(image))
        heatmaps_aug.append(seq_det.augment_heatmaps([heatmap])[0])

    # We want to generate an image of inputs before/after augmentation.
    # It is supposed to have five columns: (1) original image, (2) augmented image,
    # (3) augmented heatmap on top of augmented image, (4) augmented heatmap on top of augmented
    # image with a vertical line added to the heatmap *after* augmentation.
    # We now generate the cells of these columns.
    cells = []
    for image_aug, heatmap_aug in zip(images_aug, heatmaps_aug):
        arr = heatmap_aug.get_arr()  # float32, shape (128, 128, 1)
        arr[10:-10, 64-4:64+4] = 0.5
        arr_heatmap = ia.HeatmapsOnImage(arr, shape=image_aug.shape)

        cells.append(image)                                    # column 1
        cells.append(image_aug)                                # column 2
        cells.append(heatmap_aug.draw_on_image(image_aug)[0])  # column 3
        cells.append(arr_heatmap.draw_on_image(image_aug)[0])  # column 4

    # Convert cells to grid image and save.
    grid_image = ia.draw_grid(cells, cols=4)
    # imageio.imwrite("example_heatmaps_arr.jpg", grid_image)

    save(
        "examples_heatmaps",
        "arr_full.jpg",
        grid_image
    )


def chapter_examples_heatmaps_arr_small():
    import imageio
    import numpy as np
    import imgaug as ia
    from imgaug.augmentables.heatmaps import HeatmapsOnImage

    # Load an image and generate a heatmap array containing one horizontal line.
    image = ia.quokka(size=(128, 128), extract="square")
    heatmap = np.zeros((128, 128, 1), dtype=np.float32)
    heatmap[64-4:64+4, 10:-10, 0] = 1.0
    heatmap1 = HeatmapsOnImage(heatmap, shape=image.shape)

    # Extract the heatmap array from the heatmap object, change it and create
    # a second heatmap.
    arr = heatmap1.get_arr()
    arr[10:-10, 64-4:64+4] = 0.5
    heatmap2 = HeatmapsOnImage(arr, shape=image.shape)

    # Draw image and heatmaps before/after changing the array.
    # We draw three columns:
    # (1) original image,
    # (2) heatmap drawn on image,
    # (3) heatmap drawn on image, with some changes made to the heatmap array.
    cells = [image,
             heatmap1.draw_on_image(image)[0],
             heatmap2.draw_on_image(image)[0]]
    grid_image = np.hstack(cells)  # Horizontally stack the images
    # imageio.imwrite("example_heatmaps_arr.jpg", grid_image)

    save(
        "examples_heatmaps",
        "arr_small.jpg",
        grid_image,
        quality=90
    )


def chapter_examples_heatmaps_resizing():
    import imageio
    import numpy as np
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug.augmentables.heatmaps import HeatmapsOnImage


    def pad_by(image, amount):
        return ia.pad(image,
                      top=amount, right=amount, bottom=amount, left=amount)

    def draw_heatmaps(heatmaps, upscale=False):
        drawn = []
        for heatmap in heatmaps:
            if upscale:
                drawn.append(
                    heatmap.resize((128, 128), interpolation="nearest")
                           .draw()[0]
                )
            else:
                size = heatmap.get_arr().shape[0]
                pad_amount = (128-size)//2
                drawn.append(pad_by(heatmap.draw()[0], pad_amount))
        return drawn

    # Generate an example heatmap with two horizontal lines (first one blurry,
    # second not) and a small square.
    heatmap = np.zeros((128, 128, 1), dtype=np.float32)
    heatmap[32-4:32+4, 10:-10, 0] = 1.0
    heatmap = iaa.GaussianBlur(3.0).augment_image(heatmap)
    heatmap[96-4:96+4, 10:-10, 0] = 1.0
    heatmap[64-2:64+2, 64-2:64+2, 0] = 1.0
    heatmap = HeatmapsOnImage(heatmap, shape=(128, 128, 1))

    # Scale the heatmaps using average pooling, max pooling and resizing with
    # default interpolation (cubic).
    avg_pooled = [heatmap, heatmap.avg_pool(2), heatmap.avg_pool(4),
                  heatmap.avg_pool(8)]
    max_pooled = [heatmap, heatmap.max_pool(2), heatmap.max_pool(4),
                  heatmap.max_pool(8)]
    resized = [heatmap, heatmap.resize((64, 64)), heatmap.resize((32, 32)),
               heatmap.resize((16, 16))]

    # Draw an image of all scaled heatmaps.
    cells = draw_heatmaps(avg_pooled)\
        + draw_heatmaps(max_pooled)\
        + draw_heatmaps(resized)\
        + draw_heatmaps(avg_pooled, upscale=True)\
        + draw_heatmaps(max_pooled, upscale=True)\
        + draw_heatmaps(resized, upscale=True)
    grid_image = ia.draw_grid(cells, cols=4)
    # imageio.imwrite("example_heatmaps_scaling.jpg", grid_image)

    save(
        "examples_heatmaps",
        "resizing.jpg",
        grid_image,
        quality=90
    )


def chapter_examples_heatmaps_padding():
    import imageio
    import numpy as np
    import imgaug as ia
    from imgaug.augmentables.heatmaps import HeatmapsOnImage

    # Load example image and generate example heatmap with one horizontal line
    image = ia.quokka((128, 128), extract="square")
    heatmap = np.zeros((128, 128, 1), dtype=np.float32)
    heatmap[64-4:64+4, 10:-10, 0] = 1.0

    # Cut image and heatmap so that they are no longer squared
    image = image[32:-32, :, :]
    heatmap = heatmap[32:-32, :, :]

    heatmap = HeatmapsOnImage(heatmap, shape=(128, 128, 1))

    # Pad images and heatmaps by pixel amounts or to aspect ratios
    # We pad both back to squared size of 128x128
    images_padded = [
        ia.pad(image, top=32, bottom=32),
        ia.pad_to_aspect_ratio(image, 1.0)
    ]
    heatmaps_padded = [
        heatmap.pad(top=32, bottom=32),
        heatmap.pad_to_aspect_ratio(1.0)
    ]

    # Draw an image of all padded images and heatmaps
    cells = [
        images_padded[0],
        heatmaps_padded[0].draw_on_image(images_padded[0])[0],
        images_padded[1],
        heatmaps_padded[1].draw_on_image(images_padded[1])[0]
    ]

    grid_image = ia.draw_grid(cells, cols=2)
    # imageio.imwrite("example_heatmaps_padding.jpg", grid_image)

    save(
        "examples_heatmaps",
        "padding.jpg",
        grid_image,
        quality=90
    )


if __name__ == "__main__":
    main()
