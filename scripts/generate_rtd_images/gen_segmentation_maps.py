from __future__ import print_function, division

from .utils import save


def main():
    """Generate all example images for the chapter `Examples: Segmentation Maps`
    in the documentation."""
    chapter_examples_segmentation_maps_simple()
    # chapter_examples_segmentation_maps_bool_full()
    chapter_examples_segmentation_maps_bool_small()
    chapter_examples_segmentation_maps_array()


def chapter_examples_segmentation_maps_simple():
    import imageio
    import numpy as np
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage

    ia.seed(1)

    # Load an example image (uint8, 128x128x3).
    image = ia.quokka(size=(128, 128), extract="square")

    # Define an example segmentation map (int32, 128x128).
    # Here, we arbitrarily place some squares on the image.
    # Class 0 is our intended background class.
    segmap = np.zeros((128, 128, 1), dtype=np.int32)
    segmap[28:71, 35:85, 0] = 1
    segmap[10:25, 30:45, 0] = 2
    segmap[10:25, 70:85, 0] = 3
    segmap[10:110, 5:10, 0] = 4
    segmap[118:123, 10:110, 0] = 5
    segmap = SegmentationMapsOnImage(segmap, shape=image.shape)

    # Define our augmentation pipeline.
    seq = iaa.Sequential([
        iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
        iaa.Sharpen((0.0, 1.0)),       # sharpen the image
        iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
        iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
    ], random_order=True)

    # Augment images and segmaps.
    images_aug = []
    segmaps_aug = []
    for _ in range(5):
        images_aug_i, segmaps_aug_i = seq(image=image, segmentation_maps=segmap)
        images_aug.append(images_aug_i)
        segmaps_aug.append(segmaps_aug_i)

    # We want to generate an image containing the original input image and
    # segmentation maps before/after augmentation. (Both multiple times for
    # multiple augmentations.)
    #
    # The whole image is supposed to have five columns:
    # (1) original image,
    # (2) original image with segmap,
    # (3) augmented image,
    # (4) augmented segmap on augmented image,
    # (5) augmented segmap on its own in.
    #
    # We now generate the cells of these columns.
    #
    # Note that draw_on_image() and draw() both return lists of drawn
    # images. Assuming that the segmentation map array has shape (H,W,C),
    # the list contains C items.
    cells = []
    for image_aug, segmap_aug in zip(images_aug, segmaps_aug):
        cells.append(image)                                         # column 1
        cells.append(segmap.draw_on_image(image)[0])                # column 2
        cells.append(image_aug)                                     # column 3
        cells.append(segmap_aug.draw_on_image(image_aug)[0])        # column 4
        cells.append(segmap_aug.draw(size=image_aug.shape[:2])[0])  # column 5

    # Convert cells to a grid image and save.
    grid_image = ia.draw_grid(cells, cols=5)
    # imageio.imwrite("example_segmaps.jpg", grid_image)

    save(
        "examples_segmentation_maps",
        "simple.jpg",
        grid_image,
        quality=90
    )


def chapter_examples_segmentation_maps_bool_full():
    import imgaug as ia
    from imgaug import augmenters as iaa
    import imageio
    import numpy as np

    ia.seed(1)

    # Load an example image (uint8, 128x128x3).
    image = ia.quokka(size=(128, 128), extract="square")

    # Create an example mask (bool, 128x128).
    # Here, we just randomly place a square on the image.
    segmap = np.zeros((128, 128), dtype=bool)
    segmap[28:71, 35:85] = True
    segmap = ia.SegmentationMapOnImage(segmap, shape=image.shape)

    # Define our augmentation pipeline.
    seq = iaa.Sequential([
        iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
        iaa.Sharpen((0.0, 1.0)),       # sharpen the image
        iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects heatmaps)
        iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects heatmaps)
    ], random_order=True)

    # Augment images and heatmaps.
    images_aug = []
    segmaps_aug = []
    for _ in range(5):
        seq_det = seq.to_deterministic()
        images_aug.append(seq_det.augment_image(image))
        segmaps_aug.append(seq_det.augment_segmentation_maps([segmap])[0])

    # We want to generate an image of original input images and heatmaps before/after augmentation.
    # It is supposed to have five columns: (1) original image, (2) augmented image,
    # (3) augmented heatmap on top of augmented image, (4) augmented heatmap on its own in jet
    # color map, (5) augmented heatmap on its own in intensity colormap,
    # We now generate the cells of these columns.
    #
    # Note that we add a [0] after each heatmap draw command. That's because the heatmaps object
    # can contain many sub-heatmaps and hence we draw command returns a list of drawn sub-heatmaps.
    # We only used one sub-heatmap, so our lists always have one entry.
    cells = []
    for image_aug, segmap_aug in zip(images_aug, segmaps_aug):
        cells.append(image)                                      # column 1
        cells.append(segmap.draw_on_image(image))                # column 2
        cells.append(image_aug)                                  # column 3
        cells.append(segmap_aug.draw_on_image(image_aug))        # column 4
        cells.append(segmap_aug.draw(size=image_aug.shape[:2]))  # column 5

    # Convert cells to grid image and save.
    grid_image = ia.draw_grid(cells, cols=5)
    #imageio.imwrite("example_segmaps_bool.jpg", grid_image)

    save(
        "examples_segmentation_maps",
        "bool_full.jpg",
        grid_image,
        quality=90
    )


def chapter_examples_segmentation_maps_bool_small():
    import imageio
    import numpy as np
    import imgaug as ia
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage

    # Load an example image (uint8, 128x128x3).
    image = ia.quokka(size=(128, 128), extract="square")

    # Create an example mask (bool, 128x128).
    # Here, we arbitrarily place a square on the image.
    segmap = np.zeros((128, 128, 1), dtype=bool)
    segmap[28:71, 35:85, 0] = True
    segmap = SegmentationMapsOnImage(segmap, shape=image.shape)

    # Draw three columns: (1) original image,
    # (2) original image with mask on top, (3) only mask
    cells = [
        image,
        segmap.draw_on_image(image)[0],
        segmap.draw(size=image.shape[:2])[0]
    ]

    # Convert cells to a grid image and save.
    grid_image = ia.draw_grid(cells, cols=3)
    # imageio.imwrite("example_segmaps_bool.jpg", grid_image)

    save(
        "examples_segmentation_maps",
        "bool_small.jpg",
        grid_image,
        quality=90
    )


def chapter_examples_segmentation_maps_array():
    import imageio
    import numpy as np
    import imgaug as ia
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage

    # Load an example image (uint8, 128x128x3).
    image = ia.quokka(size=(128, 128), extract="square")

    # Create an example segmentation map (int32, 128x128).
    # Here, we arbitrarily place some squares on the image.
    # Class 0 is the background class.
    segmap = np.zeros((128, 128, 1), dtype=np.int32)
    segmap[28:71, 35:85, 0] = 1
    segmap[10:25, 30:45, 0] = 2
    segmap[10:25, 70:85, 0] = 3
    segmap[10:110, 5:10, 0] = 4
    segmap[118:123, 10:110, 0] = 5
    segmap1 = SegmentationMapsOnImage(segmap, shape=image.shape)

    # Read out the segmentation map's array, change it and create a new
    # segmentation map
    arr = segmap1.get_arr()
    arr[10:110, 5:10, 0] = 5
    segmap2 = ia.SegmentationMapsOnImage(arr, shape=image.shape)

    # Draw three columns: (1) original image, (2) original image with
    # unaltered segmentation map on top, (3) original image with altered
    # segmentation map on top
    cells = [
        image,
        segmap1.draw_on_image(image)[0],
        segmap2.draw_on_image(image)[0]
    ]

    # Convert cells to grid image and save.
    grid_image = ia.draw_grid(cells, cols=3)
    # imageio.imwrite("example_segmaps_array.jpg", grid_image)

    save(
        "examples_segmentation_maps",
        "array.jpg",
        grid_image,
        quality=90
    )


if __name__ == "__main__":
    main()
