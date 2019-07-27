from __future__ import print_function, division

from .utils import save, grid


def main():
    """Generate all example images for the chapter `Examples: Bounding Boxes`
    in the documentation."""
    chapter_examples_bounding_boxes_simple()
    chapter_examples_bounding_boxes_rotation()
    chapter_examples_bounding_boxes_ooi()
    chapter_examples_bounding_boxes_shift()
    chapter_examples_bounding_boxes_projection()
    chapter_examples_bounding_boxes_iou()


def chapter_examples_bounding_boxes_simple():
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


    ia.seed(1)

    image = ia.quokka(size=(256, 256))
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=65, y1=100, x2=200, y2=150),
        BoundingBox(x1=150, y1=80, x2=200, y2=130)
    ], shape=image.shape)

    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
        iaa.Affine(
            translate_px={"x": 40, "y": 60},
            scale=(0.5, 0.7)
        ) # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])

    # Augment BBs and images.
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    # print coordinates before/after augmentation (see below)
    # use .x1_int, .y_int, ... to get integer coordinates
    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
            i,
            before.x1, before.y1, before.x2, before.y2,
            after.x1, after.y1, after.x2, after.y2)
        )

    # image with BBs before/after augmentation (shown below)
    image_before = bbs.draw_on_image(image, size=2)
    image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])

    # ------------

    save(
        "examples_bounding_boxes",
        "simple.jpg",
        grid([image_before, image_after], cols=2, rows=1),
        quality=90
    )


def chapter_examples_bounding_boxes_rotation():
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    image = ia.quokka(size=(256, 256))
    bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=65, y1=100, x2=200, y2=150),
        ia.BoundingBox(x1=150, y1=80, x2=200, y2=130)
    ], shape=image.shape)

    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
        iaa.Affine(
            rotate=45,
        )
    ])

    # Make our sequence deterministic.
    # We can now apply it to the image and then to the BBs and it will
    # lead to the same augmentations.
    # IMPORTANT: Call this once PER BATCH, otherwise you will always get the
    # exactly same augmentations for every batch!
    seq_det = seq.to_deterministic()

    # Augment BBs and images.
    # As we only have one image and list of BBs, we use
    # [image] and [bbs] to turn both into lists (batches) for the
    # functions and then [0] to reverse that. In a real experiment, your
    # variables would likely already be lists.
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    # print coordinates before/after augmentation (see below)
    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        print("BB %d: (%d, %d, %d, %d) -> (%d, %d, %d, %d)" % (
            i,
            before.x1, before.y1, before.x2, before.y2,
            after.x1, after.y1, after.x2, after.y2)
        )

    # image with BBs before/after augmentation (shown below)
    image_before = bbs.draw_on_image(image, size=2)
    image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])

    # ------------

    save(
        "examples_bounding_boxes",
        "rotation.jpg",
        grid([image_before, image_after], cols=2, rows=1),
        quality=90
    )


def chapter_examples_bounding_boxes_ooi():
    import numpy as np
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


    ia.seed(1)

    GREEN = [0, 255, 0]
    ORANGE = [255, 140, 0]
    RED = [255, 0, 0]

    # Pad image with a 1px white and (BY-1)px black border
    def pad(image, by):
        image_border1 = ia.pad(image, top=1, right=1, bottom=1, left=1,
                               mode="constant", cval=255)
        image_border2 = ia.pad(image_border1, top=by-1, right=by-1,
                               bottom=by-1, left=by-1,
                               mode="constant", cval=0)
        return image_border2

    # Draw BBs on an image
    # and before doing that, extend the image plane by BORDER pixels.
    # Mark BBs inside the image plane with green color, those partially inside
    # with orange and those fully outside with red.
    def draw_bbs(image, bbs, border):
        image_border = pad(image, border)
        for bb in bbs.bounding_boxes:
            if bb.is_fully_within_image(image.shape):
                color = GREEN
            elif bb.is_partly_within_image(image.shape):
                color = ORANGE
            else:
                color = RED
            image_border = bb.shift(left=border, top=border)\
                             .draw_on_image(image_border, size=2, color=color)

        return image_border

    # Define example image with three small square BBs next to each other.
    # Augment these BBs by shifting them to the right.
    image = ia.quokka(size=(256, 256))
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=25, x2=75, y1=25, y2=75),
        BoundingBox(x1=100, x2=150, y1=25, y2=75),
        BoundingBox(x1=175, x2=225, y1=25, y2=75)
    ], shape=image.shape)

    seq = iaa.Affine(translate_px={"x": 120})
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    # Draw the BBs (a) in their original form, (b) after augmentation,
    # (c) after augmentation and removing those fully outside the image,
    # (d) after augmentation and removing those fully outside the image and
    # clipping those partially inside the image so that they are fully inside.
    image_before = draw_bbs(image, bbs, 100)
    image_after1 = draw_bbs(image_aug, bbs_aug, 100)
    image_after2 = draw_bbs(image_aug, bbs_aug.remove_out_of_image(), 100)
    image_after3 = draw_bbs(image_aug, bbs_aug.remove_out_of_image().clip_out_of_image(), 100)

    # ------------

    save(
        "examples_bounding_boxes",
        "ooi.jpg",
        grid([image_before, image_after1, np.zeros_like(image_before), image_after2, np.zeros_like(image_before), image_after3], cols=2, rows=3),
        #grid([image_before, image_after1], cols=2, rows=1),
        quality=90
    )


def chapter_examples_bounding_boxes_shift():
    import imgaug as ia
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


    ia.seed(1)

    # Define image and two bounding boxes
    image = ia.quokka(size=(256, 256))
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=25, x2=75, y1=25, y2=75),
        BoundingBox(x1=100, x2=150, y1=25, y2=75)
    ], shape=image.shape)

    # Move both BBs 25px to the right and the second BB 25px down
    bbs_shifted = bbs.shift(left=25)
    bbs_shifted.bounding_boxes[1] = bbs_shifted.bounding_boxes[1].shift(top=25)

    # Draw images before/after moving BBs
    image = bbs.draw_on_image(image, color=[0, 255, 0], size=2, alpha=0.75)
    image = bbs_shifted.draw_on_image(image, color=[0, 0, 255], size=2, alpha=0.75)

    # ------------

    save(
        "examples_bounding_boxes",
        "shift.jpg",
        grid([image], cols=1, rows=1),
        quality=90
    )


def chapter_examples_bounding_boxes_projection():
    import imgaug as ia
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


    ia.seed(1)

    # Define image with two bounding boxes
    image = ia.quokka(size=(256, 256))
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=25, x2=75, y1=25, y2=75),
        BoundingBox(x1=100, x2=150, y1=25, y2=75)
    ], shape=image.shape)

    # Rescale image and bounding boxes
    image_rescaled = ia.imresize_single_image(image, (512, 512))
    bbs_rescaled = bbs.on(image_rescaled)

    # Draw image before/after rescaling and with rescaled bounding boxes
    image_bbs = bbs.draw_on_image(image, size=2)
    image_rescaled_bbs = bbs_rescaled.draw_on_image(image_rescaled, size=2)

    # ------------

    save(
        "examples_bounding_boxes",
        "projection.jpg",
        grid([image_bbs, image_rescaled_bbs], cols=2, rows=1),
        quality=90
    )


def chapter_examples_bounding_boxes_iou():
    import numpy as np
    import imgaug as ia
    from imgaug.augmentables.bbs import BoundingBox


    ia.seed(1)

    # Define image with two bounding boxes.
    image = ia.quokka(size=(256, 256))
    bb1 = BoundingBox(x1=50, x2=100, y1=25, y2=75)
    bb2 = BoundingBox(x1=75, x2=125, y1=50, y2=100)

    # Compute intersection, union and IoU value
    # Intersection and union are both bounding boxes. They are here
    # decreased/increased in size purely for better visualization.
    bb_inters = bb1.intersection(bb2).extend(all_sides=-1)
    bb_union = bb1.union(bb2).extend(all_sides=2)
    iou = bb1.iou(bb2)

    # Draw bounding boxes, intersection, union and IoU value on image.
    image_bbs = np.copy(image)
    image_bbs = bb1.draw_on_image(image_bbs, size=2, color=[0, 255, 0])
    image_bbs = bb2.draw_on_image(image_bbs, size=2, color=[0, 255, 0])
    image_bbs = bb_inters.draw_on_image(image_bbs, size=2, color=[255, 0, 0])
    image_bbs = bb_union.draw_on_image(image_bbs, size=2, color=[0, 0, 255])
    image_bbs = ia.draw_text(
        image_bbs, text="IoU=%.2f" % (iou,),
        x=bb_union.x2+10, y=bb_union.y1+bb_union.height//2,
        color=[255, 255, 255], size=13
    )

    # ------------

    save(
        "examples_bounding_boxes",
        "iou.jpg",
        grid([image_bbs], cols=1, rows=1),
        quality=90
    )


if __name__ == "__main__":
    main()
