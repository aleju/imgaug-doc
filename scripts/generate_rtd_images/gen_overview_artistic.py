from __future__ import print_function, division

import os

import imageio

import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.augmenters.meta as meta
import imgaug.parameters as iap

from .utils import run_and_save_augseq

FILE_DIR = os.path.realpath(os.path.dirname(__file__))
INPUT_IMAGES_DIR = os.path.join(FILE_DIR, "..", "..", "images",
                                "input_images")


def main():
    chapter_augmenters_cartoon()


def chapter_augmenters_cartoon():
    image1 = imageio.imread(os.path.join(
        INPUT_IMAGES_DIR, "1024px-Barack_Obama_family_portrait_2011.jpg"))
    image2 = imageio.imread(os.path.join(
        INPUT_IMAGES_DIR, "Pahalgam_Valley.jpg"))
    image3 = imageio.imread(os.path.join(
        INPUT_IMAGES_DIR, "1024px-Salad_platter.jpg"))

    image1 = ia.imresize_single_image(image1, 0.25)
    image2 = ia.imresize_single_image(image2, 0.25)
    image3 = ia.imresize_single_image(image3, 0.25)

    aug = iaa.Cartoon()

    run_and_save_augseq(
        "artistic/cartoon_people.jpg",
        IdentityOnFirstCall(aug),
        [image1] * 4, cols=4, rows=1
    )
    run_and_save_augseq(
        "artistic/cartoon_landscape.jpg",
        IdentityOnFirstCall(aug),
        [image2] * 4, cols=4, rows=1
    )
    run_and_save_augseq(
        "artistic/cartoon_object.jpg",
        IdentityOnFirstCall(aug),
        [image3] * 4, cols=4, rows=1
    )

    aug = iaa.Cartoon(blur_ksize=3, segmentation_size=1.0,
                      saturation=2.0, edge_prevalence=1.0)
    run_and_save_augseq(
        "artistic/cartoon_nonstochastic_people.jpg",
        IdentityOnFirstCall(aug),
        [image1] * 4, cols=4, rows=1
    )
    run_and_save_augseq(
        "artistic/cartoon_nonstochastic_landscape.jpg",
        IdentityOnFirstCall(aug),
        [image2] * 4, cols=4, rows=1
    )
    run_and_save_augseq(
        "artistic/cartoon_nonstochastic_object.jpg",
        IdentityOnFirstCall(aug),
        [image3] * 4, cols=4, rows=1
    )


class IdentityOnFirstCall(meta.Augmenter):
    def __init__(self, after_first_call, *args, **kwargs):
        super(IdentityOnFirstCall, self).__init__(*args, **kwargs)
        self.after_first_call = after_first_call
        self.call_idx = 0

    def _augment_batch_(self, batch, random_state, parents, hooks):
        self.call_idx += 1
        if self.call_idx == 1:
            return batch
        return self.after_first_call._augment_batch_(batch, random_state,
                                                     parents, hooks)

    def get_parameters(self):
        return []


if __name__ == "__main__":
    main()
