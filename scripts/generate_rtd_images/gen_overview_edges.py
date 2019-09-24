from __future__ import print_function, division

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import run_and_save_augseq


def main():
    chapter_augmenters_canny()


def chapter_augmenters_canny():
    fn_start = "edges/canny"

    aug = iaa.Canny()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.Canny(alpha=(0.0, 0.5))
    run_and_save_augseq(
        fn_start + "_alpha.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.Canny(
        alpha=(0.0, 0.5),
        colorizer=iaa.RandomColorsBinaryImageColorizer(
            color_true=255,
            color_false=0
        )
    )
    run_and_save_augseq(
        fn_start + "_alpha_white_on_black.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.Canny(alpha=(0.5, 1.0), sobel_kernel_size=[3, 7])
    run_and_save_augseq(
        fn_start + "_sobel_kernel_size.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.Alpha(
        (0.0, 1.0),
        iaa.Canny(alpha=1),
        iaa.MedianBlur(13)
    )
    run_and_save_augseq(
        fn_start + "_alpha_median_blur.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


if __name__ == "__main__":
    main()
