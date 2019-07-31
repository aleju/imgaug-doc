from __future__ import print_function, division

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import run_and_save_augseq


def main():
    chapter_augmenters_resize()
    chapter_augmenters_cropandpad()
    chapter_augmenters_pad()
    chapter_augmenters_crop()
    chapter_augmenters_padtofixedsize()
    chapter_augmenters_croptofixedsize()


def chapter_augmenters_resize():
    aug = iaa.Resize({"height": 32, "width": 64})
    run_and_save_augseq(
        "size/resize_32x64.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.Resize({"height": 32, "width": "keep-aspect-ratio"})
    run_and_save_augseq(
        "size/resize_32xkar.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.Resize((0.5, 1.0))
    run_and_save_augseq(
        "size/resize_50_to_100_percent.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.Resize({"height": (0.5, 0.75), "width": [16, 32, 64]})
    run_and_save_augseq(
        "size/resize_h_uniform_w_choice.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_cropandpad():
    aug = iaa.CropAndPad(percent=(-0.25, 0.25))
    run_and_save_augseq(
        "size/cropandpad_percent.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.CropAndPad(
        percent=(0, 0.2),
        pad_mode=["constant", "edge"],
        pad_cval=(0, 128)
    )
    run_and_save_augseq(
        "size/cropandpad_mode_cval.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.CropAndPad(
        px=((0, 30), (0, 10), (0, 30), (0, 10)),
        pad_mode=ia.ALL,
        pad_cval=(0, 128)
    )
    run_and_save_augseq(
        "size/cropandpad_pad_complex.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(32)], cols=8, rows=4
    )

    aug = iaa.CropAndPad(
        px=(-10, 10),
        sample_independently=False
    )
    run_and_save_augseq(
        "size/cropandpad_correlated.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )


def chapter_augmenters_pad():
    pass


def chapter_augmenters_crop():
    pass


def chapter_augmenters_padtofixedsize():
    fn_start = "size/padtofixedsize"
    aug_cls = iaa.PadToFixedSize

    aug = aug_cls(width=100, height=100)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(80, 80)) for _ in range(4*2)], cols=4, rows=2
    )

    aug = aug_cls(width=100, height=100, position="center")
    run_and_save_augseq(
        fn_start + "_center.jpg", aug,
        [ia.quokka(size=(80, 80)) for _ in range(4*1)], cols=4, rows=1
    )

    aug = aug_cls(width=100, height=100, pad_mode=ia.ALL)
    run_and_save_augseq(
        fn_start + "_pad_mode.jpg", aug,
        [ia.quokka(size=(80, 80)) for _ in range(4*2)], cols=4, rows=2
    )

    aug = iaa.Sequential([
        iaa.PadToFixedSize(width=100, height=100),
        iaa.CropToFixedSize(width=100, height=100)
    ])
    run_and_save_augseq(
        fn_start + "_with_croptofixedsize.jpg", aug,
        [ia.quokka(size=(80, 120)) for _ in range(4*2)], cols=4, rows=2
    )


def chapter_augmenters_croptofixedsize():
    fn_start = "size/croptofixedsize"
    aug_cls = iaa.CropToFixedSize

    aug = aug_cls(width=100, height=100)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(120, 120)) for _ in range(4*2)], cols=4, rows=2
    )

    aug = aug_cls(width=100, height=100, position="center")
    run_and_save_augseq(
        fn_start + "_center.jpg", aug,
        [ia.quokka(size=(120, 120)) for _ in range(4*1)], cols=4, rows=1
    )

    # third example is identical to third example in padtofixedsize
    # and already generated there


if __name__ == "__main__":
    main()
