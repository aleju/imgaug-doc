from __future__ import print_function, division

import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap

from .utils import run_and_save_augseq


def main():
    chapter_augmenters_averagepooling()
    chapter_augmenters_maxpooling()
    chapter_augmenters_minpooling()
    chapter_augmenters_medianpooling()


def chapter_augmenters_averagepooling():
    fn_start = "pooling/averagepooling"
    aug_cls = iaa.AveragePooling

    aug = aug_cls(2)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)

    aug = aug_cls(2, keep_size=False)
    run_and_save_augseq(
        fn_start + "_keep_size_false.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)

    aug = aug_cls([2, 8])
    run_and_save_augseq(
        fn_start + "_choice.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls((1, 7))
    run_and_save_augseq(
        fn_start + "_uniform.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls(((1, 7), (1, 7)))
    run_and_save_augseq(
        fn_start + "_unsymmetric.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_maxpooling():
    fn_start = "pooling/maxpooling"
    aug_cls = iaa.MaxPooling

    aug = aug_cls(2)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)

    aug = aug_cls(2, keep_size=False)
    run_and_save_augseq(
        fn_start + "_keep_size_false.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)

    aug = aug_cls([2, 8])
    run_and_save_augseq(
        fn_start + "_choice.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls((1, 7))
    run_and_save_augseq(
        fn_start + "_uniform.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls(((1, 7), (1, 7)))
    run_and_save_augseq(
        fn_start + "_unsymmetric.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_minpooling():
    fn_start = "pooling/minpooling"
    aug_cls = iaa.MinPooling

    aug = aug_cls(2)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)

    aug = aug_cls(2, keep_size=False)
    run_and_save_augseq(
        fn_start + "_keep_size_false.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)

    aug = aug_cls([2, 8])
    run_and_save_augseq(
        fn_start + "_choice.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls((1, 7))
    run_and_save_augseq(
        fn_start + "_uniform.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls(((1, 7), (1, 7)))
    run_and_save_augseq(
        fn_start + "_unsymmetric.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_medianpooling():
    fn_start = "pooling/medianpooling"
    aug_cls = iaa.MedianPooling

    aug = aug_cls(2)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)

    aug = aug_cls(2, keep_size=False)
    run_and_save_augseq(
        fn_start + "_keep_size_false.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*1)], cols=4, rows=1)

    aug = aug_cls([2, 8])
    run_and_save_augseq(
        fn_start + "_choice.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls((1, 7))
    run_and_save_augseq(
        fn_start + "_uniform.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = aug_cls(((1, 7), (1, 7)))
    run_and_save_augseq(
        fn_start + "_unsymmetric.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


if __name__ == "__main__":
    main()
