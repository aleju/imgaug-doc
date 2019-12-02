from __future__ import print_function, division

import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap

from .utils import run_and_save_augseq


def main():
    chapter_augmenters_add()
    chapter_augmenters_addelementwise()
    chapter_augmenters_additivegaussiannoise()
    chapter_augmenters_additivelaplacenoise()
    chapter_augmenters_additivepoissonnoise()
    chapter_augmenters_replaceelementwise()
    chapter_augmenters_impulsenoise()
    chapter_augmenters_saltandpepper()
    chapter_augmenters_coarsesaltandpepper()
    chapter_augmenters_salt()
    chapter_augmenters_coarsesalt()
    chapter_augmenters_pepper()
    chapter_augmenters_coarsepepper()
    chapter_augmenters_multiply()
    chapter_augmenters_multiplyelementwise()
    chapter_augmenters_dropout()
    chapter_augmenters_coarsedropout()
    chapter_augmenters_dropout2d()
    chapter_augmenters_totaldropout()
    chapter_augmenters_invert()
    chapter_augmenters_solarize()
    chapter_augmenters_contrastnormalization()
    chapter_augmenters_jpegcompression()


def chapter_augmenters_add():
    aug = iaa.Add((-40, 40))
    run_and_save_augseq(
        "arithmetic/add.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )

    aug = iaa.Add((-40, 40), per_channel=0.5)
    run_and_save_augseq(
        "arithmetic/add_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2)


def chapter_augmenters_addelementwise():
    aug = iaa.AddElementwise((-40, 40))
    run_and_save_augseq(
        "arithmetic/addelementwise.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1,
        quality=90
    )

    aug = iaa.AddElementwise((-40, 40), per_channel=0.5)
    run_and_save_augseq(
        "arithmetic/addelementwise_per_channel.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1,
        quality=90
    )


def chapter_augmenters_additivegaussiannoise():
    aug = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
    run_and_save_augseq(
        "arithmetic/additivegaussiannoise.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95
    )

    aug = iaa.AdditiveGaussianNoise(scale=0.2*255)
    run_and_save_augseq(
        "arithmetic/additivegaussiannoise_large.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1,
        quality=95
    )

    aug = iaa.AdditiveGaussianNoise(scale=0.2*255, per_channel=True)
    run_and_save_augseq(
        "arithmetic/additivegaussiannoise_per_channel.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1,
        quality=95
    )


def chapter_augmenters_additivelaplacenoise():
    aug_cls = iaa.AdditiveLaplaceNoise
    fn_start = "arithmetic/additivelaplacenoise"

    aug = aug_cls(scale=(0, 0.2*255))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95
    )

    aug = aug_cls(scale=0.2*255)
    run_and_save_augseq(
        fn_start + "_large.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1,
        quality=95
    )

    aug = aug_cls(scale=0.2*255, per_channel=True)
    run_and_save_augseq(
        fn_start + "_per_channel.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1,
        quality=95
    )


def chapter_augmenters_additivepoissonnoise():
    aug_cls = iaa.AdditivePoissonNoise
    fn_start = "arithmetic/additivepoissonnoise"

    aug = aug_cls((0, 40))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95
    )

    aug = aug_cls(40)
    run_and_save_augseq(
        fn_start + "_large.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1,
        quality=95
    )

    aug = aug_cls(40, per_channel=True)
    run_and_save_augseq(
        fn_start + "_per_channel.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1,
        quality=95
    )


def chapter_augmenters_replaceelementwise():
    aug_cls = iaa.ReplaceElementwise
    fn_start = "arithmetic/replaceelementwise"

    aug = aug_cls(0.05, [0, 255])
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95,
        seed=2
    )

    aug = aug_cls(0.05, [0, 255], per_channel=0.5)
    run_and_save_augseq(
        fn_start + "_per_channel_050.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95,
        seed=2
    )

    aug = aug_cls(0.1, iap.Normal(128, 0.4*128), per_channel=0.5)
    run_and_save_augseq(
        fn_start + "_gaussian_noise.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95,
        seed=2
    )

    aug = aug_cls(
        iap.FromLowerResolution(iap.Binomial(0.1), size_px=8),
        iap.Normal(128, 0.4*128),
        per_channel=0.5)
    run_and_save_augseq(
        fn_start + "_gaussian_noise_coarse.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95,
        seed=2
    )


def chapter_augmenters_impulsenoise():
    fn_start = "arithmetic/impulsenoise"

    aug = iaa.ImpulseNoise(0.1)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95
    )


def chapter_augmenters_saltandpepper():
    fn_start = "arithmetic/saltandpepper"

    aug = iaa.SaltAndPepper(0.1)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95
    )

    aug = iaa.SaltAndPepper(0.1, per_channel=True)
    run_and_save_augseq(
        fn_start + "_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95
    )


def chapter_augmenters_coarsesaltandpepper():
    fn_start = "arithmetic/coarsesaltandpepper"

    aug = iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95
    )

    aug = iaa.CoarseSaltAndPepper(0.05, size_px=(4, 16))
    run_and_save_augseq(
        fn_start + "_pixels.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95
    )

    aug = iaa.CoarseSaltAndPepper(
        0.05, size_percent=(0.01, 0.1), per_channel=True)
    run_and_save_augseq(
        fn_start + "_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95
    )


def chapter_augmenters_salt():
    fn_start = "arithmetic/salt"

    aug = iaa.Salt(0.1)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95
    )


def chapter_augmenters_coarsesalt():
    fn_start = "arithmetic/coarsesalt"

    aug = iaa.CoarseSalt(0.05, size_percent=(0.01, 0.1))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95
    )


def chapter_augmenters_pepper():
    fn_start = "arithmetic/pepper"

    aug = iaa.Pepper(0.1)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95
    )


def chapter_augmenters_coarsepepper():
    fn_start = "arithmetic/coarsepepper"

    aug = iaa.CoarsePepper(0.05, size_percent=(0.01, 0.1))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=95
    )


def chapter_augmenters_multiply():
    aug = iaa.Multiply((0.5, 1.5))
    run_and_save_augseq(
        "arithmetic/multiply.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.Multiply((0.5, 1.5), per_channel=0.5)
    run_and_save_augseq(
        "arithmetic/multiply_per_channel.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )


def chapter_augmenters_multiplyelementwise():
    aug = iaa.MultiplyElementwise((0.5, 1.5))
    run_and_save_augseq(
        "arithmetic/multiplyelementwise.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1,
        quality=90
    )

    aug = iaa.MultiplyElementwise((0.5, 1.5), per_channel=True)
    run_and_save_augseq(
        "arithmetic/multiplyelementwise_per_channel.jpg", aug,
        [ia.quokka(size=(512, 512)) for _ in range(1)], cols=1, rows=1,
        quality=90
    )


def chapter_augmenters_dropout():
    aug = iaa.Dropout(p=(0, 0.2))
    run_and_save_augseq(
        "arithmetic/dropout.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2)

    aug = iaa.Dropout(p=(0, 0.2), per_channel=0.5)
    run_and_save_augseq(
        "arithmetic/dropout_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2)


def chapter_augmenters_coarsedropout():
    aug = iaa.CoarseDropout(0.02, size_percent=0.5)
    run_and_save_augseq(
        "arithmetic/coarsedropout.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2)

    aug = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))
    run_and_save_augseq(
        "arithmetic/coarsedropout_both_uniform.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        seed=2
    )

    aug = iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5)
    run_and_save_augseq(
        "arithmetic/coarsedropout_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        seed=2
    )


def chapter_augmenters_dropout2d():
    fn_start = "arithmetic/dropout2d"

    aug = iaa.Dropout2d(p=0.5)
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*4)], cols=4, rows=4
    )

    aug = iaa.Dropout2d(p=0.5, nb_keep_channels=0)
    run_and_save_augseq(
        fn_start + "_keep_no_channels.jpg", aug,
        [ia.quokka(size=(100, 100)) for _ in range(6*6)], cols=6, rows=6
    )


def chapter_augmenters_totaldropout():
    fn_start = "arithmetic/totaldropout"

    aug = iaa.TotalDropout(1.0)
    run_and_save_augseq(
        fn_start + "_100_percent.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2
    )

    aug = iaa.TotalDropout(0.5)
    run_and_save_augseq(
        fn_start + "_50_percent.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2
    )


def chapter_augmenters_invert():
    aug = iaa.Invert(0.5)
    run_and_save_augseq(
        "arithmetic/invert.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.Invert(0.25, per_channel=0.5)
    run_and_save_augseq(
        "arithmetic/invert_per_channel.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )


def chapter_augmenters_solarize():
    aug = iaa.Solarize(0.5, threshold=(32, 128))
    run_and_save_augseq(
        "arithmetic/solarize.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )


def chapter_augmenters_contrastnormalization():
    aug = iaa.ContrastNormalization((0.5, 1.5))
    run_and_save_augseq(
        "arithmetic/contrastnormalization.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    aug = iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)
    run_and_save_augseq(
        "arithmetic/contrastnormalization_per_channel.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )


def chapter_augmenters_jpegcompression():
    fn_start = "arithmetic/jpegcompression"

    aug = iaa.JpegCompression(compression=(70, 99))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(20)], cols=4, rows=5,
        quality=100
    )


if __name__ == "__main__":
    main()
