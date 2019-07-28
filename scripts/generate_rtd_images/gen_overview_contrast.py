from __future__ import print_function, division

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import run_and_save_augseq


def main():
    chapter_augmenters_gammacontrast()
    chapter_augmenters_sigmoidcontrast()
    chapter_augmenters_logcontrast()
    chapter_augmenters_linearcontrast()
    chapter_augmenters_allchannelsclahe()


def chapter_augmenters_gammacontrast():
    fn_start = "contrast/gammacontrast"

    aug = iaa.GammaContrast((0.5, 2.0))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.GammaContrast((0.5, 2.0), per_channel=True)
    run_and_save_augseq(
        fn_start + "_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_sigmoidcontrast():
    fn_start = "contrast/sigmoidcontrast"

    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.SigmoidContrast(
        gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True)
    run_and_save_augseq(
        fn_start + "_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_logcontrast():
    fn_start = "contrast/logcontrast"

    aug = iaa.LogContrast(gain=(0.4, 1.6))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.LogContrast(gain=(0.4, 1.6), per_channel=True)
    run_and_save_augseq(
        fn_start + "_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_linearcontrast():
    fn_start = "contrast/linearcontrast"

    aug = iaa.LinearContrast((0.6, 1.4))
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.LinearContrast((0.6, 1.4), per_channel=True)
    run_and_save_augseq(
        fn_start + "_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


def chapter_augmenters_allchannelsclahe():
    fn_start = "contrast/allchannelsclahe"

    aug = iaa.AllChannelsCLAHE()
    run_and_save_augseq(
        fn_start + ".jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.AllChannelsCLAHE(clip_limit=(1, 10))
    run_and_save_augseq(
        fn_start + "_random_clip_limit.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)

    aug = iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True)
    run_and_save_augseq(
        fn_start + "_per_channel.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(4*2)], cols=4, rows=2)


if __name__ == "__main__":
    main()
