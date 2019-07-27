from __future__ import print_function, division

import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import run_and_save_augseq


def main():
    chapter_augmenters_convolve()
    chapter_augmenters_sharpen()
    chapter_augmenters_emboss()
    chapter_augmenters_edgedetect()
    chapter_augmenters_directededgedetect()


def chapter_augmenters_convolve():
    matrix = np.array([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]])
    aug = iaa.Convolve(matrix=matrix)
    run_and_save_augseq(
        "convolutional/convolve.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2,
        quality=50
    )

    def gen_matrix(image, nb_channels, random_state):
        matrix_A = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])
        matrix_B = np.array([[0, 0, 0],
                             [0, -4, 1],
                             [0, 2, 1]])
        if random_state.rand() < 0.5:
            return [matrix_A] * nb_channels
        else:
            return [matrix_B] * nb_channels
    aug = iaa.Convolve(matrix=gen_matrix)
    run_and_save_augseq(
        "convolutional/convolve_callable.jpg", aug,
        [ia.quokka(size=(128, 128)) for _ in range(8)], cols=4, rows=2
    )


def chapter_augmenters_sharpen():
    aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
    run_and_save_augseq(
        "convolutional/sharpen.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    #alphas = [1/8*i for i in range(8)]
    alphas = np.linspace(0, 1.0, num=8)
    run_and_save_augseq(
        "convolutional/sharpen_vary_alpha.jpg",
        [iaa.Sharpen(alpha=alpha, lightness=1.0) for alpha in alphas],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1,
        quality=90
    )

    #lightnesses = [1/8*i for i in range(8)]
    lightnesses = np.linspace(0.75, 1.5, num=8)
    run_and_save_augseq(
        "convolutional/sharpen_vary_lightness.jpg",
        [iaa.Sharpen(alpha=1.0, lightness=lightness) for lightness in lightnesses],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1,
        quality=90
    )


def chapter_augmenters_emboss():
    aug = iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))
    run_and_save_augseq(
        "convolutional/emboss.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    #alphas = [1/8*i for i in range(8)]
    alphas = np.linspace(0, 1.0, num=8)
    run_and_save_augseq(
        "convolutional/emboss_vary_alpha.jpg",
        [iaa.Emboss(alpha=alpha, strength=1.0) for alpha in alphas],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1
    )

    #strengths = [0.5+(0.5/8)*i for i in range(8)]
    strengths = np.linspace(0.5, 1.5, num=8)
    run_and_save_augseq(
        "convolutional/emboss_vary_strength.jpg",
        [iaa.Emboss(alpha=1.0, strength=strength) for strength in strengths],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1
    )


def chapter_augmenters_edgedetect():
    aug = iaa.EdgeDetect(alpha=(0.0, 1.0))
    run_and_save_augseq(
        "convolutional/edgedetect.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    #alphas = [1/8*i for i in range(8)]
    alphas = np.linspace(0, 1.0, num=8)
    run_and_save_augseq(
        "convolutional/edgedetect_vary_alpha.jpg",
        [iaa.EdgeDetect(alpha=alpha) for alpha in alphas],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1
    )


def chapter_augmenters_directededgedetect():
    aug = iaa.DirectedEdgeDetect(alpha=(0.0, 1.0), direction=(0.0, 1.0))
    run_and_save_augseq(
        "convolutional/directededgedetect.jpg", aug,
        [ia.quokka(size=(64, 64)) for _ in range(16)], cols=8, rows=2
    )

    #alphas = [1/8*i for i in range(8)]
    alphas = np.linspace(0, 1.0, num=8)
    run_and_save_augseq(
        "convolutional/directededgedetect_vary_alpha.jpg",
        [iaa.DirectedEdgeDetect(alpha=alpha, direction=0) for alpha in alphas],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1
    )

    #strength = [0.5+(0.5/8)*i for i in range(8)]
    directions = np.linspace(0.0, 1.0, num=8)
    run_and_save_augseq(
        "convolutional/directededgedetect_vary_direction.jpg",
        [iaa.DirectedEdgeDetect(alpha=1.0, direction=direction) for direction in directions],
        [ia.quokka(size=(64, 64)) for _ in range(8)], cols=8, rows=1
    )


if __name__ == "__main__":
    main()
