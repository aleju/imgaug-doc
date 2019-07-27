from __future__ import print_function, division
import numpy as np
import imgaug as ia
from imgaug import parameters as iap

from .utils import save, grid

PARAMETERS_DEFAULT_SIZE = (350, 350)
PARAMETER_DEFAULT_QUALITY = 25


def main():
    chapter_parameters_introduction()
    chapter_parameters_continuous()
    chapter_parameters_discrete()
    chapter_parameters_arithmetic()
    chapter_parameters_special()


def draw_distributions_grid(params, rows=None, cols=None, graph_sizes=PARAMETERS_DEFAULT_SIZE, sample_sizes=None, titles=False):
    return iap.draw_distributions_grid(
        params, rows=rows, cols=cols, graph_sizes=graph_sizes,
        sample_sizes=sample_sizes, titles=titles
    )


def chapter_parameters_introduction():
    ia.seed(1)
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap

    seq = iaa.Sequential([
        iaa.GaussianBlur(
            sigma=iap.Uniform(0.0, 1.0)
        ),
        iaa.ContrastNormalization(
            iap.Choice(
                [1.0, 1.5, 3.0],
                p=[0.5, 0.3, 0.2]
            )
        ),
        iaa.Affine(
            rotate=iap.Normal(0.0, 30),
            translate_px=iap.RandomSign(iap.Poisson(3))
        ),
        iaa.AddElementwise(
            iap.Discretize(
                (iap.Beta(0.5, 0.5) * 2 - 1.0) * 64
            )
        ),
        iaa.Multiply(
            iap.Positive(iap.Normal(0.0, 0.1)) + 1.0
        )
    ])

    images = np.array([ia.quokka_square(size=(128, 128)) for i in range(16)])
    images_aug = [seq.augment_image(images[i]) for i in range(len(images))]
    save(
        "parameters",
        "introduction.jpg",
        grid(images_aug, cols=4, rows=4),
        quality=25
    )


def chapter_parameters_continuous():
    ia.seed(1)

    # -----------------------
    # Normal
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Normal(0, 1),
        iap.Normal(5, 3),
        iap.Normal(iap.Choice([-3, 3]), 1),
        iap.Normal(iap.Uniform(-3, 3), 1)
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "continuous_normal.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Laplace
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Laplace(0, 1),
        iap.Laplace(5, 3),
        iap.Laplace(iap.Choice([-3, 3]), 1),
        iap.Laplace(iap.Uniform(-3, 3), 1)
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "continuous_laplace.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # ChiSquare
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.ChiSquare(1),
        iap.ChiSquare(3),
        iap.ChiSquare(iap.Choice([1, 5])),
        iap.RandomSign(iap.ChiSquare(3))
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "continuous_chisquare.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Weibull
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Weibull(0.5),
        iap.Weibull(1),
        iap.Weibull(1.5),
        iap.Weibull((0.5, 1.5))
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "continuous_weibull.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Uniform
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Uniform(0, 1),
        iap.Uniform(iap.Normal(-3, 1), iap.Normal(3, 1)),
        iap.Uniform([-1, 0], 1),
        iap.Uniform((-1, 0), 1)
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "continuous_uniform.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Beta
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Beta(0.5, 0.5),
        iap.Beta(2.0, 2.0),
        iap.Beta(1.0, 0.5),
        iap.Beta(0.5, 1.0)
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "continuous_beta.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )


def chapter_parameters_discrete():
    ia.seed(1)

    # -----------------------
    # Binomial
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Binomial(0.5),
        iap.Binomial(0.9)
    ]
    gridarr = draw_distributions_grid(params, rows=1)
    save(
        "parameters",
        "continuous_binomial.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # DiscreteUniform
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.DiscreteUniform(0, 10),
        iap.DiscreteUniform(-10, 10),
        iap.DiscreteUniform([-10, -9, -8, -7], 10),
        iap.DiscreteUniform((-10, -7), 10)
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "continuous_discreteuniform.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Poisson
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Poisson(1),
        iap.Poisson(2.5),
        iap.Poisson((1, 2.5)),
        iap.RandomSign(iap.Poisson(2.5))
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "continuous_poisson.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )


def chapter_parameters_arithmetic():
    ia.seed(1)

    # -----------------------
    # Add
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Uniform(0, 1) + 1, # identical to: Add(Uniform(0, 1), 1)
        iap.Add(iap.Uniform(0, 1), iap.Choice([0, 1], p=[0.7, 0.3])),
        iap.Normal(0, 1) + iap.Uniform(-5.5, -5) + iap.Uniform(5, 5.5),
        iap.Normal(0, 1) + iap.Uniform(-7, 5) + iap.Poisson(3),
        iap.Add(iap.Normal(-3, 1), iap.Normal(3, 1)),
        iap.Add(iap.Normal(-3, 1), iap.Normal(3, 1), elementwise=True)
    ]
    gridarr = draw_distributions_grid(
        params,
        rows=2,
        sample_sizes=[ # (iterations, samples per iteration)
            (1000, 1000), (1000, 1000), (1000, 1000),
            (1000, 1000), (1, 100000), (1, 100000)
        ]
    )
    save(
        "parameters",
        "arithmetic_add.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Multiply
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Uniform(0, 1) * 2, # identical to: Multiply(Uniform(0, 1), 2)
        iap.Multiply(iap.Uniform(0, 1), iap.Choice([0, 1], p=[0.7, 0.3])),
        (iap.Normal(0, 1) * iap.Uniform(-5.5, -5)) * iap.Uniform(5, 5.5),
        (iap.Normal(0, 1) * iap.Uniform(-7, 5)) * iap.Poisson(3),
        iap.Multiply(iap.Normal(-3, 1), iap.Normal(3, 1)),
        iap.Multiply(iap.Normal(-3, 1), iap.Normal(3, 1), elementwise=True)
    ]
    gridarr = draw_distributions_grid(
        params,
        rows=2,
        sample_sizes=[ # (iterations, samples per iteration)
            (1000, 1000), (1000, 1000), (1000, 1000),
            (1000, 1000), (1, 100000), (1, 100000)
        ]
    )
    save(
        "parameters",
        "arithmetic_multiply.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Divide
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Uniform(0, 1) / 2, # identical to: Divide(Uniform(0, 1), 2)
        iap.Divide(iap.Uniform(0, 1), iap.Choice([0, 2], p=[0.7, 0.3])),
        (iap.Normal(0, 1) / iap.Uniform(-5.5, -5)) / iap.Uniform(5, 5.5),
        (iap.Normal(0, 1) * iap.Uniform(-7, 5)) / iap.Poisson(3),
        iap.Divide(iap.Normal(-3, 1), iap.Normal(3, 1)),
        iap.Divide(iap.Normal(-3, 1), iap.Normal(3, 1), elementwise=True)
    ]
    gridarr = draw_distributions_grid(
        params,
        rows=2,
        sample_sizes=[ # (iterations, samples per iteration)
            (1000, 1000), (1000, 1000), (1000, 1000),
            (1000, 1000), (1, 100000), (1, 100000)
        ]
    )
    save(
        "parameters",
        "arithmetic_divide.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Power
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Uniform(0, 1) ** 2, # identical to: Power(Uniform(0, 1), 2)
        iap.Clip(iap.Uniform(-1, 1) ** iap.Normal(0, 1), -4, 4)
    ]
    gridarr = draw_distributions_grid(
        params,
        rows=1
    )
    save(
        "parameters",
        "arithmetic_power.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )


def chapter_parameters_special():
    ia.seed(1)

    # -----------------------
    # Choice
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Choice([0, 1, 2]),
        iap.Choice([0, 1, 2], p=[0.15, 0.5, 0.35]),
        iap.Choice([iap.Normal(-3, 1), iap.Normal(3, 1)]),
        iap.Choice([iap.Normal(-3, 1), iap.Poisson(3)])
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "special_choice.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Clip
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Clip(iap.Normal(0, 1), -2, 2),
        iap.Clip(iap.Normal(0, 1), -2, None)
    ]
    gridarr = draw_distributions_grid(params, rows=1)
    save(
        "parameters",
        "special_clip.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Discretize
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Discretize(iap.Normal(0, 1)),
        iap.Discretize(iap.ChiSquare(3))
    ]
    gridarr = draw_distributions_grid(params, rows=1)
    save(
        "parameters",
        "special_discretize.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # Absolute
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.Absolute(iap.Normal(0, 1)),
        iap.Absolute(iap.Laplace(0, 1))
    ]
    gridarr = draw_distributions_grid(params, rows=1)
    save(
        "parameters",
        "special_absolute.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # RandomSign
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.ChiSquare(3),
        iap.RandomSign(iap.ChiSquare(3)),
        iap.RandomSign(iap.ChiSquare(3), p_positive=0.75),
        iap.RandomSign(iap.ChiSquare(3), p_positive=0.9)
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "special_randomsign.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )

    # -----------------------
    # ForceSign
    # -----------------------
    from imgaug import parameters as iap
    params = [
        iap.ForceSign(iap.Normal(0, 1), positive=True),
        iap.ChiSquare(3) - 3.0,
        iap.ForceSign(iap.ChiSquare(3) - 3.0, positive=True, mode="invert"),
        iap.ForceSign(iap.ChiSquare(3) - 3.0, positive=True, mode="reroll")
    ]
    gridarr = draw_distributions_grid(params)
    save(
        "parameters",
        "special_forcesign.jpg",
        gridarr,
        quality=PARAMETER_DEFAULT_QUALITY
    )


if __name__ == "__main__":
    main()
