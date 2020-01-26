from __future__ import print_function, division

import numpy as np

from imgaug import parameters as iap
from .utils import save, grid


def main():
    """Generate all example images for the chapter `Alpha`
    in the documentation."""
    chapter_alpha_masks_introduction()
    chapter_alpha_constant()
    chapter_alpha_masks_simplex()
    chapter_alpha_masks_frequency()
    chapter_alpha_masks_iterative()
    chapter_alpha_masks_sigmoid()


def chapter_alpha_masks_introduction():
    # -----------------------------------------
    # example introduction
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(2)

    # Example batch of images.
    # The array has shape (8, 128, 128, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(128, 128)) for _ in range(8)],
        dtype=np.uint8
    )

    seqs = [
        iaa.BlendAlpha(
            (0.0, 1.0),
            foreground=iaa.MedianBlur(11),
            per_channel=True
        ),
        iaa.BlendAlphaSimplexNoise(
            foreground=iaa.EdgeDetect(1.0),
            per_channel=False
        ),
        iaa.BlendAlphaSimplexNoise(
            foreground=iaa.EdgeDetect(1.0),
            background=iaa.LinearContrast((0.5, 2.0)),
            per_channel=0.5
        ),
        iaa.BlendAlphaFrequencyNoise(
            foreground=iaa.Affine(
                rotate=(-10, 10),
                translate_px={"x": (-4, 4), "y": (-4, 4)}
            ),
            background=iaa.AddToHueAndSaturation((-40, 40)),
            per_channel=0.5
        ),
        iaa.BlendAlphaSimplexNoise(
            foreground=iaa.BlendAlphaSimplexNoise(
                foreground=iaa.EdgeDetect(1.0),
                background=iaa.LinearContrast((0.5, 2.0)),
                per_channel=True
            ),
            second=iaa.BlendAlphaFrequencyNoise(
                exponent=(-2.5, -1.0),
                foreground=iaa.Affine(
                    rotate=(-10, 10),
                    translate_px={"x": (-4, 4), "y": (-4, 4)}
                ),
                background=iaa.AddToHueAndSaturation((-40, 40)),
                per_channel=True
            ),
            per_channel=True,
            aggregation_method="max",
            sigmoid=False
        )
    ]

    cells = []
    for seq in seqs:
        images_aug = seq(images=images)
        cells.extend(images_aug)

    # ------------

    save(
        "alpha",
        "introduction.jpg",
        grid(cells, cols=8, rows=5)
    )


def chapter_alpha_constant():
    # -----------------------------------------
    # example 1 (sharpen + dropout)
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    # Example batch of images.
    # The array has shape (8, 128, 128, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(128, 128)) for _ in range(8)],
        dtype=np.uint8
    )

    seq = iaa.BlendAlpha(
        factor=(0.2, 0.8),
        foreground=iaa.Sharpen(1.0, lightness=2),
        background=iaa.CoarseDropout(p=0.1, size_px=8)
    )

    images_aug = seq(images=images)

    # ------------

    save(
        "alpha",
        "alpha_constant_example_basic.jpg",
        grid(images_aug, cols=4, rows=2)
    )

    # -----------------------------------------
    # example 2 (per channel)
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    # Example batch of images.
    # The array has shape (8, 128, 128, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(128, 128)) for _ in range(8)],
        dtype=np.uint8
    )

    seq = iaa.BlendAlpha(
        factor=(0.2, 0.8),
        foreground=iaa.Sharpen(1.0, lightness=2),
        background=iaa.CoarseDropout(p=0.1, size_px=8),
        per_channel=True
    )

    images_aug = seq(images=images)

    # ------------

    save(
        "alpha",
        "alpha_constant_example_per_channel.jpg",
        grid(images_aug, cols=4, rows=2)
    )

    # -----------------------------------------
    # example 3 (affine + per channel)
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    # Example batch of images.
    # The array has shape (8, 128, 128, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(128, 128)) for _ in range(8)],
        dtype=np.uint8
    )

    seq = iaa.BlendAlpha(
        factor=(0.2, 0.8),
        foreground=iaa.Affine(rotate=(-20, 20)),
        per_channel=True
    )

    images_aug = seq(images=images)

    # ------------

    save(
        "alpha",
        "alpha_constant_example_affine.jpg",
        grid(images_aug, cols=4, rows=2)
    )


def chapter_alpha_masks_simplex():
    # -----------------------------------------
    # example 1 (basic)
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    # Example batch of images.
    # The array has shape (8, 128, 128, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(128, 128)) for _ in range(8)],
        dtype=np.uint8
    )

    seq = iaa.BlendAlphaSimplexNoise(
        foreground=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)
    )

    images_aug = seq(images=images)

    # ------------

    save(
        "alpha",
        "alpha_simplex_example_basic.jpg",
        grid(images_aug, cols=4, rows=2)
    )

    # -----------------------------------------
    # example 1 (per_channel)
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    # Example batch of images.
    # The array has shape (8, 128, 128, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(128, 128)) for _ in range(8)],
        dtype=np.uint8
    )

    seq = iaa.BlendAlphaSimplexNoise(
        foreground=iaa.EdgeDetect(1.0),
        per_channel=True
    )

    images_aug = seq(images=images)

    # ------------

    save(
        "alpha",
        "alpha_simplex_example_per_channel.jpg",
        grid(images_aug, cols=4, rows=2)
    )

    # -----------------------------------------
    # noise masks
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    seed = 1
    ia.seed(seed)

    seq = iaa.BlendAlphaSimplexNoise(
        foreground=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)
    )

    masks = [
        seq.factor.draw_samples(
            (64, 64), random_state=ia.new_random_state(seed+1+i)
        ) for i in range(16)]
    masks = np.hstack(masks)
    masks = np.tile(masks[:, :, np.newaxis], (1, 1, 1, 3))
    masks = (masks * 255).astype(np.uint8)

    # ------------

    save(
        "alpha",
        "alpha_simplex_noise_masks.jpg",
        grid(masks, cols=16, rows=1)
    )

    # -----------------------------------------
    # noise masks, upscale=nearest
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    seed = 1
    ia.seed(seed)

    seq = iaa.SimplexNoiseAlpha(
        first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
        upscale_method="nearest"
    )

    masks = [
        seq.factor.draw_samples(
            (64, 64), random_state=ia.new_random_state(seed+1+i)
        ) for i in range(16)]
    masks = np.hstack(masks)
    masks = np.tile(masks[:, :, np.newaxis], (1, 1, 1, 3))
    masks = (masks * 255).astype(np.uint8)

    # ------------

    save(
        "alpha",
        "alpha_simplex_noise_masks_nearest.jpg",
        grid(masks, cols=16, rows=1)
    )

    # -----------------------------------------
    # noise masks linear
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    seed = 1
    ia.seed(seed)

    seq = iaa.BlendAlphaSimplexNoise(
        foreground=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
        upscale_method="linear"
    )

    masks = [
        seq.factor.draw_samples(
            (64, 64), random_state=ia.new_random_state(seed+1+i)
        ) for i in range(16)]
    masks = np.hstack(masks)
    masks = np.tile(masks[:, :, np.newaxis], (1, 1, 1, 3))
    masks = (masks * 255).astype(np.uint8)

    # ------------

    save(
        "alpha",
        "alpha_simplex_noise_masks_linear.jpg",
        grid(masks, cols=16, rows=1)
    )


def chapter_alpha_masks_frequency():
    # -----------------------------------------
    # example 1 (basic)
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap

    ia.seed(1)

    # Example batch of images.
    # The array has shape (8, 64, 64, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(128, 128)) for _ in range(8)],
        dtype=np.uint8
    )

    seq = iaa.BlendAlphaFrequencyNoise(
        first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)
    )

    images_aug = seq(images=images)

    # ------------

    save(
        "alpha",
        "alpha_frequency_example_basic.jpg",
        grid(images_aug, cols=4, rows=2)
    )

    # -----------------------------------------
    # example 1 (per_channel)
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa

    ia.seed(1)

    # Example batch of images.
    # The array has shape (8, 128, 128, 3) and dtype uint8.
    images = np.array(
        [ia.quokka(size=(128, 128)) for _ in range(8)],
        dtype=np.uint8
    )

    seq = iaa.BlendAlphaFrequencyNoise(
        foreground=iaa.EdgeDetect(1.0),
        per_channel=True
    )

    images_aug = seq(images=images)

    # ------------

    save(
        "alpha",
        "alpha_frequency_example_per_channel.jpg",
        grid(images_aug, cols=4, rows=2)
    )

    # -----------------------------------------
    # noise masks
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap

    seed = 1
    ia.seed(seed)

    seq = iaa.BlendAlphaFrequencyNoise(
        first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)
    )

    masks = [
        seq.factor.draw_samples(
            (64, 64), random_state=ia.new_random_state(seed+1+i)
        ) for i in range(16)]
    masks = [np.tile(mask[:, :, np.newaxis], (1, 1, 3)) for mask in masks]
    masks = [(mask * 255).astype(np.uint8) for mask in masks]

    # ------------

    save(
        "alpha",
        "alpha_frequency_noise_masks.jpg",
        grid(masks, cols=8, rows=2)
    )

    # -----------------------------------------
    # noise masks, varying exponent
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap

    seed = 1
    ia.seed(seed)

    masks = []
    nb_rows = 4
    exponents = np.linspace(-4.0, 4.0, 16)

    for i, exponent in enumerate(exponents):
        seq = iaa.BlendAlphaFrequencyNoise(
            exponent=exponent,
            foreground=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
            size_px_max=32,
            upscale_method="linear",
            iterations=1,
            sigmoid=False
        )

        group = []
        for row in range(nb_rows):
            mask = seq.factor.draw_samples(
                (64, 64), random_state=ia.new_random_state(seed+1+i*10+row))
            mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))
            mask = (mask * 255).astype(np.uint8)
            if row == nb_rows - 1:
                mask = np.pad(
                    mask,
                    ((0, 20), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=255)
                mask = ia.draw_text(
                    mask,
                    y=64+2,
                    x=6,
                    text="%.2f" % (exponent,),
                    size=10,
                    color=[0, 0, 0])
            group.append(mask)
        masks.append(np.vstack(group))

    # ------------

    save(
        "alpha",
        "alpha_frequency_noise_masks_exponents.jpg",
        grid(masks, cols=16, rows=1)
    )

    # -----------------------------------------
    # noise masks, upscale=nearest
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap

    seed = 1
    ia.seed(seed)

    seq = iaa.BlendAlphaFrequencyNoise(
        first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
        upscale_method="nearest"
    )

    masks = [
        seq.factor.draw_samples(
            (64, 64), random_state=ia.new_random_state(seed+1+i)
        ) for i in range(16)]
    masks = [np.tile(mask[:, :, np.newaxis], (1, 1, 3)) for mask in masks]
    masks = [(mask * 255).astype(np.uint8) for mask in masks]

    # ------------

    save(
        "alpha",
        "alpha_frequency_noise_masks_nearest.jpg",
        grid(masks, cols=8, rows=2)
    )

    # -----------------------------------------
    # noise masks linear
    # -----------------------------------------
    import imgaug as ia
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap

    seed = 1
    ia.seed(seed)

    seq = iaa.BlendAlphaFrequencyNoise(
        first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True),
        upscale_method="linear"
    )

    masks = [
        seq.factor.draw_samples(
            (64, 64), random_state=ia.new_random_state(seed+1+i)
        ) for i in range(16)]
    masks = [np.tile(mask[:, :, np.newaxis], (1, 1, 3)) for mask in masks]
    masks = [(mask * 255).astype(np.uint8) for mask in masks]

    # ------------

    save(
        "alpha",
        "alpha_frequency_noise_masks_linear.jpg",
        grid(masks, cols=8, rows=2)
    )


def chapter_alpha_masks_iterative():
    # -----------------------------------------
    # IterativeNoiseAggregator varying number of iterations
    # -----------------------------------------
    import imgaug as ia
    from imgaug import parameters as iap

    seed = 1
    ia.seed(seed)

    masks = []
    iterations_all = [1, 2, 3, 4]

    for iterations in iterations_all:
        noise = iap.IterativeNoiseAggregator(
            other_param=iap.FrequencyNoise(
                exponent=(-4.0, 4.0),
                upscale_method=["linear", "nearest"]
            ),
            iterations=iterations,
            aggregation_method="max"
        )

        row = [
            noise.draw_samples(
                (64, 64), random_state=ia.new_random_state(seed+1+i)
            ) for i in range(16)]
        row = np.hstack(row)
        row = np.tile(row[:, :, np.newaxis], (1, 1, 3))
        row = (row * 255).astype(np.uint8)
        row = np.pad(
            row,
            ((0, 0), (50, 0), (0, 0)),
            mode="constant",
            constant_values=255)
        row = ia.draw_text(
            row,
            y=24,
            x=2,
            text="%d iter." % (iterations,),
            size=14,
            color=[0, 0, 0])
        masks.append(row)

    # ------------

    save(
        "alpha",
        "iterative_vary_iterations.jpg",
        grid(masks, cols=1, rows=len(iterations_all))
    )

    # -----------------------------------------
    # IterativeNoiseAggregator varying methods
    # -----------------------------------------
    import imgaug as ia
    from imgaug import parameters as iap

    seed = 1
    ia.seed(seed)

    iterations_all = [1, 2, 3, 4, 5, 6]
    methods = ["min", "avg", "max"]
    cell_idx = 0
    rows = []

    for method_idx, method in enumerate(methods):
        row = []
        for iterations in iterations_all:
            noise = iap.IterativeNoiseAggregator(
                other_param=iap.FrequencyNoise(
                    exponent=-2.0,
                    size_px_max=32,
                    upscale_method=["linear", "nearest"]
                ),
                iterations=iterations,
                aggregation_method=method
            )

            cell = noise.draw_samples(
                (64, 64), random_state=ia.new_random_state(seed+1+method_idx))
            cell = np.tile(cell[:, :, np.newaxis], (1, 1, 3))
            cell = (cell * 255).astype(np.uint8)

            if iterations == 1:
                cell = np.pad(
                    cell,
                    ((0, 0), (40, 0), (0, 0)),
                    mode="constant",
                    constant_values=255)
                cell = ia.draw_text(
                    cell,
                    y=27,
                    x=2,
                    text="%s" % (method,),
                    size=14,
                    color=[0, 0, 0])
            if method_idx == 0:
                cell = np.pad(
                    cell,
                    ((20, 0), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=255)
                cell = ia.draw_text(
                    cell,
                    y=0,
                    x=12+40*(iterations == 1),
                    text="%d iter." % (iterations,),
                    size=14,
                    color=[0, 0, 0])
            cell = np.pad(
                cell,
                ((0, 1), (0, 1), (0, 0)),
                mode="constant",
                constant_values=255)

            row.append(cell)
            cell_idx += 1
        rows.append(np.hstack(row))
    gridarr = np.vstack(rows)

    # ------------

    save(
        "alpha",
        "iterative_vary_methods.jpg",
        gridarr
    )


def chapter_alpha_masks_sigmoid():
    # -----------------------------------------
    # Sigmoid varying on/off
    # -----------------------------------------
    import imgaug as ia
    from imgaug import parameters as iap

    seed = 1
    ia.seed(seed)

    masks = []

    for activated in [False, True]:
        noise = iap.Sigmoid.create_for_noise(
            other_param=iap.FrequencyNoise(
                exponent=(-4.0, 4.0),
                upscale_method="linear"
            ),
            activated=activated
        )

        row = [
            noise.draw_samples(
                (64, 64),
                random_state=ia.new_random_state(seed+1+i))
            for i in range(16)]
        row = np.hstack(row)
        row = np.tile(row[:, :, np.newaxis], (1, 1, 3))
        row = (row * 255).astype(np.uint8)
        row = np.pad(
            row,
            ((0, 0), (90, 0), (0, 0)),
            mode="constant",
            constant_values=255)
        row = ia.draw_text(
            row,
            y=17,
            x=2,
            text="activated=\n%s" % (activated,),
            size=14,
            color=[0, 0, 0])
        masks.append(row)

    # ------------

    save(
        "alpha",
        "sigmoid_vary_activated.jpg",
        grid(masks, cols=1, rows=2)
    )

    # -----------------------------------------
    # Sigmoid varying on/off
    # -----------------------------------------
    import imgaug as ia
    from imgaug import parameters as iap

    seed = 1
    ia.seed(seed)

    masks = []
    nb_rows = 3

    class ConstantNoise(iap.StochasticParameter):
        def __init__(self, noise, seed):
            super(ConstantNoise, self).__init__()
            self.noise = noise
            self.seed = seed

        def _draw_samples(self, size, random_state):
            return self.noise.draw_samples(
                size, random_state=ia.new_random_state(self.seed))

    for rowidx in range(nb_rows):
        row = []
        for tidx, threshold in enumerate(np.linspace(-10.0, 10.0, 10)):
            noise = iap.Sigmoid.create_for_noise(
                other_param=ConstantNoise(
                    iap.FrequencyNoise(
                        exponent=(-4.0, 4.0),
                        upscale_method="linear"
                    ),
                    seed=seed+100+rowidx
                ),
                activated=True,
                threshold=threshold
            )

            cell = noise.draw_samples(
                (64, 64),
                random_state=ia.new_random_state(seed+tidx))
            cell = np.tile(cell[:, :, np.newaxis], (1, 1, 3))
            cell = (cell * 255).astype(np.uint8)
            if rowidx == 0:
                cell = np.pad(
                    cell,
                    ((20, 0), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=255)
                cell = ia.draw_text(
                    cell,
                    y=2,
                    x=15,
                    text="%.1f" % (threshold,),
                    size=14,
                    color=[0, 0, 0])
            row.append(cell)
        row = np.hstack(row)
        masks.append(row)

    gridarr = np.vstack(masks)
    # ------------

    save(
        "alpha",
        "sigmoid_vary_threshold.jpg",
        gridarr
    )


if __name__ == "__main__":
    main()
