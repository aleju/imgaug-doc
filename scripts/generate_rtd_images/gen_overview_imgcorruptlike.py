from __future__ import print_function, division

import imgaug as ia
import imgaug.augmenters as iaa

from .utils import run_and_save_augseq


def main():
    # all examples have the same schema, which we can exploit here
    augnames = [
        "GaussianNoise",
        "ShotNoise",
        "ImpulseNoise",
        "SpeckleNoise",
        "GaussianBlur",
        "GlassBlur",
        "DefocusBlur",
        "MotionBlur",
        "ZoomBlur",
        "Fog",
        "Frost",
        "Snow",
        "Spatter",
        "Contrast",
        "Brightness",
        "Saturate",
        "JpegCompression",
        "Pixelate",
        "ElasticTransform"
    ]
    for augname in augnames:
        generate_generic_chapter(augname)


def generate_generic_chapter(augname):
    aug = getattr(iaa.imgcorruptlike, augname)(severity=2)
    run_and_save_augseq(
        "imgcorruptlike/%s.jpg" % (augname.lower(),), aug,
        [ia.quokka(size=(64, 64)) for _ in range(8*1)], cols=8, rows=1
    )


if __name__ == "__main__":
    main()
