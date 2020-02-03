from __future__ import print_function, division

import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap

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
    aug = getattr(iaa.imgcorruptlike, augname)(
        severity=_CallwiseDeterministicList([1, 1, 1,
                                             2, 2, 2,
                                             3, 3, 3,
                                             4, 4, 4,
                                             5, 5, 5]))
    run_and_save_augseq(
        "imgcorruptlike/%s.jpg" % (augname.lower(),), aug,
        [ia.quokka(size=(128, 128)) for _ in range(5*3)], cols=3, rows=5
    )


class _CallwiseDeterministicList(iap.StochasticParameter):
    def __init__(self, values):
        super(_CallwiseDeterministicList, self).__init__()
        self.values = np.array(values)
        self.call_idx = 0

    def _draw_samples(self, size, random_state):
        value = self.values[self.call_idx]
        self.call_idx += 1
        return np.full(size, value, dtype=np.int32)


if __name__ == "__main__":
    main()
