***********************
augmenters.segmentation
***********************

Superpixels
-----------

Completely or partially transform images to their superpixel representation.

Generate about 64 superpixels per image. Replace each one with a probability
of 50% by its average pixel color. ::

    aug = iaa.Superpixels(p_replace=0.5, n_segments=64)

.. figure:: ../../images/overview_of_augmenters/segmentation/superpixels_50_64.jpg
    :alt: Superpixels

Generate 16 to 128 superpixels per image. Replace each superpixel with a
probability between 10 and 100% (sampled once per image) by its average pixel
color. ::

    aug = iaa.Superpixels(p_replace=(0.1, 1.0), n_segments=(16, 128))

.. figure:: ../../images/overview_of_augmenters/segmentation/superpixels.jpg
    :alt: Superpixels random

Effect of setting ``n_segments`` to a fixed value of 64 and then
increasing ``p_replace`` from 0.0 and 1.0:

.. figure:: ../../images/overview_of_augmenters/segmentation/superpixels_vary_p.jpg
    :alt: Superpixels varying p

Effect of setting ``p_replace`` to a fixed value of 1.0 and then
increasing ``n_segments`` from 1\*16 to 9\*16=144:

.. figure:: ../../images/overview_of_augmenters/segmentation/superpixels_vary_n.jpg
    :alt: Superpixels varying n


Voronoi
-------

TODO


UniformVoronoi
--------------

TODO


RegularGridVoronoi
------------------

TODO


RelativeRegularGridVoronoi
--------------------------

TODO

