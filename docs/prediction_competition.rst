.. _pred_competition:

Prediction Competition
======================

Starting 24.08.2020 we are hosting a `Kaggle competition <https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/overview>`_ about predicting future movements of other traffic participants.
This page serves as introduction point for it and gives additional information.

Scoring
-------

When taking part in the competition, you will be asked to submit predictions for a private test set (no ground truth is available),
and your solutions will be scored by Kaggle. Overall 30.000 USD as prizes are available!
As traffic scenes can contain a large amount of ambiguity and uncertainty, we encourage the submission of multi-modal predictions.
For scoring, we calculate the *negative log-likelihood* of the ground truth data given these multi-modal predictions.
Let us take a closer look at this.
Assume, ground truth positions of a sample trajectory are

.. image:: https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20x_1%2C%20%5Cldots%2C%20x_T%2C%20y_1%2C%20%5Cldots%2C%20y_T
   :alt: Equation


and we predict K hypotheses, represented by means

.. image:: https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20%5Cbar%7Bx%7D_1%5Ek%2C%20%5Cldots%2C%20%5Cbar%7Bx%7D_T%5Ek%2C%20%5Cbar%7By%7D_1%5Ek%2C%20%5Cldots%2C%20%5Cbar%7By%7D_T%5Ek
   :alt: Equation

In addition, we predict confidences c of these K hypotheses.
We assume the ground truth positions to be modelled by a mixture of multi-dimensional independent Normal distributions over time,
yielding the likelihood

.. image:: https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20p%28x_%7B1%2C%20%5Cldots%2C%20T%7D%2C%20y_%7B1%2C%20%5Cldots%2C%20T%7D%7Cc%5E%7B1%2C%20%5Cldots%2C%20K%7D%2C%20%5Cbar%7Bx%7D_%7B1%2C%20%5Cldots%2C%20T%7D%5E%7B1%2C%20%5Cldots%2C%20K%7D%2C%20%5Cbar%7By%7D_%7B1%2C%20%5Cldots%2C%20T%7D%5E%7B1%2C%20%5Cldots%2C%20K%7D%29
   :alt: Equation

.. image:: https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20%3D%20%5Csum_k%20c%5Ek%20%5Cmathcal%7BN%7D%28x_%7B1%2C%20%5Cldots%2C%20T%7D%7C%5Cbar%7Bx%7D_%7B1%2C%20%5Cldots%2C%20T%7D%5E%7Bk%7D%2C%20%5CSigma%3D1%29%20%5Cmathcal%7BN%7D%28y_%7B1%2C%20%5Cldots%2C%20T%7D%7C%5Cbar%7By%7D_%7B1%2C%20%5Cldots%2C%20T%7D%5E%7Bk%7D%2C%20%5CSigma%3D1%29
   :alt: Equation

.. image:: https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20%3D%20%5Csum_k%20c%5Ek%20%5Cprod_t%20%5Cmathcal%7BN%7D%28x_t%7C%5Cbar%7Bx%7D_t%5Ek%2C%20%5Csigma%3D1%29%20%5Cmathcal%7BN%7D%28y_t%7C%5Cbar%7By%7D_t%5Ek%2C%20%5Csigma%3D1%29
   :alt: Equation

yielding the loss

.. image:: https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20L%20%3D%20-%20%5Clog%20p%28x_%7B1%2C%20%5Cldots%2C%20T%7D%2C%20y_%7B1%2C%20%5Cldots%2C%20T%7D%7Cc%5E%7B1%2C%20%5Cldots%2C%20K%7D%2C%20%5Cbar%7Bx%7D_%7B1%2C%20%5Cldots%2C%20T%7D%5E%7B1%2C%20%5Cldots%2C%20K%7D%2C%20%5Cbar%7By%7D_%7B1%2C%20%5Cldots%2C%20T%7D%5E%7B1%2C%20%5Cldots%2C%20K%7D%29
   :alt: Equation

.. image:: https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20%3D%20-%20%5Clog%20%5Csum_k%20e%5E%7B%5Clog%28c%5Ek%29%20&plus;%20%5Csum_t%20%5Clog%20%5Cmathcal%7BN%7D%28x_t%7C%5Cbar%7Bx%7D_t%5Ek%2C%20%5Csigma%3D1%29%20%5Cmathcal%7BN%7D%28y_t%7C%5Cbar%7By%7D_t%5Ek%2C%20%5Csigma%3D1%29%7D
   :alt: Equation

.. image:: https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20%3D%20-%20%5Clog%20%5Csum_k%20e%5E%7B%5Clog%28c%5Ek%29%20-%5Cfrac%7B1%7D%7B2%7D%20%5Csum_t%20%28%5Cbar%7Bx%7D_t%5Ek%20-%20x_t%29%5E2%20&plus;%20%28%5Cbar%7By%7D_t%5Ek%20-%20y_t%29%5E2%7D
   :alt: Equation

You can find our implementation `here <https://github.com/woven-planet/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py#L4>`_, which uses *error* as placeholder for the exponent

.. image:: https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20L%20%3D%20-%5Clog%20%5Csum_k%20e%5E%7B%5Ctexttt%7Berror%7D%7D)
   :alt: Equation

and for numeral stability further applies the `log-sum-exp trick <https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations>`_:
Assume, we need to calculate the logarithm of a sum of exponentials:

.. image:: https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20LSE%28x_1%2C%20%5Cldots%2C%20x_n%29%20%3D%20%5Clog%28e%5E%7Bx_1%7D%20&plus;%20%5Cldots%20&plus;%20e%5E%7Bx_n%7D%29
   :alt: Equation

Then, we rewrite this by substracting the maximum value x\ :sup:`*` \ from each exponent, resulting in much increased numerical stability:

.. image:: https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20LSE%28x_1%2C%20%5Cldots%2C%20x_n%29%20%3D%20x%5E*%20&plus;%20%5Clog%28e%5E%7Bx_1%20-%20x%5E%7B*%7D%7D%20&plus;%20%5Cldots%20&plus;%20e%5E%7Bx_n%20-%20x%5E%7B*%7D%7D%29
   :alt: Equation

Coordinates System for the competition
--------------------------------------

Please refer to :ref:`this doc <coordinate>` for a full description of the different coordinate systems used in L5Kit.

The ground truth coordinates for the competition are stored as positional displacements in the :ref:`world coordinate system <world_coordinate>`.
However, you will likely predict relative displacements for the agent of interest either in the :ref:`agent coordinate system <agent_coordinate>`
or in the :ref:`image coordinate system <image_coordinate>`.
Before using `our utils <https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/evaluation/csv_utils.py>`_ to write a CSV file for you predictions convert them
into the world coordinate system using the appropriate transformation matrix available as part of :ref:`the input data <data_abstraction>` and subtract the centroid.

Yaw is not required/used for this competition.

Additional Metrics
------------------

Scoring multi-modal prediction models is a highly complex task, and while we chose the metric described above due to its elegance and support for multi-modality,
we encourage participants to also employ other metrics for assessing their models.
Examples of such other metrics, commonly used in literature, are *Average Displacement Error* (ADE) and *Final Displacement Error* (FDE) (see 
`our dataset paper <https://arxiv.org/pdf/2006.14480.pdf>`_ or `SophieGAN <https://arxiv.org/pdf/1806.01482.pdf>`_):
ADE is the average displacement error (L2 distance between prediction and ground truth averaged over all timesteps), while FDE 
reports the final displacement error (L2 distance between prediction and ground truth, evaluated only at the last timestep).
As we consider multiple predictions, we offer `implementations for both these metrics <https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/evaluation/metrics.py>`_ either averaging over all hypotheses
or using the best hypothesis (oracle variant) - ignoring generated confidence scores in both cases.