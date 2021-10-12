.. _planning_open_loop:

Open-loop Planning
==================

Planning is an essential piece of the Autonomous Vehicles (AV) stack.

In this series of notebooks you will train and evaluate a data-driven ML policy.

**Before starting, please download the Lyft L5 Prediction Dataset 2020 and follow the instructions to correctly organise it.**

* `Lyft L5 Prediction Dataset 2020 <https://self-driving.lyft.com/level5/prediction/>`_
* `Instructions <https://github.com/lyft/l5kit#download-the-datasets>`_

Training an effective ML policy for planning can take several hours on the best performing hardware.
For this reason, we provide trained models you can experiment with in our evaluations notebooks,
without requiring to train one yourself. Still, we suggest you to go through the training notebook we provide,
as it contains key insights on how to effectively train an ML policy for planning.

Notebook Tutorial
-----------------

We provide 3 notebooks for a deep dive into planning for a Self Driving Vehicle (SDV).

Training Notebook
~~~~~~~~~~~~~~~~~

You can train your first ML policy for planning using our `training notebook <https://github.com/lyft/l5kit/blob/master/examples/planning/train.ipynb>`_

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/lyft/l5kit/blob/master/examples/planning/train.ipynb
   :alt: Open In Colab

In this notebook you are going to train your own ML policy to fully control a Self Driving Vehicle (SDV). You will train your model using the Lyft Prediction Dataset and L5Kit.

The policy will be a deep neural network (DNN) which will be invoked by the SDV to obtain the next command to execute.

More in details, you will be working with a CNN architecture based on ResNet50.

.. image:: images/planning/model.svg
   :alt: model

Inputs
++++++

The network will get a BEV of the scene surrounding the SDV as the only input. This has been rasterised in a fixed grid image to comply with the CNN input. L5Kit is shipped with various rasterisers. Each one of them captures different aspects of the scene (e.g. lanes or satellite view).

This input representation is very similar to the one used in the `prediction competition <https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/overview>`_. Please refer to our `competition baseline notebook <https://github.com/lyft/l5kit/blob/master/examples/agent_motion_prediction/agent_motion_prediction.ipynb>`_
and our `data format notebook <https://github.com/lyft/l5kit/blob/master/examples/visualisation/visualise_data.ipynb>`_ if you want to learn more about it.

Outputs
+++++++

The network outputs the driving signals required to fully control the SDV. In particular, this is a trajectory of XY and yaw displacements which can be used to move and steer the vehicle.

After enough training, your model will be able to drive the SDV along a specific route. Among others, it will do lane-following while respecting traffic lights.


Pre-Trained Models
++++++++++++++++++

We provide a collection of pre-trained models you can experiment with and use in your own experiments.
Please refer to the training notebook for additional details and download links.


Open-Loop Evaluation Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can evaluate your model in the open-loop setting using our `open-loop evaluation notebook <https://github.com/lyft/l5kit/blob/master/examples/planning/open_loop_test.ipynb>`_

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/lyft/l5kit/blob/master/examples/planning/open_loop_test.ipynb
   :alt: Open In Colab


What is open-loop evaluation?
+++++++++++++++++++++++++++++

In open-loop evaluation we evaluate our model prediction as we follow the annotated ground truth.

In each frame, we compare the predictions of our model against the annotated ground truth. This can be done with different metrics.

**Regardless of the metric used, this evaluation protocol doesn't modify the future locations according to the model's predictions.**

.. image:: images/planning/open-loop.svg
   :alt: open-loop


Closed-Loop Evaluation Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can evaluate your model in the closed-loop setting using our `closed-loop evaluation notebook <https://github.com/lyft/l5kit/blob/master/examples/planning/closed_loop_test.ipynb>`_

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/lyft/l5kit/blob/master/examples/planning/closed_loop_test.ipynb
   :alt: Open In Colab

In this notebook you are going to evaluate a CNN-based policy to control the SDV with a protocol named *closed-loop* evaluation.

What is closed-loop evaluation?
+++++++++++++++++++++++++++++++

In closed-loop evaluation the model is in **full control of the SDV**. At each time step, we predict the future trajectory and then move the AV to the first of the model's predictions.

We refer to this process with the terms **forward-simulate** or **unroll**.

.. image:: images/planning/closed-loop.svg
   :alt: closed-loop


Video Tutorial
--------------

Check out this short tutorial that describes the task of planning for autonomous driving

.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube.com/embed/Jygsh17QbxY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>