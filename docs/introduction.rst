.. _intro:

Introduction
============

This repository and the associated datasets constitute a framework for developing learning-based solutions to prediction, planning and simulation problems in self-driving. State-of-the-art solutions to these problems still require significant amounts of hand-engineering and unlike, for example, perception systems, have not benefited much from deep learning and the vast amount of driving data available.

The purpose of this framework is to enable engineers and researchers to experiment with data-driven approaches to planning and simulation problems using real world driving data and contribute to state-of-the-art solutions.


.. image:: images/pipeline.png
   :width: 800
   :alt: Modern AV pipeline

You can use this framework to build systems which:

* Turn prediction, planning and simulation problems into data problems and train them on real data.
* Use neural networks to model key components of the Autonomous Vehicle (AV) stack.
* Use historical observations to predict future movement of cars around an AV.
* Plan behavior of an AV in order to imitate human driving.
* Study the improvement in performance of these systems as the amount of data increases.


This software is developed by Lyft Level 5 self-driving division and is :ref:`open to external contributors <contribute>`.


Video Tutorial
--------------

Here is a short video tour introducing the L5Kit and the functionalities of the library.

.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube.com/embed/1cfXBS0i92Q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
