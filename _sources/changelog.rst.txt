Changelog
===============================================================================

.. toctree::
   :maxdepth: 1

In this section you'll find information about what's new in the newer
releases of L5Kit.

Release v.1.4.0 `(18 Aug 2021)`
-------------------------------------------------------------------------------
    * Added L5Kit gym-compatible environment;
    * Added unicycle kinematic model;
    * Added metric sets for closed loop evaluation;
    * Updated the structure and style of L5Kit documentation;
    * Small refactoring of closed loop simulator;

Release v.1.3.1 `(8 Jun 2021)`
-------------------------------------------------------------------------------
    * Hot-fix to add the visualizer package in the release wheel;
    * Add init to visualizer package;
    * Small fixes in notebooks;

Release v.1.3.0 `(19 Apr 2021)`
-------------------------------------------------------------------------------
    * Remove black formatter;
    * Add ackerman exact for perturbations, skip applying when the agent is static;
    * Add flag to enable/disable ego history rasterization;
    * Add simulation metrics suite;
    * Add simulation loop;
    * Add new visualizer for improved visualisation based on Bokeh;
    * Add notebooks for simulation;

Release v.1.2.0 `(13 Jan 2021)`
-------------------------------------------------------------------------------
    * Add an interface for matrix batch transformation;
    * Speed up rasterisation;
    * Improve documentation;
    * Add flip in the rasteriser context (but keep a flag for compatibility);
    * Make agent_sampling modular;
    * Improve MapAPI interface and add lanes interpolation;
    * SatelliteRasteriser for rectangular-shaped has been fixed (thanks pascal-pfeiffer!);
    * Speed up get_frame_indices call;

Release v.1.1.0 `(29 Sep 2020)`
-------------------------------------------------------------------------------
    * :code:`target_positions` are now in :code:`agent` space. See :ref:`Coordinate Systems <coordinate>` and notebooks for examples on how to use it and its relation with the other spaces. This fixes a misalignment which is affecting the competition;
    * Ego and AgentDataset now returns more matrices to convert between spaces;
    * Transformations Matrices notation has changed from "A_to_B" to "B_from_A";
    * BoxRasterizer now support subpixel precision;
    * Fix an issue with swapped width and height which was preventing to use rectangular-shaped inputs;
    * Introduce Pipenv for deterministic builds;
    * Add option to disable traffic lights;
    * Improve trajectories visualisation params;

Release v.1.0.6 `(24 Aug 2020)`
-------------------------------------------------------------------------------
    * Add support for multiple metrics;
    * Improve documentation;
    * Improve API for zarr interaction;
    * Bug fixes;
    * Refactor config files;
    * Provide mock-test for competition in prediction notebook;

Release v.1.0.5 `(6 Aug 2020)`
-------------------------------------------------------------------------------
    * Add first traffic lights implementation;
    * Speed up semantic rasterisation;
    * Introduce MapAPI to handle protobuf;
    * Refactor agents_mask to also include static agents;
    * Improve tests suite;

Release v.1.0.3 `(8 Jul 2020)`
-------------------------------------------------------------------------------
    * Drop strictyaml;
    * Enforce some deps versions;

Release v.1.0.2 `(3 Jul 2020)`
-------------------------------------------------------------------------------
    * Speed up training and inference, especially for long scenes;
    * Add better print for dataset;

Release v.1.0.0 `(22 Jun 2020)`
-------------------------------------------------------------------------------
    * First release of the project;