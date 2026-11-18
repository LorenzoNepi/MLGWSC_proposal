.. MLGWSC_proposal documentation master file, created by
   sphinx-quickstart on Wed Sep 23 09:31:02 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``MLGWSC_proposal`` documentation
=================================

The ``MLGWSC_proposal`` package implements a Deep Learning pipeline designed for the **1st
Machine Learning Gravitational-Wave Search Mock Data Challenge (MLGWSC-1)** [Sc22]_. This project was
designed for the *Computing Methods for Experimental Physics and Data Analysis* course for the
Master's Degree in Physics at the University of Pisa.

This repository provides all the software needed to define, train, and evaluate various 1D ResNet-like
architectures [He16]_ of virtually any depth and use them for binary classification of strain timeseries segments,
distinguishing between data that only contains background noise and data with actual GW signals.
Specifically, during implementation, the model was trained on the datasets provided by the MLGWSC-1 organizers
at [MLGWSCRepo]_, which resemble strain timeseries from the LIGO Livingston and LIGO Hanford interferometers
with progressively more realistic noise backgrounds.

The model itself is meant to be a generalization (even though the ML model is a little simplified)
of the work presented in [No23]_, and it is, in fact, trained on similar data and evaluated with the same methods.

.. toctree::
   :maxdepth: 2
   :caption: Table of contents:

   installation
   api
   ai_help
   bib