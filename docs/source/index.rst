.. MLGWSC_proposal documentation master file, created by
   sphinx-quickstart on Wed Sep 23 09:31:02 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``MLGWSC_proposal`` documentation
=================================

The ``MLGWSC_proposal`` package implements a Deep Learning pipeline designed for the **1st first
Machine Learning Gravitational-Wave Search mock data Challenge (MLGWSC-1)** [Sch2022]_. This project was
designed for the *Computing Methods for Experimental Physiscs and Data Analysis* cuorse for the
Master's Degree in Physisc at the University of Pisa.

This repository provides all the software needed to define, train and evaluate ``GW_ResNet152``,
a 1D ResNet152-based model [He2016]_ for binary classification of strain data, distinguishing data
that only contains background noise and data with actual GWs signals.
Specifically, during implementation the model was trained on the
datasets provided by the MLGWSC-1 at [MLGWSCRepo]_, which resemble strain timeseries from Ligo Livingston
and Ligo Hanford interferometers with progressively more realistic noise backgrounds.

The model itself is meant to be a deeper and non-professional version of the architecture presented in [Nou2023]_,
and it in fact trained on similar data and evaluated with the same methods.

.. toctree::
   :maxdepth: 2
   :caption: Table of contents:

   installation
   api
   bib
