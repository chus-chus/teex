.. teex documentation master file, created by
   sphinx-quickstart on Tue Jun 29 12:41:10 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: images/teex_logo__.png
   :width: 115
   :align: center
   :alt: Our AI generated logo. Comes from the prompt: 
      'logo of a t, inspired by an AI that is fair and responsible.


teex
================================

A Python Toolbox for the evaluation of machine learning explanations.

This project aims to provide a simple way of evaluating individual black box
explanations. Moreover, it contains a collection of easy-to-access datasets with available ground truth explanations.

**teex** contains a subpackage for each explanation type, and each subpackage contains two modules:

   - ``eval``: with methods for explanation evaluation.
   - ``data``: with methods for data generation, loading and manipulation.

Visit our `GitHub <https://github.com/chus-chus/teex>`_ for source.

.. toctree:: 
   :maxdepth: 2
   :caption: Examples

   demos/examples

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/modules
