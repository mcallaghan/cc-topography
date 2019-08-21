# A Topography of Climate Change Research

This repository contains the code for the paper *A Topography of Climate Change Research*

It depends heavily on https://github.com/mcallaghan/tmv, which is an extension to a platform to view topic models in a web browser written by Allison J.B Chaney. It has been extended into a framework for managing collections of scientific documents and topic models.

Model runs are created using the `do_nmf` function of that framework, which runs topic models on collections of documents. The script in this repository at `code/run_models.py` sets up these model runs.

The results of the topic model are analysed with the scripts and notebooks contained in this repository's `code` folder, where the plots accompanying the paper are created.
