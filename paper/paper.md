---
title: 'teex: a Toolbox for the Evaluation of Explanations'
tags:
  - explainable AI
  - responsible AI
  - explanation evaluation
  - Python
authors:
  - name: Jesús M. Antoñanzas
    orcid: 0000-0001-8781-4338
    corresponding: true
    affiliation: "1, 2"
  - name: Yunzhe Jia
    orcid: 0000-0001-7376-5838
    affiliation: 1
  - name: Eibe Frank
    orcid: 0000-0001-6152-7111
    affiliation: 4  
  - name: Albert Bifet
    orcid: 0000-0002-8339-7773
    affiliation: "1, 3"
  - name: Bernhard Pfahringer
    orcid: 0000-0002-3732-5787
    affiliation: 4
affiliations:
 - name: AI Institute, University of Waikato, Hamilton, New Zealand
   index: 1
- name: Department of Physics, Universitat Politècnica de Catalunya, Barcelona, Spain
   index: 2
 - name: LTCI, Télecom Paris, Institut Polytechnique de Paris, Palaiseau, France
   index: 3
 - name: Department of Computer Science and AI Institute, University of Waikato, Hamilton, New Zealand
   index: 4
date: 10 November 2022
bibliography: paper.bib
---

# Summary

The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration).

# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Acknowledgements

`teex` has been developed as part of the TAIAO project (Time-Evolving Data Science / Artificial Intelligence for Advanced Open Environmental Science), funded by the New Zealand Ministry of Business, Innovation, and Employment (MBIE).

# References