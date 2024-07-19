Copyright (c) Safran, 2024

All rights reserved.

This source code can only be used as Safran participation entry of the codabench competition
[MACHINE LEARNING FOR PHYSICAL SIMULATION CHALLENGE](https://www.codabench.org/competitions/1534).
The publication of the source code is a requirement of the competition for appearing in the final leaderboard.


The method used is called MMGP proposed at SafranTech, the research center of [Safran group](https://www.safran-group.com/), in the paper: Mesh Morphing Gaussian Process-based machine learning method for regression of physical problems under non-parameterized geometrical variability ([arXiv preprint](https://arxiv.org/abs/2305.12871)), accepted at NeurIPS 2023. The main difference with the paper is the use of a more involved morphing stage to address the nu_t field, not considered in the paper.

If you want to use this method, please consider using [the open-source implementation of the library mmgp](https://gitlab.com/drti/mmgp)
in BSD 3-Clause license, and citing the paper:

```bibtex
@article{mmgp2023,
      title={MMGP: a Mesh Morphing Gaussian Process-based machine learning method for regression of physical problems under non-parameterized geometrical variability},
      author={Casenave, F. and Staber, B. and Roynard, X.},
      journal={arXiv preprint arXiv:2305.12871},
      year={2023}
}
```