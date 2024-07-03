# Starting Kit - Machine Learning for Physical Simulation Challenge
This starting kit provides a set of jupyter notebooks helping the challenge participants to better understand the use case, the dataset and how to contribute to this competition. For general information concerning the challenge and submit your solutions, you can refer to the competition [Codabench page](https://www.codabench.org/competitions/1534/).

Prerequisites
--------------
Most of the notebooks provided in this repository are based on LIPS platform. To be able to execute the jupyter notebooks provided in this repository and described in the following section, the [LIPS platform](https://lips.irt-systemx.fr/) should be installed properly. The installation procedure is explained in the [LIPS package repository](https://github.com/IRT-SystemX/LIPS), in [this section](https://github.com/IRT-SystemX/LIPS#installation).

To get familiar with LIPS platform itself, it includes its own set of [jupyter notebooks](https://github.com/IRT-SystemX/LIPS/tree/main/getting_started). For this competition, the participants may focus on [these notebooks](https://github.com/IRT-SystemX/LIPS/tree/main/getting_started/AirfoilUsecase) (focusing on Airfoil use case) provided in LIPS package.  

Notebooks description
---------------------
In the following, we describe the content of the jupyter notebooks : 

- **0_Basic_Competition_Information**: This notebook contains general information concerning the competition organization, phases, deadlines and terms. The content is the same as the one shared in the competition Codabench page. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRT-SystemX/ml4physim_startingkit/blob/main/0_Basic_Competition_Information.ipynb) 

- **1-Airfoil_design_basic_simulation**: This notebook aims to familiarize the participants with the use case and to facilitate their comprehension. It allows the visualization of some simulation results. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRT-SystemX/ml4physim_startingkit/blob/main/1_Airfoils_design_basic_simulation.ipynb)

- **2-Import_Airfoil_design_Dataset**: Shows how the challenge datasets could be downloaded and imported using proper functions. These data will be used in the following notebook to train and evaluate an augmented simulator. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRT-SystemX/ml4physim_startingkit/blob/main/2-Import_Airfoil_design_Dataset.ipynb) 

- **3-Reproduce_baseline_results**: This notebook shows how the baseline results could be reproduced. It includes the whole pipeline of training, evaluation and score calculation of an augmented simulator using [LIPS platform](https://github.com/IRT-SystemX/LIPS). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRT-SystemX/ml4physim_startingkit/blob/main/3_Reproduce_baseline_result.ipynb) 

- **4-How_to_Contribute**: This notebook shows 3 ways of contribution for beginner, intermediate and advanced users. The submissions should respect one of these forms to be valid and also to enable their proper evaluation through the LIPS platform which will be used for the final evaluation of the results. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRT-SystemX/ml4physim_startingkit/blob/main/4_How_to_contribute.ipynb)

       * Beginner Contributor: You only have to calibrate the parameters of existing augmented simulators
       * Intermediate Contributor: You can implement an augmented simulator respecting a given template (provided by the LIPS platform)
       * Advanced Contributor: you can implement your architecture independently from LIPS platform and use only the evaluation part of the framework to assess your model performance.

- **4a-How_to_Contribute_Tensorflow**: This notebook shows how to contribute using the existing augmented simulators based on Tensorflow library. The procedure to customize the architecture is fairly the same as pytorch (shown in Notebook 4).

- **5-Scoring**: This notebook shows firstly how the score is computed by describing its different components. Next, it provides a script which can be used locally by the participants to obtain a score for their contributions. We encourage participants to evaluate their solutions via codabench (which uses the same scoring module as the one described in this notebook). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRT-SystemX/ml4physim_startingkit/blob/main/5_Scoring.ipynb)

- **6-Submission:** This notebook presents the composition of a submission bundle for [Codabench](https://www.codabench.org/competitions/1534/) and usable parameters. 

- **7-Submission_examples:** This notebook shows how to submit on [Codabench](https://www.codabench.org/competitions/1534/) and examples of submissions bundles.  



