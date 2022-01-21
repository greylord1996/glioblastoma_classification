# glioblastoma_classification

### Silver medal solution (66th place over 1550 teams) for https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/overview
## Installation

First of all, you should have python 3.x to work with this project. The recommended Python version is 3.6 or greater.

Note for Windows users: You should start a command line with administrator's privileges.

First of all, clone the repository:

    git clone https://github.com/greylord1996/glioblastoma_classification.git
    cd glioblastoma_classification/

Create a new virtual environment:

    # on Linux:
    python -m venv gliomavenv
    # on Windows:
    python -m venv gliomavenv

Activate the environment:

    # on Linux:
    source gliomavenv/bin/activate
    # on Windows:
    call gliomavenv\Scripts\activate.bat

Install required dependencies:

    # on Linux:
    pip install -r requirements.txt
    # on Windows:
    python -m pip install -r requirements.txt


## Data

To use this code you need to download data and specify paths on it:

- Kaggle input data: https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/data
- Public Kernel with trained weights: https://www.kaggle.com/greylord1996/resnet34-all-mri
