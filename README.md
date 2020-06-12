# Intro

Product recommendation is a filtering system that recommends products that might interest the user and may eventually lead to a user purchasing the product. 
Recommendation engines at their core are information filtering tools that use different algorithms and data to recommend relevant products to users. 
These systems if configured properly can significantly boost revenues. 
Basic Idea behind Collaborative filtering: This filtering method is usually based on collecting and analyzing information on user's behaviors, their activities or preferences, and predicting what they will like based on the similarity with other users.
Collaborative filtering is based on the assumption that people who agreed in the past will agree in the future, and that they will like similar kinds of items as they liked in the past.

In this project, I will be implementing Collaborative filtering, both user based and item based using the following approaches:
- Memory Based
- Model Based
-Matrix Factorisation
   
## Working on this project

- Always work on a branch and then send a pull request.
- Review outstanding PRs. If you do not feel comfortable merging PRs, comment with a "+1" to signal your co-collaborators that it's passed your review.
- Feel free to push directly to master only for micro-fixes such as typos.

## Directory structure
Adapted from [here](https://github.com/drivendata/cookiecutter-data-science). Let's modify this structure as we add files. Can delete once we have a working pipeline.
```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- CSV files.
│   └── raw            <- The original, immutable data dump.
│
├── graphs             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── eda.py
|   |   └── data_cleaning.py... 
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│ 
```


