# Propability Functions

In this challenge we will practice our knowledge in probability and statistics,
fundamental knowledge for any data scientist.

## Objective:

The objective of this challenge is to explore the main functions about probability distributions
such as PDF, CDF and quantiles and the relationships between two of the main distributions: normal and binomial.

For this, we will use artificial and real data. As real data, we’ll explore the _data set_
[Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2) provided by Dr. Robert Lyon of the University of Manchester.

This _data set_ consists of 8 variables regarding 17898 star observations. These
stars were considered "candidates" to be pulsar type stars, which have strong
importance for astrophysicists. A ninth column of the _data set_ specifies whether the star is
really a pulsar (if positive, 1) or not (negative, 0).


## Topics

In this challenge we will explore:

* Probabilidade
* Estatística
* NumPy
* SciPy
* StatsModels

## Requirements

You will need Python 3 and pip. It is highly recommended to use virtual environments
with virtualenv and the `requirements.txt` file to install the dependency packages
challenge:

```bash
$ pip3 install virtualenv
$ virtualenv venv -p python3
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Windows

```bash
> pip3 install virtualenv
> virtualenv ..\venv -p python3
> ..\venv\Scripts\activate
> pip install -r requirements.txt
```

When finished, you can disable the virtualenv virtual environment with:

```bash
$ deactivate
```
