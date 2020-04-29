#!/usr/bin/env python
# coding: utf-8

# # Codenation Challenge - Week 3
# 
# In this challenge, we will practice our knowledge of probability distributions. For this,
# we will divide this challenge into two parts:
#     
# 1. The first part will have 3 questions about an artificial * data set * with data from a normal sample and
#     a binomial.
# 2. The second part will be about the analysis of the distribution of a variable of the _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), containing 2 questions.

# ## General _setup_ 

# In[183]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# ## Part 1
# <p><p>
#     
# ### _Setup_

# In[185]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# # Defining DataStatAnalysis
# 
# This class will be used to abstract some methods, based on pandas and ECDF libraries to support all calculus needed to solve part 1 and part 2 questions.

# In[186]:


class DataStatAnalysis(object):
    
    def __init__(self, df):
        """
        Class created to abstract some pandas commands and statistics/propabilities functions.
        """
        self.df = df
        self.columns = df.columns
        self.n_index = df.shape[0]
        self.n_columns = df.shape[1]
        self.ecdf = None
        self.actual_ecdf_calibration = None
        self.mean = None
        self.variance = None
        
    def quantile_calc(self, column):
        """
        This function receives a specific column name and return a list with quantiles where:
        index 0 - 25%
        index 1 - 50%
        index 2 - 75%
        """
        quantiles = []
        
        quantiles.append(self.df[column].quantile(0.25))
        quantiles.append(self.df[column].quantile(0.50))
        quantiles.append(self.df[column].quantile(0.75))
        
        return quantiles
    
    def ecdf_calibrate(self, column):
        """
        This function calibrates the ECDF function using a dataframe column.
        """
        self.ecdf = ECDF(self.df[column])
        self.actual_ecdf_calibration = 'ECDF calibrated with {} column'.format(column)
        
    def ecdf_calc(self,interval):
        """
        This function calculates probabilies using ecdf.
        If a value is passed the function returns a single value.
        If a list or array or dataframe column is passed the function return a numpy array.
        """
        if self.ecdf is not None:
            print(self.actual_ecdf_calibration)
            return self.ecdf(interval)
        else:
            print('Calibrate ECDF function using ecdf_calibrate method')
            
    def mean_calc(self, column):
        """
        This function calculate the mean of a dataframe column just to abstract pandas methods.
        """
        return self.df[column].mean()
    
    def variance_calc(self, column):
        """
        This function calculate the variance of a dataframe column just to abstract pandas methods.
        """
        return self.df[column].var()
    
    def std_calc(self, column):
        """
        This function calculate the standard deviation of a dataframe column just to abstract pandas methods.
        """
        return self.df[column].std()


# In[187]:


# Create a instance of DataStatAnalysis class to solve problems on part 1.
stat_analysis_part1 = DataStatAnalysis(dataframe)

# Use quntile_calc method to obtain 25% 50% and 75% quntiles for normal and binomial distributions
q_norm =  stat_analysis_part1.quantile_calc('normal')
q_binom = stat_analysis_part1.quantile_calc('binomial')

"""
==============
Question - 1
==============
"""
# q_diff is a list with the difference between norm and binomial
q_diff = [q_norm[x] - q_binom[x] for x in range(len(q_norm))]

"""
==============
Question - 2
==============
"""
# Calculate the probability of interval (mean - standard deviation), (mean + standard deviation) of normal distribution

# Firt step: calibrate ECDF model with normal distribution
stat_analysis_part1.ecdf_calibrate('normal')
mean_normal = stat_analysis_part1.mean_calc('normal')
std_normal = stat_analysis_part1.std_calc('normal')

# ecdf_calc calculates the probability of a single value or each element into a list.
probabilities = stat_analysis_part1.ecdf_calc([mean_normal - std_normal, mean_normal + std_normal])

# In this case to know the probability of interval we just have to calculate (mean + std) - (mean - std)
interval_probability = probabilities[1] - probabilities[0]

"""
==============
Question - 3
==============
"""
# Calculate the difference between means and variances, comparing binomial and normal distribution
normal_mean = stat_analysis_part1.mean_calc('normal')
normal_var = stat_analysis_part1.variance_calc('normal')
binomial_mean = stat_analysis_part1.mean_calc('binomial')
binomial_var = stat_analysis_part1.variance_calc('binomial')


# ## Question 1
# 
# What is the difference between the quartiles (Q1, Q2 and Q3) of the `normal` and` binomial` variables of `dataframe`? Respond as a tuple of three elements rounded to three decimal places.
# 
# In other words, let `q1_norm`,` q2_norm` and `q3_norm` be the quantiles of the variable` normal` and `q1_binom`,` q2_binom` and `q3_binom` the quantiles of the variable` binom`, what is the difference `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom) `?

# In[188]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return round(q_diff[0], 3), round(q_diff[1], 3), round(q_diff[2], 3)
    pass


# ## Question 2
# 
# Consider the interval $ [\bar{x} - s, \bar{x} + s] $, where $ \bar{x} $ is the sample mean and $ s $ is the standard deviation. What is the probability in this interval, calculated by the empirical cumulative distribution function (empirical CDF) of the `normal` variable? Respond as a single scalar rounded to three decimal places.

# In[189]:


def q2():
    return round(float(interval_probability), 3)
    pass


# ## Question 3
# 
# What is the difference between the means and variances of the binomial and normal variables? Respond as a tuple of two elements rounded to three decimal places.
# 
# In other words, let m_binom and v_binom be the mean and variance of the binomial variable, and m_norm and v_norm the mean and variance of the normal variable. What are the differences (m_binom - m_norm, v_binom - v_norm)?

# In[190]:


def q3():
    return round(binomial_mean - normal_mean, 3), round(binomial_var - normal_var, 3)
    pass


# ## Part - 2
# <p><p>
#     
# ### _Setup_ 

# In[191]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# In[209]:


# Creating a variable to store mean_profile when target is false (0)
mean_profile = stars[stars['target'] == 0][['mean_profile']]

# standardized variable:
false_pulsar_mean_profile_standardized = mean_profile['mean_profile'].apply(lambda x: (x-mean_profile.mean())/mean_profile.std())


"""
==============
Question - 4
==============
"""
# These cumulative probabilities will be used into ppf function to obtain the values correlated with the 
probabilities_part2 = [0.8, 0.9, 0.95]

# values that result into given propabilities
values = sct.norm.ppf(probabilities_part2, loc=0, scale=1)

# Create an instance of DataStatAnalysis to analyse use ecdf function.
stat_analysis_par2 = DataStatAnalysis(false_pulsar_mean_profile_standardized)

# Calibrate ecdf model with mean_profile column
stat_analysis_par2.ecdf_calibrate('mean_profile')

# Calculate probabilities using ecdf and theotical values.
probabilities_ecdf = stat_analysis_par2.ecdf_calc(values)

"""
==============
Question - 5
==============
"""
# Calcule quartile (25%, 50% and 75%) from data (mean_profile column)
quartiles_from_data = stat_analysis_par2.quantile_calc('mean_profile')

# Calculate quartile (25%, 50% and 75%) of a normal distribution of mean 0 and variance 1
quartiles_theorical = sct.norm.ppf([0.25, 0.5, 0.75], loc=0, scale=1)

# Calculate the difference between data values and theorical values
q_diff_part2 = [round(quartiles_from_data[x] - quartiles_theorical[x], 3) for x in range(len(quartiles_theorical))]


# ## Question 4
# 
# Considering the `mean_profile` variable of` stars`:
# 
# 1. Filter only the values ​​of `mean_profile` where` target == 0` (ie, where the star is not a pulsar).
# 2. Standardize the `mean_profile` variable previously filtered to have mean 0 and variance 1.
# 
# We will call the resulting variable `false_pulsar_mean_profile_standardized`.
# 
# Find the theoretical quantiles for a normal distribution of mean 0 and variance 1 for 0.80, 0.90 and 0.95 using the `norm.ppf ()` function available in `scipy.stats`.
# 
# What are the probabilities associated with these quantiles using the empirical CDF of the variable `false_pulsar_mean_profile_standardized`? Respond as a tuple of three elements rounded to three decimal places.
# In other words, let m_binom and v_binom be the mean and variance of the binomial variable, and m_norm and v_norm the mean and variance of the normal variable. What are the differences (m_binom - m_norm, v_binom - v_norm)?

# In[210]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return round(probabilities_ecdf[0], 3), round(probabilities_ecdf[1], 3), round(probabilities_ecdf[2], 3)
    pass


# ## Question 5
# 
# What is the difference between the Q1, Q2 and Q3 quantiles of `false_pulsar_mean_profile_standardized` and the same theoretical quantiles of a normal distribution of mean 0 and variance 1? Respond as a tuple of three elements rounded to three decimal places.

# In[211]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return round(q_diff_part2[0], 3), round(q_diff_part2[1], 3), round(q_diff_part2[2], 3)
    pass

