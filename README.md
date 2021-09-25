# project1_DC
### Table of Contents:
* [Installation](#Installation)
* [Project Motivation](#Project-Motivation)
* [File Descriptions](#File-Descriptions)
* [Results](#Results)
* [Licensing, Authors, and Acknowledgements](#Licensing,-Authors,-and-Acknowledgements)
* [Blog](#Blog)

## Installation
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.

## Project Motivation
### For this project, I was interestested in using In Hospital Mortality Prediction to better understand:
The predictors of in-hospital mortality for intensive care units (ICU)-admitted HF patients remain poorly characterized. In this project, I created validate a prediction model for all-cause in-hospital mortality among ICU-admitted HF patients.

The primary outcome of the study was in-hospital mortality, defined as the vital status at the time of hospital discharge in survivors and non-survivors; the tasks involved are the following:

The full set of files related to this course are owned by Udacity, so they are not publicly available here. However, you can see pieces of the analysis here. This README also serves as a template for students to follow in creating their own project README files.

## libraries used

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
%matplotlib inline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

## File Descriptions
There are 3 notebooks available here to showcase work related to the above study. Each of the notebooks is exploratory in searching through the data pertaining to showcased by the notebook title. Markdown cells were used to assist in walking through the thought process for individual steps.

There is an additional .py file that runs the necessary code to obtain the final model used to predict .

## Results
After applying the different classification models, we have got below accuracies with different models:

Logistic Regression — 89%
Nearest Neighbor — 90%
Support Vector Machines — 89%
Kernel SVM — 92%
Naive Bayes — 93%
Decision Tree Algorithm — 91%
Random Forest Classification — 90%

## Blog 

Title preview :
In Hospital Mortality Prediction. Machine Learning Engineer | by RazanAL-asiri | Sep, 2021 | Medium
link> https://medium.com/@razanaasiri2/in-hospital-mortality-prediction-ad2048863b91

## Licensing, Authors, Acknowledgements
Must give credit to In Hospital Mortality Prediction Data You can find the Licensing for the data and other descriptive information at the Kaggle link available here. Otherwise, feel free to use the code here as you would like!
