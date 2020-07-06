# System T: A Selective Data Acquisition Framework for Accurate and Fair Machine Learning Models

## Motivation
As machine learning becomes democratized in the era of Software 2.0, a serious bottleneck is acquiring enough data to ensure accurate and fair models. Recent techniques including crowdsourcing provide cost-effective ways to gather such data. 

However, simply acquiring data as much as possible is not necessarily an effective strategy for optimizing accuracy and fairness. For example, if an online app store has enough training data for certain slices of data (say American customers), but not for others, obtaining more American customer data will only bias the model training. 

## System T
We contend that one needs to selectively acquire data and propose System T, which acquires possibly-different amounts of data per slice such that the model accuracy and fairness on all slices are optimized. This problem is different than labeling existing data (as in active learning or weak supervision) because the goal is obtaining the right amounts of new data. At its core, System T maintains learning curves of slices that estimate the model accuracies given more data and uses convex optimization to find the best data acquisition strategy. 

The key challenges of estimating learning curves are that they may be inaccurate if there is not enough data, and there may be dependencies among slices where acquiring data for one slice influences the learning curves of others. We solve these issues by iteratively and efficiently updating the learning curves as more data is acquired. We evaluate System T on real datasets using crowdsourcing for data acquisition and show that System T significantly outperforms baselines in terms of model accuracy and fairness, even when the learning curves cannot be reliably estimated. 

## Architecture
System T receives as input a set of slices and their data and estimatesthe learning curves of the slices by training models on samples ofdata. .Next, System T performs the selective data acquisition optimizationwhere it determines how much data should be acquired per slice inorder to minimize the loss and unfairness. As data is acquired, thelearning curves can be iteratively updated. 

<p align="center"><img src=https://user-images.githubusercontent.com/67897374/86583165-c9960e00-bfbd-11ea-8a1f-4becfc53138c.jpg></p>
