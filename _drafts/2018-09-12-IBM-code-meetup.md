---
layout: post
title:  "IBM Code Meetup"
date:   2018-09-12
---

## Event

Link

Attended by academics, recruiters and teachers (at least). 



## Predicting Component Failures - a Data Science Journey (Elena Hensinger)

Company [Toumetis](https://toumetis.com/)

Reduce unknown failures. 

Normal operation --> failure indicators --> failure. 

Data: labeled data of many years time series of instrument measurements. 

Supervised sequence classification. Didn't work as labels weren't accurate.  

Unsupervised learning: cluster subsequences in similar groups. Groups: Failure, failure indicative, normal. Clusters identified by experts.  Now you have labeled clusters. :w


 

## Data-driven traffic network forecasting using Diffusion Convolutional Recurrent Neural Networks (Frank Kelly)

[Hal24K](https://hal24k.com/) @norhustla Focus on built environment. 

Transport flows are non-stationary and spatially correlated. 

Traditional approach: ARIMA good for single time series. 

Deep learning. Recurrent neural network - LSTM RNN (Long short term memory unit). Make prediction for transport flows and detect anomolies.  For traffic flows feed in tweets.  Doesn't take in spatial information. 

Convolution NN for spatial temporal stuff.   Plot traffic speeds as images. Not great as it doesn't represent full spatial information. 

Better: use Graphs.  osmnx

Diffusion convolutional recurrent neural network

Graph CNN.  tkipf.github.io/graph-convolutional-networks. 

[pyflux](https://github.com/RJT1990/pyflux)






## What can data science tell us about the University of Bristol? (Natalie Thurlby)
