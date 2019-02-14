---
output:
  html_document: default
  pdf_document: default
---
# Introduction {#intro}

In the era of large scale data collection we are trying to make meaningful intepretation of data. 

There are two ways to meaningfully intepret data and they are 

1. Mechanistic or mathematical modeling based
2. Descriptive or Data Driven

We are here to discuss the later approach using machine learning (ML) approaches. 

## What is machine learning?

We use - computers - more precisely - algorithms to see patterns and learn concepts from data - without being explicitly programmed.

For example

1. Google ranking web pages
2. Facebook or Gmail classifying Spams
3. Biological research projects that we are doing - we use ML approaches to interpret effects of mutations in the noncoding regions. 

We are given a set of 

1. Predictors
2. Features or 
3. Inputs

that we call 'Explanatory Variables' 

and we ask different statistical methods, such as 

1. Linear Regression
2. Logistic Regression
3. Neural Networks

to formulate an hypothesis i.e.

1. Describe associations
2. Search for patterns
3. Make predictions 

for the Outcome Variables 

A bit of a background: ML grew out of AI and Neural Networks

## Aspects of ML

There are two aspects of ML

1. Unsupervised learning
2. Supervised learning

**Unsupervised learning**: When we ask an algorithm to find patterns or structure in the data without any specific outcome variables e.g. clustering. We have little or no idea how the results should look like.

**Supervised learning**: When we give both input and outcome variables and we ask the algorithm to formulate an hypothesis that closely captures the relationship. 

## What actually happened under the hood
The algorithms take a subset of observations called as the training data and tests them on a different subset of data called as the test data. 

The error between the prediction of the outcome variable the actual data is evaulated as test error. The objective function of the algorithm is to minimise these test errors by tuning the parameters of the hypothesis. 

Models that successfully capture these desired outcomes are further evaluated for **Bias** and **Variance** (overfitting and underfitting). 

All the above concepts will be discussed in detail in the following lectures. 


