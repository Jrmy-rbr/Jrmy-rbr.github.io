---
title: "Tweets: Disaster or Not"
subtitle: A Natural Language Processing (NLP) project using the pretrained bert model
author: Jeremy Ribeiro
layout: post
icon: fa-lightbulb
icon-style: regular
published: true
hidden: true
---

In this blog post I will present a Kaggle based [NLP](https://www.wikiwand.com/en/Natural_language_processing)
project. The project is the following.
We are given a data set containing tweets and some extra information about them. These tweets 
are labeled according to whether they speak about disasters or not. The goal of the project is simple: Making a 
model that automatically classifies tweets into the category "it speaks about a disaster" or 
"it does not speak about a disaster".

To do that I will use the pretrained bert model by Google that allows to extract the meaning of words (embedding).
What it does is that it transforms words into a 768-dimensional vector, such that words with similar 
meaning are somewhat close to each other. On top of this bert layer, I will add two dense layers to the model
that are here to learn the classification task. I'll explain the training procedure, and the interest of using 
a pretrained layer.

I will also use more traditional machine learning algorithms and method to perform this task, by first 
trying to create key features form the tweets, and then learning from these features. These key features 
are features of the tweets that are not explicit in the tweets, like the mean word length of a tweet for 
example. In the following I will call the features meta-data.

I will later explain how to combine the bert based model and the meta-data based classifiers to try to improve 
the overall performance of the model. To do so, I will breifly explain a few approaches and 
develop a little more the "Stacking" strategy I have used.

I am using the occasion of this blog post to also explain a little some strategies one can use 
to try to understand what the model is doing, and why it classifies tweets the way they do.

But first things first. Let me start with presenting the data, and how one can clean the data
before it is used in the machine learning models.

## Table of content
1. [Data](#Data)
2. [Classification using meta-data only](#meta-data_clf)
3. [Classification using the pretrained Bert model](#Bert)
4. [Combining the bert model with meta-data based model](#Combine)
5. [Inegrate the whole model into pipeline](#Pipeline)
6. [Conclusion](#Conclusion)


## Data <a name='Data'></a>

'''python
data_set = pd.read_csv('./train.csv')
test_set = pd.read_csv('./test.csv')

data_set.head()
'''

<table class="dataframe" border="1" sytel="font-size: 14px; width=200px">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Our Deeds are the Reason of this #earthquake M...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Forest fire near La Ronge Sask. Canada</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>All residents asked to 'shelter in place' are ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13,000 people receive #wildfires evacuation or...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Just got sent this photo from Ruby #Alaska as ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

### Cleaning process

### Feature extraction: adding meta-data

## Classification using meta-data only <a name='meta-data_clf'></a>

### Random Froest

#### explaination

### Logistic Regression

#### explaination

## Classification using the pretrained Bert model  <a name='Bert'></a>

## Combining the Bert model with meta-data based model <a name='Combine'></a>

## Inegrate the whole model into a pipeline <a name='Pipeline'></a>

## Conclusion <a name='Conclusion'></a>
