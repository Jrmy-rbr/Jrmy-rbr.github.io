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

In this blog post I will present a [Kaggle](https://www.kaggle.com/) based [NLP](https://www.wikiwand.com/en/Natural_language_processing)
project (see the project on Kaggle [https://www.kaggle.com/c/nlp-getting-started/overview](https://www.kaggle.com/c/nlp-getting-started/overview)). The project is the following.
We are given a data set containing tweets and some extra information about them. These tweets 
are labeled according to whether they speak about disasters or not. The goal of the project is simple: Making a 
model that automatically classifies tweets into the category "it speaks about a disaster" or 
"it does not speak about a disaster".

To do that I will use the pretrained bert model, created by Google, that allows to extract the meaning of words (embedding).
What it does is that it transforms words into a 768-dimensional vector, such that words with similar 
meaning are somewhat close to each other. On top of this bert layer, I will add two dense layers
that are there to learn the classification task. I'll explain the training procedure, and the interest of using 
a pretrained layer.

I will also use more traditional machine learning algorithms and method to perform this task by first 
trying to create features form the tweets, and then learning from these features. These features 
are features of the tweets that are not explicit in the tweets, like the mean word length of a tweet for 
example. For this reason I will call the features meta-data in the following.

I will later explain how to combine the bert based model and the meta-data based classifiers to try to improve 
the overall performance of the model. To do so, I will briefly explain a few approaches and 
develop a little more the "Stacking" strategy I have used.

I am using the occasion of this blog post to also explain a little some strategies one can use 
to try to understand what the model is doing, and why it classifies tweets the way they do.

But first things first. Let me start with presenting the data, and how one can clean the data
before it is used in the machine learning models.

## Table of content
1. [Data & metrics](#Data)
2. [Classification using meta-data only](#meta-data_clf)
3. [Classification using the pretrained Bert model](#Bert)
4. [Combining the bert model with meta-data based model](#Combine)
5. [Inegrate the whole model into pipeline](#Pipeline)
6. [Conclusion](#Conclusion)


## Data & metrics<a name='Data'></a>

I use the [data](https://www.kaggle.com/c/nlp-getting-started/data) provided by Kaggle.
To have an idea of what the data looks like, let's run the following in python.

```python
data_set = pd.read_csv('./train.csv')
test_set = pd.read_csv('./test.csv')

data_set.head()
```
From which we get the following output:

<center>
<table class="dataframe"  border="1">
  <thead>
    <tr style="text-align: right; font-size: 0.7em;">
      <th></th>
      <th>id</th>
      <th>keyword</th>
      <th>location</th>
      <th>text</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody style="font-size: 0.7em;">
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
</center>

As you can see the data contains 5 columns, and one of them is the target label: The target label is equal to $$1$$ whenever 
the tweet is about a disaster, and is equal to $$0$$ otherwise. We already see that in some columns that there are some NaN values
that we will have to take care during the cleaning phase of the data. To make sure that 
the NaN values are only in the columns 'keyword' and 'location' we can simply count all of the NaNs for each column, and we get the following.

 | Feature | number of NaN |
 |:--------|:-------------:|
 | id      |     0         |
 |keyword  |     61        |
 |location |     2533      |
 |text     |     0         |
 |target   |     0         |

The shape of the data set is (7613,5). This means that the data set contains 7613 samples. For each of them we have 4 features and the target column.
Let us check how many of those samples are in each classes:

  <center> 
    {% include image.html url="/assets/images/Kaggle:NLP-Twitter/count_sample_inclass.png" description="Figure 1." %} 
  </center>

We can see a small imbalance (40:60) between the two classes, but it is not to bad to work with.

Now that we have an idea of the data we have, let's talk about what metric we will use to measure the 
the predictive power of the model.

### Metrics

In order to assess the quality of the model, we need to choose a metric. The Kaggle project page
suggests the so called f1 score. Let us see what is this score and why it a good metric.

The f1 score is an aggregation of two other metrics called the recall and the precision.
To explain the these metrics are let us look at the following figure.

  <center> 
    {% include image.html url="/assets/images/Kaggle:NLP-Twitter/case_description.svg" description="Figure 2." %} 
  </center>

The figure represents all the tweets of the data set: Each dot reprents a tweet. When green, the dot represents
a tweet talking about a disaster, otherwise it is red. The goal of the model is to automatically find the 
tweets talking about a disaster, ie it should find the green dots. The following figure represents a
possible outcome of a model.

  <center> 
    {% include image.html url="/assets/images/Kaggle:NLP-Twitter/high_recall.svg" description="Figure 3.
  The dots inside the 'circle' represent the tweets that have been classified as 'tweet talking about disaster' by the model." %} 
  </center>
  
  Here the model correctly classified all of the green dots, but there are many dots inside the circle are red. We can define what 
  the recall and precision are using this example. The recall (or recall score) is the fraction of green dots that are in the 
  circle (ie correctly classified): In this example it would be $$100\%$$ since all the green dots are in the 
  circle. The precision is the fraction of dots in the circle that are green: Here it would be less than $$50\%$$ since most of the dot in the 
  circle are red. Ideally we would like a model to have a high recall **and** a high precision. 
  
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
