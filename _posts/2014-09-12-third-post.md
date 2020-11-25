---
title: "Tweets: Disaster or Not"
subtitle: A Natural Language Processing (NLP) project using the pretrained bert model
author: Jeremy Ribeiro
layout: post
icon: fa-briefcase
icon-style: fas
published: true
hidden: true

---

In this blog post I will present a [Kaggle](https://www.kaggle.com/) based [NLP](https://www.wikiwand.com/en/Natural_language_processing)
project (see the project on Kaggle [https://www.kaggle.com/c/nlp-getting-started/overview](https://www.kaggle.com/c/nlp-getting-started/overview)). The project is the following.
We are given a data set containing tweets and some extra information about them. These tweets 
are labeled according to whether or not they speak about disasters. The goal of the project is simple: Making a 
model that automatically classifies tweets into the category "it speaks about a disaster" or 
"it does not speak about a disaster".

To do that I will use the pretrained bert model, created by Google, that allows to extract the meaning of words (embedding).
What it does is that it transforms words into a 768-dimensional vector, such that the vectors of words with similar 
meaning are somewhat close to each other. On top of this bert layer, I will add two dense layers
that are there to learn the classification task. I'll explain the training procedure, and the interest of using 
a pretrained layer.

I will also use more traditional machine learning algorithms and methods to perform this task by first 
trying to create features form the tweets, and then learning from these features. These features 
are features of the tweets that are not explicit in the tweets, like the mean word length of a tweet for 
example. For this reason I will call the features meta-data in the following.

I will later explain how to combine the bert based model and the meta-data based classifiers to try to improve 
the overall performance of the model. To do so, I will briefly explain a few approaches and 
develop a little more of the "Stacking" strategy I have used.

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
the NaN values are only in the columns 'keyword' and 'location' we can simply count all the NaNs for each column, and we get the following.

 | Feature | number of NaN |
 |:--------|:-------------:|
 | id      |     0         |
 |keyword  |     61        |
 |location |     2533      |
 |text     |     0         |
 |target   |     0         |

The shape of the data set is (7613,5). This means that the data set contains 7613 samples. For each of them we have 4 features and the target column.
Let us check how many of those samples are in each class:

  <center> 
    {% include image.html url="/assets/images/Kaggle:NLP-Twitter/count_sample_inclass.png" description="Figure 1." %} 
  </center>

We can see a small imbalance (40:60) between the two classes, but it is not too bad to work with.

Now that we have an idea of the data we have, let's talk about what metric we will use to measure 
the predictive power of the model.

### Metrics

In order to assess the quality of the model, we need to choose a metric. The Kaggle project page
suggests using the so called f1 score. Let us see what is this score and why it is a good metric.

The f1 score is an aggregation of two other metrics called the recall and the precision.
To explain these metrics are let us look at the following figure.

  <center> 
    {% include image.html url="/assets/images/Kaggle:NLP-Twitter/case_description.svg" description="Figure 2." %} 
  </center>
  <br>
The figure represents all the tweets of the data set: Each dot represents a tweet. When green, the dot represents
a tweet talking about a disaster, otherwise it is red. The goal of the model is to automatically find the 
tweets talking about a disaster, ie it should find the green dots. The following figure represents a
possible outcome of a model.

  <center> 
    {% include image.html url="/assets/images/Kaggle:NLP-Twitter/high_recall.svg" description="Figure 3.
  The dots inside the 'circle' represent the tweets that have been classified by the model as 'tweet talking about disaster'.
  Here the model correctly classified all the green dots, but there are many dots inside the circle are red." %} 
  </center>
  <br>
  We can define what 
  the recall and precision are using this example. The recall (or recall score) is the fraction of green dots that are in the 
  circle (ie correctly classified): In this example it would be $$100\%$$ since all the green dots are in the 
  circle. The precision is the fraction of dots in the circle that are green: Here it would be less than $$50\%$$ since most of the dots in the 
  circle are red. 
  
  <center>
  {% include image.html url="/assets/images/Kaggle:NLP-Twitter/high_precision.svg" description="Figure 4. In this 
  figure the recall less than 50%, while the precision is 100%" %} 
  </center>
  <br>
  Ideally, we would like a model to have a high recall **and** a high precision. In order to deal 
  a unique number, one can aggregate these two metrics into a single one. The f1 score is such an aggregation of 
  the recall and the precision. In particular the f1 score is defined as being the harmonic mean of the recall and the precision, ie
  if we call the recall $$R$$ and the precision $$P$$, then the f1 score is defined as,
  
  $$f1:= \frac{2 R P}{R+P}.$$
  
  For example, the illustration of Figure 5. depicts a model with a high f1 score, ie with a high recall **and** a high precision.
  
  <center>
  {% include image.html url="/assets/images/Kaggle:NLP-Twitter/high_f1.svg" description="Figure 5. In this 
  figure both the recall and the precision are close 100%" %} 
  </center>
  <br>
  
  You might now wonder why we pick an expression relatively complicated to aggregate the precision and the recall. Indeed,
  one could for example, choose to simply compute the arithmetic mean $$\tfrac{R+P}{2}$$ of the recall and the precision. This is 
  indeed a possibility, but the arithmetic mean as the inconvenience that its value does not depend on the difference between 
  the recall and the precision: the arithmetic mean will be the same when $$(R,P)=(1,0)$$ and when $$(R,P)=(0.5, 0.5)$$. On the other 
  hand, the f1 score can be seen as the arithmetic mean to which we add a penalty term that depends on the difference of $$R$$ and $$P$$. Indeed,
  the above expression of the f1 score can be rewritten as,
  
  $$f1 = \frac{R+P}{2}-\frac{(R-P)^2}{2(R+P)}.$$
  
  Moreover, the f1 score is not very sensitive to imbalanced data 
  (one class of the classification being more represented in the data than the other, like in Figure 1.). 
  Some metrics are quite sensitive to an imbalanced data set, like the [accuracy](https://www.wikiwand.com/en/Accuracy_and_precision#/In_binary_classification).
  
### Cleaning process

I will now explain how to perform the cleaning process of the tweets. Indeed, text data, and in particular tweets, are too 
messy to be used as is in a model. We need to "standardize" as much as possible the text that will be use as input 
to the model. For example, the following tweet,
> only had a car for not even a week and got in a fucking car accident .. Mfs can't fucking drive . 

becomes after cleaning,
> only had a car for not even a week and got in a fucking car accident  .  .  mfs cannot fucking drive  .  

Or the following,
> .@NorwayMFA #Bahrain police had previously died in a road accident they were not killed by explosion https://t.co/gFJfgTodad 

becomes
>  .  @ norwaymfa  # bahrain police had previously died in a road accident they we are not killed by explosion url 

In these two examples, we see that all capital letter have been set to lower case fonts, punctuation has been separated from 
words, contraction (eg. can't, shouldn't...) are expanded (cannot, should not...), the '@' symbol and the '#' symbol are also 
separated from words. Moreover, our modification should include some common typo corrections, split some word glued together
(eg 'autoaccidents' -> 'auto accidents'), etc.

But how do we know what transformation to make, which typos to correct, and which "glued words" to split?

The strategy here is to use an embedding model (like GloVe or FastText) or a list of vocabulary that already exists, and see how many words in 
the tweets can be found in the embedding or the vocabulary list. For the following we define the text coverage and the vocabulary coverage as follows:
- vocabulary coverage: It is the fraction of **unique** words of the text (the tweets) that can be found in the embedding or vocabulary list.
- text coverage: It is the fraction of (**not necessarily unique**) words of the text (the tweets) that can be found in the embedding or vocabulary list.

For example, let us say that we look at the following text: "The first President of the United States is GeorgeWashington". 
The 8 unique words in this text are (we lower case all word for simplicity): "the", "first", "president", "of", "united", "states", "is", "georgewashington"
Let us assume that the vocabulary list we use contains the following words: "the", "first", "president", "of", "united", "states", "is", "george", "washington". Among the 8 unique words of the text, 7 can be found in the vocabulary list, since "georgewashingtion" is not in this list. Therefore, 
the vocabulary coverage in this example is $$7/8 = 0.875$$. On the other hand, there are 9 words in the text (since the word "the" is repeated), and 8 of them 
are in the vocabulary list, so the text coverage is $$8/9 \approx 0.89$$.

The goal of the cleaning procedure will be to transform the text so that the text and the vocabulary coverage become as close as possible to 1. In the above example, one only needs to split "georgewashington" into "george washington" in order to make both the text coverage and vocabulary coverage equal to 1. To do so, 
we need to find all the words in the text that are not in the vocabulary list, sort them from the most frequent to the least frequent, and make the necessary changes in the cleaning function so that the text coverage and vocabulary coverage increase. You can find some code about this cleaning process in the section 4 of [this notebook on Kaggle](https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert).

Personally, I merely reuse the cleaning function created by the author of the above mentioned notebook on Kaggle. I modified this function so that it runs faster 
and so that it generalizes more easily to other text data set. I do so by using the power of regular expressions more extensively. 
You can find my cleaning function on my [own notebook]().

### Feature extraction: adding meta-data

We will now see that from the raw tweet one can extract some features that are not explicit in the text body (here the tweets). 
This meta-data can be used for checking that the training set and the test set we have have the same statistics. Indeed,
if this weren't the case, then there are chances that the learning on the traning set poorly generalizes on the test set. On the 
contrary, if they have similar statistics for several of these fetures then one can be more confident that the model 
will give good resuts on the test set. It is therefore a good sanity check to do before even starting to work on the model.

On the other hand, the statistics of these feature might be slightly different for the tweets that speak about disters compared to the ones that do not.
If this is the case, then one could use this meta-data and leverage these differences in the statistics to develop a model classifying the tweets.

I found this idea interesting and so, as a firt step, I tried to classifiy the tweets using the meta data only, and see how I can go with this. I will
develop more about this the models I used for that in the next section. Let me now tell you what features I have extracted from the tweets. 
Several of these features are presented in this [Kaggle notebook](https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert), 
and others have been added by myself. In total I have extracted 15 features:
1. The number of hashtags of a tweet.
2. The number of capipitalized words in a tweet
3. The number of words of a tweet
4. The number of unique words of a tweet
5. The number url of a tweet
6. The mean word length of a tweet
7. The number of characters of a tweet
8. the number of punctuation characters of a tweet
9. The number of mentions of a tweet
10. The number of mentions of a tweet that can be found in tweets of the training set labeled as 'tweet taklking about a disater'.
  The idea is that, some mentions might be particularly frequent in the tweets talking about disasters. One can think about 
  mention of newspapers' accounts etc.
11. The number of mentions of a tweet that can be found in tweets of the training set labeled as 'tweet *not* taklking about a disater'.
    The idea is the same as above, but with the tweets that do not talk about a disaster.
12. The difference between the last two features. 
13. The number of [2-grams](https://www.wikiwand.com/en/N-gram) of a tweet that are among the top 100 2-grams in tweets of the training set labeled as 'tweet taklking about a disater'
14. The number of 2-grams of a tweet that are among the top 100 2-grams in tweets of the training set labeled as 'tweet *not* taklking about a disater'.
  The idea of this feature and the previous one is that some 2-grams can be more frequent in "disaster tweets" and other 2-gram more present in "none disaster tweets".
15. The difference between the last two features

<i class="fas fa-exclamation-triangle" style="font-size:2em;"></i> One needs to be very careful when writing the code that will extract these features: 
Some of them (the features 10, 11, 13, and 14) use the labels ('target' feature of the data set) of the training set 
in order to extract the relevant information. 
You should allow your code to use the labels on the traning set **only**, **not** on the validation set. Not paying attention to 
that can result in a label leakage (aka target leakage) which [wikipedia defines](https://www.wikiwand.com/en/Leakage_(machine_learning)) as follows:

> In statistics and machine learning, leakage (also data leakage, or target leakage) is the use of information in the model training process which would not be expected to be available at prediction time, causing the predictive scores (metrics) to overestimate the model's utility when run in a production environment.[1]
Leakage is often subtle and indirect, making it hard to detect and eliminate. Leakage can cause modeler to select a suboptimal model, which otherwise could be outperformed by a leakage-free model.[1] 

In order to avoid this issue, one needs to split the data set into a training set and a validation set *before* adding any features.
```python
X_train, X_val, y_train, y_val = train_test_split(data_set[['keyword','text', 'text_cleaned']], 
                                                  data_set['target_corrected'])
```

Then for the steps that use the labels of the training set one needs to make use of transformers like the ones provided by the library [scikit-learn](https://scikit-learn.org/stable/index.html). Let's see a code example of that:
```python
# Create the transformer object of the classe CountMentionInClass(). 
# This class is a customized version of a scikit-learn transformer that I have programed for 
# creating the above mentioned features 10 and 11.
mentionCounter = CountMentionsInClass() 

# Create and add the features 10 and 11 to the traning set X_train. 
# Note that the fit_transform() method uses the labels stored in y_train
X_train = mentionCounter.fit_transform(X_train, y_train, column='text')

# Create and add the features 10 and 11 to the validation set X_val. 
# Note that the transform() method does not use any label.
X_val = mentionCounter.transform(X_val, column='text')
```
The code for the definition of the CountMentionInClass class can be found in the MyClasses.py file [here]().
By splitting the data set before adding the features, and by using transformers as shown above, allows to reduce the risk 
of a label leakage. It has the additional advantage that code written like this can 
easily be included into a pipeline as we will see later in this post.


## Classification using meta-data only <a name='meta-data_clf'></a>

Once the meta-data features are created one can try to use these in order to classify the tweets. I will quickly present
two models for that: One using Random Forest and one using Logistic Regression. For these two models I will
present ways of trying to interpret the model. In particular, I will present a method to see what features are 
important for the model. Then I will present a method that allows to interpret a classification done by the model, 
ie we will try to answer the following question: Given a tweet, why did the model "decided" to classify it the way it did?

In both cases we will use X_train, y_train as training data, and we will use 
X_val, y_val for to assess the model. We will consider that all 
the meta-data features have already been added to this training data. By running the following
```python
X_val.head(3)
```
we get,
<center>
<div style="resize: both;  overflow: auto;">
<table class="dataframe" border="1">
  <thead>
    <tr style="text-align: right;  font-size: 0.7em;">
      <th></th>
      <th>keyword</th>
      <th>text</th>
      <th>text_cleaned</th>
      <th>hastags_count</th>
      <th>capital_words_count</th>
      <th>word_count</th>
      <th>unique_word_count</th>
      <th>url_count</th>
      <th>mean_word_length</th>
      <th>char_count</th>
      <th>punctuation_count</th>
      <th>mention_count</th>
      <th>count_mentions_in_disaster</th>
      <th>count_mentions_in_ndisaster</th>
      <th>difference_mentions_count</th>
      <th>count_2-grams_in_disaster</th>
      <th>count_2-grams_in_ndisaster</th>
      <th>difference_2-grams_count</th>
    </tr>
  </thead>
  <tbody style="font-size: 0.7em;">
    <tr>
      <th>6447</th>
      <td>suicide%20bombing</td>
      <td>@Haji_Hunter762 @MiddleEastEye maybe some muzz...</td>
      <td>haji hunter middle east eye maybe some muzzies...</td>
      <td>0</td>
      <td>1</td>
      <td>17</td>
      <td>17</td>
      <td>0</td>
      <td>6.941176</td>
      <td>134</td>
      <td>13</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7433</th>
      <td>wounded</td>
      <td>@wocowae Officer Wounded Suspect Killed in Exc...</td>
      <td>@ wocowae officer wounded suspect killed in e...</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>7.363636</td>
      <td>91</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4836</th>
      <td>mass%20murder</td>
      <td>@yelllowheather controlled murder is fine. mas...</td>
      <td>@ yelllowheather controlled murder is fine . ...</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>15</td>
      <td>0</td>
      <td>5.750000</td>
      <td>107</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</center>



### Random Forest


#### Model explaination

### Logistic Regression

#### Model explaination

## Classification using the pretrained Bert model  <a name='Bert'></a>

### Model explaination

## Combining the Bert model with meta-data based model <a name='Combine'></a>

## Integrate the whole model into a pipeline <a name='Pipeline'></a>

## Conclusion <a name='Conclusion'></a>
