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
This meta-data can be used for checking that the training set and the test set we have the same statistics. Indeed,
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

In both cases we will use X_train (shape=(5709, 18)), y_train as training data, and we will use 
X_val (shape=(1904, 18)), y_val for to assess the model. We will consider that all 
the meta-data features have already been added to this training data. By running the following
```python
X_val.head(3)
```
we get,
<center>
<div style="overflow: auto;">
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
</div>
</center>

As you can see there are 18 columns. The first two were already present in the data set[^1], then follows the cleaned tweets, and finally the 15 added features.
For the classification we will use the fisrt column and the 15 added feature, ie the only column I don't use are the tweets and their cleaned version.

[^1]: Note that I revomed the location colum from the data set. This is because there are too many unique locations, which makes this column 
not useful for the classification.

### Random Forest

The model I will present in here is based on a [random forsest](https://www.wikiwand.com/en/Random_forest) classifier. In particular I will use 
the [random forest classifier from scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#forest), to which I add some data preparation
steps for the features. Note that we have a lot of numerical features, and one categorcal feature, and we need to treat them separatly in the
data preparation steps. Let us then define the following:
```python
# Categorical meta-data
cat_metaData_features = ['keyword']

# numerical meta-data
numerical_metaData_features = ['hastags_count', 'capital_words_count', 'word_count', 'unique_word_count', 
                               'url_count', 'mean_word_length', 'char_count', 'punctuation_count',
                               'mention_count', 'count_mentions_in_disaster', 'count_mentions_in_ndisaster', 
                               'difference_mentions_count', 'count_2-grams_in_disaster', 'count_2-grams_in_ndisaster',
                               'difference_2-grams_count']
# text feature
txt_feature = ['text_cleaned']
```

This allows us to quickly refer to the categorical data or to the numerical data. Here, the data preparation will be extremely 
simple. We will rescale all the numerical feature, so that their standard deviation equals 1[^2]. For the categorical feature
we need to encode them. For that, I will use the [OneHotEncoder](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features) from scikit-learn.

[^2]: This step is not necesary for a tree based model like the Random Forest since they are not sensitive to scaling, 
but it is a good habit to get so I choose to do it anyways. In all cases, I'll have to do that for the Logistic Regression in the next section.

```python
# Define the scaling and the encoding
enc_scale = ColumnTransformer([('scaler',StandardScaler(), numerical_metaData_features),
                               ('enc', OneHotEncoder(handle_unknown='ignore'), cat_metaData_features)]).fit(X_train,y_train)

# Definie the Radom Forest model
Model_Forest = RandomForestClassifier(n_estimators=900, max_depth=23, n_jobs=8, class_weight='balanced')

# Train the model on the data after scaling and encoding.
# In a next section we will see how to use pipelines in order to intgrate the model and the data preparation in a single object.
Model_Forest.fit(enc_scale.transform(X_train), y_train)


# Evaluate the model on the traning set and on the validation set.
y_train_pred = Model_Forest.predict(enc_scale.transform(X_train))
y_val_pred = Model_Forest.predict(enc_scale.transform(X_val))

# training score
print("Training scores:\n",
      "precision={:.2f}".format(skl.metrics.precision_score(y_true=y_train, y_pred=y_train_pred)),
      "recall={:.2f}".format(skl.metrics.recall_score(y_true=y_train, y_pred=y_train_pred)),
      "f1={:.2f}".format(skl.metrics.f1_score(y_true=y_train, y_pred=y_train_pred))
      )

# validation score
print("\nValidation scores:\n",
      "precision={:.2f}".format(skl.metrics.precision_score(y_true=y_val, y_pred=y_val_pred)),
      "recall={:.2f}".format(skl.metrics.recall_score(y_true=y_val, y_pred=y_val_pred)),
      "f1={:.2f}".format(skl.metrics.f1_score(y_true=y_val, y_pred=y_val_pred))
      )
```
By runing the above code we obtain the following output.

> <div style="font-family: NewCM, Mono, sans serif;">  Training scores:<br> precision=0.81 recall=0.91 f1=0.86<br><br> Validation scores:<br> precision=0.60 recall=0.74 f1=0.67</div>

To have an idea of how well the model perfroms, one should look at the f1 score for the validation set, since it should tell us what would be the score on truely new data. 
Printing the f1 score of the traning set can be useful: If it is too high compared to the score on the validation set, then the model might overfit the data. 
Overfitting can hurt the performance of the data so it is important to detect it. Here, the training score is 0.86 while the validation score is 0.67. 
There is probably some overfitting here. I suspect that this is due to the added features 10, 11, 13, and 14, which by construction 
"memorise" some text specific of the training set. It would be worth exploring this further to see weather the perfromance of the model can be improved.
However, for the purpose of this blog post, I won't do that. Instead I'll move on to the model explaination, which can actually be part of the required work for 
improving the model and reducing the overfitting. Indeed, Model explanation can is as a tool meant to diagnose issues with the model. 

Besides, it can also be used to justify the "decision" made by the model for any given example provided as input. You might want to do that 
to convince yourself, or maybe to convince others, that the model is doing something that makes sense, and that it can therefore be trusted.

### Model explaination

In order to explain the model I will use the eli5 library, that is a library specialised into model explaination. Let's start 
with Feature Importance. Feature Importance is simply a measure of the importance of each feature for the model.
Here, I will use the so called Permutation Importance. Permutation Importance compute the importance of a feature as follows.
1. Apply the model on the validation set, and evaluate the perfromance of the model.
2. - Pick a column of the validation set (X_val), *randomly* permute the values of the column. 
   - Apply the model on the validation set (on which the column has been permuted), and evaluate the new perfromance of the model.
   - Permute the column back to recover the original validation set.
3. Perform step 2 for all the columns of the validation set.

Once this is done, you can see by how much the performance has change after the permutation of a column compare to the initial perfromance 
on the original set. This difference is the permutation importance of the column.

Intuitively, *randomly* permuting a column basically erases all the corelations between the values of this column and the values of the target (stored in y_val). 
In terms of information, it is as if you erased the information contained in the column. We then expect that, the more a feature (column) is important for the model, the more the model performance drops after the permutation of this feature.

The operation I have describe above is automatically performed by the PermutationImportance class of the library eli5.

```python
##### First redefine and train the Random Forest model in a way that will be accepted by the PermutationImportance calss.

# copy X_train into X
X = X_train[cat_metaData_features+numerical_metaData_features].copy()
X[cat_metaData_features] = OrdinalEncoder().fit_transform(X[cat_metaData_features])

# The model is the same as before, but integrated into a pipeline. It is fitted on the same traning set X_train (X is used as a proxy for X_train).
model_metadata = make_pipeline(enc_scale, Model_Forest).fit(X[cat_metaData_features+numerical_metaData_features], y_train)


##### Feature importance: permutation importance

# copy X_val into X
X = X_val[cat_metaData_features+numerical_metaData_features].copy()

# Have to use OrdinalEncoder() because PermutationImportance() will internally try to convert 
# the categories into floats (idk why...), and it will fail if I keep the categories as strings
X[cat_metaData_features] = OrdinalEncoder().fit_transform(X[cat_metaData_features]) 

perm_importance = PermutationImportance(model_metadata).fit(X, y_val)  # performs the operations I have described above.
eli5.show_weights(perm_importance, feature_names=cat_metaData_features+numerical_metaData_features)  # display the importance of each feature in a table


```
    
After running the above code we get the following table.

<table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>
    
        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0243
                
                    ± 0.0114
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                mean_word_length
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 82.36%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0203
                
                    ± 0.0084
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                keyword
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 82.75%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0196
                
                    ± 0.0073
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                difference_2-grams_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 89.42%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0098
                
                    ± 0.0102
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                char_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 90.48%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0084
                
                    ± 0.0051
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                count_2-grams_in_ndisaster
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 90.65%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0082
                
                    ± 0.0030
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                count_2-grams_in_disaster
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 93.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0045
                
                    ± 0.0021
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                count_mentions_in_disaster
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 94.04%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0043
                
                    ± 0.0045
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital_words_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 94.88%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0035
                
                    ± 0.0039
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                difference_mentions_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 96.02%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0024
                
                    ± 0.0044
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                count_mentions_in_ndisaster
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 96.91%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0017
                
                    ± 0.0038
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                punctuation_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.27%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0007
                
                    ± 0.0014
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hastags_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(0, 100.00%, 99.56%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0001
                
                    ± 0.0031
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                mention_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(0, 100.00%, 96.65%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0019
                
                    ± 0.0053
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                url_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(0, 100.00%, 93.94%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0044
                
                    ± 0.0042
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                word_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(0, 100.00%, 93.05%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0054
                
                    ± 0.0050
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                unique_word_count
            </td>
        </tr>
    
    
    </tbody>
</table>

We see that for this model, the three most important features are "mean_word_length", "keyword", "difference_2-grams_count". 
The weights associated to each feature is the amount by which the perfromance of the model drops.

The above tells use how important each feature is for the model, by looking at the whole training set. There are at least two pieces 
of information on which it says nothing: It does not allow to explain the predictions of the model for individual sample of the data set, and 
does not tell in which direction a given feature influances the prediction, it only says whether or not it will have a big influence.
Eli5 gives some methods to explain the model with these two extra pieces of information. However, because of the one hot encoding,
our model is not supported by these explainers. 

I will therefore use the [shap](https://shap.readthedocs.io/en/latest/#) library which 
gives for every sample of the data set a "contribution score" for each feature. It does so 
by computing what is called the [Shapley value](https://www.wikiwand.com/en/Shapley_value) 
for each feature of a given sample. The Shapley value 
is a concept that has been developed in the context of game theory. So a priori 
it has very little to do with model explaination. The Shapeley value would deserve a blog post on its own,
but in short, the Shapley value is the solution to how to share profit among collaborator based on
a notion of "merit". The notion of merit can be given a precise definition in this game theoretic
framework, but it roughly says that if an individual contributes more he should get larger share of the profit.
We can already see some analogy with a "contribution score" of a feature. But you might wonder
what is the "profit" in our context? It's acually the difference between the predicted probability 
given by our model on a given example and a base value which can be thaugh of the probability 
the model would predict if it were not given any features. Let us see an example for the following sample,
```python
X_val.iloc[5].loc['text']
```
> <div style="font-family: NewCM, Mono, sans serif;"> '4 kidnapped ladies rescued by police in Enugu | Nigerian Tribune http://t.co/xYyEV89WIz' </div>.

Without diving into how to use the shap library, here is the output we get,

  <center> 
    {% include image.html url="/assets/images/Kaggle:NLP-Twitter/Shap_rand-forest.png" description="Figure 6." %} 
  </center>

You see that each feature is assigned to a score (positive or negative) depending on the value taken by this feature. The 
sum of all the Shapley values should be equal to the difference between the base value and the predicted probability. When a score is 
negative (in blue) it tends to decrease the predicted probability, and vice versa. As expected, we can see that the featues with the largest (in absolute value)
Shapley value often correspond to the most important features according to the permutation importance: For example, "mean_word_length " has 
the second largest Shapley value, and it also has the second highest permutation importance. Of course the shapeley values will change for each 
sample, since each of them has different feature values. To make sure that there really is a correcpondance between permutation importance and
the Shapley values, one would need to compute the Shapley values of many samples, and then check that the correspondance holds on average over these samples.


### Logistic Regression

In this section, we will see the exact same thing as before but by replacing RandomForestClassifier by LogisticRegression.
Since the code is exactly the same as before (except for the use of the LogisticRegression) I will 
simply show the f1 score of this new model.

> <div style="font-family: NewCM, Mono, sans serif;">Training scores:<br> precision=0.79 recall=0.83 f1=0.81 <br><br> Validation scores:<br> precision=0.69 recall=0.75 f1=0.72 </div>

All the remark made in the previous section apply here too. The main difference is that this model seems a little bit 
better, and seems to overfit a little less. One thing that is important though, is that for this model the scaling of the numerical values 
in the data preparation step is more important, so it is crucial not forget this step when using logistic regression.

### Model explaination

Again, everything we have seen about model explanation for the random forest model also applies here. Let's see 
what we get when computing the permutation importance for the new model: 

<table class="eli5-weights eli5-feature-importances" style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto;">
    <thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;">Weight</th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
    </thead>
    <tbody>
    
        <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1595
                
                    ± 0.0081
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                keyword
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 83.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.1256
                
                    ± 0.0128
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                char_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 86.73%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0888
                
                    ± 0.0090
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                word_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 92.53%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0391
                
                    ± 0.0108
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                difference_2-grams_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 95.38%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0196
                
                    ± 0.0080
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                count_2-grams_in_disaster
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 96.71%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0121
                
                    ± 0.0058
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                punctuation_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 97.50%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0082
                
                    ± 0.0025
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                count_mentions_in_disaster
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 97.77%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0069
                
                    ± 0.0035
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                difference_mentions_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 97.87%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0065
                
                    ± 0.0070
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                count_mentions_in_ndisaster
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.66%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0034
                
                    ± 0.0019
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                count_2-grams_in_ndisaster
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0027
                
                    ± 0.0050
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                mean_word_length
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 98.84%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0027
                
                    ± 0.0042
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                unique_word_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 99.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0020
                
                    ± 0.0041
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                mention_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(120, 100.00%, 99.88%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                0.0001
                
                    ± 0.0008
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                url_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(0, 100.00%, 99.69%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0004
                
                    ± 0.0046
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                capital_words_count
            </td>
        </tr>
    
        <tr style="background-color: hsl(0, 100.00%, 99.07%); border: none;">
            <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
                -0.0020
                
                    ± 0.0015
                
            </td>
            <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
                hastags_count
            </td>
        </tr>
    
    
    </tbody>
</table>


We can now see what the shap library tells use of the same sample, nemely on,
```python
X_val.iloc[5].loc['text']
```
> <div style="font-family: NewCM, Mono, sans serif;"> '4 kidnapped ladies rescued by police in Enugu | Nigerian Tribune http://t.co/xYyEV89WIz' </div>.

Once the Shapley values are calculated, we get the following,

  <center> 
    {% include image.html url="/assets/images/Kaggle:NLP-Twitter/Shap_logistic-reg.png" description="Figure 7." %} 
  </center>
  
  Here again we can see that the features with high contribution (positive or negative) often are the one 
  with a high permutation importance: For example, here it is true for "word_count" anf "char_count".


## Classification using the pretrained Bert model  <a name='Bert'></a>

### Model explaination

## Combining the Bert model with meta-data based model <a name='Combine'></a>

## Integrate the whole model into a pipeline <a name='Pipeline'></a>

## Conclusion <a name='Conclusion'></a>
