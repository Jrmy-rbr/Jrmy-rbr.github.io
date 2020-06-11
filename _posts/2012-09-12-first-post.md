---
title: COVID-19 the missing ones
subtitle : What does it take to estimate the number of undetected infected people
author: Jeremy
published: true
layout: post
---


In the COVID-19 pandemic, one interesting quantity one might want to estimate is the proportion $$p_{nd}$$
of infected people that are *not detected* by the tests (on a given date or period of time). There are
many reasons why not all infected persons are detected: For example,  many people might 
only have mild symptoms, and therefore do not try to get tested. Another reason is that we simply do
not have enough tests to test all the infected people, and therefore we need to prioritize by choosing who gets
to be tested and who does not. In the first case, and if the tests were cheap and abundant, then a
relatively easy solution exists to estimate the proportion of undetected infected people: We can pick
a random sample of the population that we get tested. From this, we can measure the proportion of
infected people within the sample, and by the law of large-numbers this proportion should equal the
proportion of infected people in the whole population (up to statistical fluctuations related to the
size of the sample). Knowing this and the number of detected infected people we can deduce the
proportion of undetected infected people.

However, when the number of test per day, or per week, is limited one cannot necessarily test a
sufficiently large random sample of the population. Moreover, if today we want to estimate what was  $$p_{nd}$$ at
the beginning of the pandemic, i.e. several weeks ago, one needs to have access to the results of tests
that would have been performed on a random sample of the population at the time for which we wish to
make our estimation. If no test has been performed on such a random sample, the technique fails by lack
of available data, and we need to use other data and ways of estimating $$p_{nd}$$.

This is what the authors of [https://science.sciencemag.org/content/368/6490/489.full] have done. In this post I will
explain the principles underlying such an estimation method for $$p_{nd}$$ through an analogue but simpler analysis using data about the total number of confirmed COVID-19 cases. I will try as much as
possible to draw parallels between my simplified approach and the work done in [https://
science.sciencemag.org/content/368/6490/489.full].


# The Data

I have taken the data from worldometer.com (for France) from which I have made a csv file (**add
link here**). It represents the evolution of:

\begin{center}
  \begin{tabular}{l  l}
    1. the daily new cases & 2. the total cases \\
     3. the daily new recoveries & 4. the total recoveries \\
     5. the daily new deaths & 6. the total new deaths  \\
     7. the active cases\\
  \end{tabular}
\end{center}

# The Models

## The model I have used
### Some explanations on the traditional SIRD model

### Modifications

#### First ingredient: slpit categories


#### Second ingredient: Agglomerate several categories into one of interest ($$T_d$$)


#### Last ingredient: split $$\beta$$

## The model used in [https://science.sciencemag.org/content/368/6490/489.full]

# Using the model
## Some intuition


## Key observation


# Conclusion






