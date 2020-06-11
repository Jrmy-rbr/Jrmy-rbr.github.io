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


## The Data

I have taken the data from worldometer.com (for France) from which I have made a csv file (**add
link here**). It represents the evolution of the following quantities through time:


|:---------------------------|:-------------------------|
|   1. the daily new cases   | 2. the total cases       |
|3. the daily new recoveries | 4.  the total recoveries |
|  5. the daily new deaths   | 6. the total new deaths  |
|    7. the active cases.    |                          |


**add figure**


## The Models

### The model I have used

#### Some explanations on the traditional SIRD model

In order to analyse the data of the COVID-19 I will use a variation of the SIRD model. SIRD stands for
Susceptible Infected Recovered Dead.  It is a simple model for epidemiology that works as follows: The
studied population, comprised of N individuals, is divided into four categories: susceptible (S),
infected (I), recovered (R), dead (D). For each of these categories the model describes the evolution
of the number of members belonging to these categories as a function of time. To do so the model uses
the following system of four differential equations,


$$
\begin{cases}
  \frac{dS}{dt}= -\beta I \frac{S}{N}\\
  \frac{dI}{dt}= \beta I \frac{S}{N} - \gamma I - \mu I\\
  \frac{dR}{dt}= \gamma I\\
  \frac{dD}{dt}= \mu I\\
\end{cases}
$$

The first line intuitively reads as follows: during a small amount of time $$dt$$
the susceptible population $$S$$ (i.e. the not-infected part population that still be infected)
varies by an amount $$dS = -\beta I \frac{S}{N} dt$$. This means that if at time $$t_0$$ there are
$$S(t_0)$$ susceptible people, at time $$t_0+dt$$ there are $$S(t_0+dt)=S(t_0)+dS = S(t_0)-\beta I
\frac{S}{N}$$. Since $$dS$$ is negative, the population $$S$$ decreases with time, i.e. there are less and
less not-infected people through time. The quantity $$|dS|$$ represents the number
of new cases of COVID-19 that have occurred during the duration $$dt$$.

Let us see why we have $$dS = -\beta I \frac{S}{N} dt$$. Let us say that
each infected person infects on average a fraction $$f_{\rm inf}$$ of the susceptible
person they meet, and that on average they meet $$\kappa \times dt$$ persons during the time
$$dt$$. Importantly, not all the met people are susceptible. If we assume that people meet each other in
a sufficiently random way, then there should be a fraction $$\frac{S}{N}$$  (where $$N$$ is the total size
of the studied population) of the met people that are susceptible.
In other words, each of the infected people meets on average $$\kappa \times dt \times \frac{S}{N}$$ {\bf
susceptible} persons and infects a fraction $$f_{\rm inf}$$ of them, i.e. they infect on average $$f_{\rm inf}
\times \kappa \times dt \times \frac{S}{N}$$ people. Since there are $$I$$ infected people, each of them
infecting on average $$f_{\rm inf} \times \kappa \times dt \times \frac{S}{N}$$ people, there are in total
$$|dS| = I\times f_{\rm inf} \times \kappa \times dt \times \frac{S}{N}$$ new infections during the time
$$dt$$.
By defining $$\beta := f_{\rm inf} \times \kappa$$ we get the desired result: $$dS = -\beta I \frac{S}{N}
dt$$ (remember that $$dS$$ must be negative since $$S$$ must decrease). The parameter $$\beta$$ is the rate a
which each of the infected people infects other people.

Similarly, the three other equations respectively describe how the numbers $$I$$ of infected people, $$R$$
of recovered people, and $$D$$ of dead evolve through time. The parameter $$\gamma$$ is the rate at which
infected people recover from the COVID-19, and $$\mu$$ is the rate at which they die from COVID-19.

(**add plot**)




#### Modifications

**First ingredient: slpit categories**


**Second ingredient: Agglomerate several categories into one of interest ($$T_d$$)**


**Last ingredient: split $$\beta$$**

### The model used in [https://science.sciencemag.org/content/368/6490/489.full]

## Using the model
### Some intuition


### Key observation


## Conclusion






