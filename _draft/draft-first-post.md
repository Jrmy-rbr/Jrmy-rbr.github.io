---
title: COVID19 the missing ones
author: Jeremy
layout: post
---

In the COVID pandemic, one interesting quantity one might want to estimate is the proportion $$p_nd$$ of 
infected people that are not detected by the tests. There are many reasons why not all infected 
persons are detected: For example, it is possible that many people only have mild symptoms, and 
therefore do not try to get tested. Another reason is that we simply do not have enough test to 
test all the infected, and therefore we need to prioritize by choosing who gets to be tested and
who does not. In the first case, and if the tests were cheap and abundant, then a relatively easy 
solution exists to estimate the proportion of non-detected infected people: We can pick a random
sample of the population that we get tested. From this, we can measure the proportion of infected
people within the sample, and by the law of large-numbers this proportion should equal the 
proportion of infection people in the whole population (up to statistical fluctuations 
related to the size of the sample). Knowing this and the number of detected infected people
we can deduce the proportion of non-detected infected people.

However, when the number of test per day, or per week, is limited one cannot necessarily 
test a random sample of the population. Moreover, if we want to estimate $$p_nd$$ at the 
beginning of the pandemic one cannot use tests performed to day to infer this quantity. 
