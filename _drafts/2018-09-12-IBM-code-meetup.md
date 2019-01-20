---
layout: post
title:  "*Model checking in Bayesinan analysis* - notes from BDA3"
date:   2018-11-24
---

## The place of model checking in applied Bayesian statistics

More than one model can adequately fit the data. 
Sensitivity analysis: how much do posterior infrerences change when other reasonble probability models are used in place of the present model?

Best idea is to set up a *super-model*: a set of all possible 'true' models. This is impossible in all but simplest situations. 

Do not ask: 'Is this model true or false?', rather 'Do the model's deficiencies have a noticeable effect on the substantive inferences?'

## Do the inferences from the model make sense?

*External validation*: Make predictions about new data, collect that data and then check the predictions.  This is not always practicable.  

There are always different choices in what predictions to make. e.g.:

* the posterior distribution, $p(\tilde{y} | y)$
* the marginal predictive distribution $p(\tilde{y}_i | y)$

## Posterior predictive checking

Replicated data generated under the model should look similar to observerd data  so check models graphically first. 

More quantititatively: We can use tail-area probabilities to quantify important differences.  $y^{rep}$ are replications of $y$ (same explantory variables, same model parameters).  $\tilde{y}$ is any future observation (may have different explanatory variables). 

For fixed $\theta$ (classical):
$$
p_C = \mathrm{Pr}(T(y^{rep})\ge T(y)|\theta)
$$

For varying $\theta$ (Bayesian):
$$
P_B = \mathrm{Pr}(T(y^{rep}, \theta)\ge T(y, \theta)|y)
$$

For $T$ don't use a sufficient statistic with uninformative prior as you will get $p_B \simeq 0.5$. Choose a $T$ that is important for your inferences. Look at $|T|$ and $p_C$ to check how well it's fitting.  There's no need to worry about multiple comparison adjustments as we're not making decisions as to which model to choose (yet).  We're just looking to quantify how the data differs from the model. Posterior predictive checking should usually be a point of departure from creating a better model or collecting more data. 

*P and u values* If $\theta$ has very low uncertainty or $T(y,\theta) = T(y)$ (i.e. distribution of T is independent of $\theta$ and continuous), $p_B \sim \mathrm{U}(0,1)$ if the model is true. As uncertainty in $\theta$ increases $p_B$ distribution becomes concentrated near 0.5 if the model is true. 

*u value* is a function of $y$ which has uniform sampling distribution. $p$ value is to $u$ value as posterior interval is to confindence interval.  

*Likelihood principle* In statistics, the likelihood principle is that, given a statistical model, all the evidence in a sample relevant to model parameters is contained in the likelihood function [wikipedia](https://en.wikipedia.org/wiki/Likelihood_principle).  If two likelihood functions are proportional to each other then they contain the same inferences about parameters.  If only difference between two data sets is the way they were collected, then the likelihood functions are proportional to each other.  

For posterior checking - differences in data collection methods only affect distribution of $p(y^{rep}|\theta)$, posterior inferences, $p(\theta|y)$, don't change. 

*Marginal predictive checks*. For marginal checks, typical statistic is $T(y) = y$. So we have a $p$ value for every 









