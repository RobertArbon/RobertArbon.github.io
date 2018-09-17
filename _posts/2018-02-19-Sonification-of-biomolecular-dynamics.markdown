---
layout: post
title:  "Sonification of biomolecular dynamics"
date:   2018-02-19
---


My colleague Alex Jones and I have been looking into extending the ways scientists display their results with sonification, or audio display.  Audio display is using sound to provide information on a data set. Our [group](http://group-wacky.com) is primarily concerned with chemical dynamics and so our datasets are usually molecular dynamics (MD) trajectories. In my research I'm concerned with systems in which the dynamics is approximately <a href="https://en.wikipedia.org/wiki/Metastability">metastable</a> (the classic paper on protein metastability is <a href="http://science.sciencemag.org/content/254/5038/1598">here</a>). In the case where the dynamics is metastable the system is well approximated with  <a href="https://en.wikipedia.org/wiki/Hidden_Markov_model">hidden Markov model</a> (see <a href="http://aip.scitation.org/doi/abs/10.1063/1.4828816">here</a> for a paper making the link between biomolecular dynamics and HMMs).

We recently took one of the simpler cases of metastable systems, Alanine Dipeptide (AD) and made an audio-visual display of the its dynamics. The goal of the sonification was to allow the scientist to look at visual information (the structure and the dynamics) while hearing information on what metastable state the system is in, how stable that state is and when it moves between the metastable states.  These qualities are very difficult to display visually (although it can be done) and so the question is - does this sonification bring out these features effectively?

The results are here so you can judge for yourself.

<iframe src="https://player.vimeo.com/video/255391814" width="640" height="573" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
<p><a href="https://vimeo.com/255391814">ad_free_energy_sonification</a> from <a href="https://vimeo.com/ajj1">Alex Jones</a> on <a href="https://vimeo.com">Vimeo</a>.</p>


All the code can be found on on our Open Science Foundation  page <a href="https://osf.io/rzp3k/">here</a>. And the paper describing our efforts is on the [ArXiv](https://arxiv.org/abs/1803.05805) There's some pretty funny hydrogen motion (the white atoms) but this is due to the 3ps Buttersworth filter applied to the trajectory. We did this to smooth the motion to make it easier on the eye as the model frame rate (1ps simulated time : 0.05s physical time) is quite low.

The synopsis of our approach is that we have taken a publicly available <a href="https://simtk.org/projects/alanine-dipeptide/">AD dataset </a> and projected a random sample of 500 of these trajectories onto a set of 500 discrete states (the two 500s are coincidental) and then fitted a Markov state model and a hidden Markov model using the 500 discrete chains as data. We used parameters of the model to synthesise sound using <a href="https://cycling74.com/">Max/MSP</a>.

The HMM we created had four metastable states and these were mapped to four different note clusters.  The more stable the state, the deeper the fundamental of the note cluster.  When AD entered a transition region between metastable states there is a noisey effect.  The kick drum/pulse sound is tied to the molecules overall stability.  There are also some synthesized tones which correspond to the fast (non-metastable) dynamics of AD.

The audio-visual display was created by taking an example trajectory and then sending the parameters of the model as they pertained to each frame of the trajectory as a message to the synthesizer. This sound was exported and overlaid to an animation - which is what you see above.

 
