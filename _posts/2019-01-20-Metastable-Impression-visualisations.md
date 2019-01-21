---
title: 'Metastable impressions: style transfer part 1'
layout: post
date: '2019-01-20'
---


My latest science/art crossover project is **Metastable impressions: Artistic Representations of Molecular Dynamics** (osf link coming soon).  It's with Alex Jones, [George Holloway](http://georgehollowaycomposer.com/) and [Pete Bennett](www.peteinfo.com). The project is in two main parts: 1) novel molecular rendering using [deep learning style transfer](https://arxiv.org/abs/1508.06576) and 2) a classical informed sonification. We are going to be using a large simulation of the enzyme setd8 by [John Chodera's lab](http://www.choderalab.org/), you can find the code and links to papers and data [here](https://github.com/choderalab/SETD8-materials). 

In order to get going with the style transfer we want to understand how this process works on different molecular renderings. To understand this I have generated a set of 728 visualization with different rendering parameters. The idea will be to use this set (or rather a smaller subset eventually) as targets for different style transfers from different style sources.  

The code is written in [VMD](https://www.ks.uiuc.edu/Research/vmd/) and you will need to manually download a `.h5` source trajectory from Chodera's data repository.  My code is [here](https://github.com/RobertArbon/metastable_impressions/tree/master/Visualisations) and may be useful for someone learning how to script in VMD. The renderings all use the VMD defaults as the idea is not to generate the best possible render, rather to quickly generate a set for testing style transfer. 

Some of my favourite renderings from an aesthetic point of view are below. The filenames give the settings: bgcol16_Diffuse_Dotted_ResType means background colour 16 (black), material is 'Diffuse', draw method is 'Dotted' and colour method is 'ResType'. 

![](/images/bgcol16_Diffuse_Dotted_ResType.png){:height="400px"}
![](/images/bgcol8_Diffuse_Polyhedra_Structure.png){:height="400px"}
![](/images/bgcol16_Goodsell_Beads_ResType.png){:height="400px"}
![](/images/bgcol16_Goodsell_Licorice_Chain.png){:height="400px"}
![](/images/bgcol16_Goodsell_Tube_Chain.png){:height="400px"}

