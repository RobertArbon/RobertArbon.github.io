---
layout: post
title:  "VMD scripting
date:   2019-01-19
---

Sets the display to be 512 x 512 pixels.  Snapshot renders produce this many pixels. 
```
display resize 512 512 
```

1. with scale matrix the identity, display distance 0, display height 2:   a line 2 units long in y direction fills the screen exactly i.e. 512 pixels long  
2. As 1 but with scale matrix 0.5I then rendered image is 256 pixels. 
3. as 2 but with display height 4 then rendered image is 256 pixels. 



