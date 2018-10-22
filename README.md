# Patch_Match_python
The algorithm of patch match implemented by python
# Introduction
This code mainly implement the algorithm of patch match, for details, please see the paper: [PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing](http://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf)

There are two patchmatch code: PatchMatch.py and PatchMatch_Bidirectional.py. The last one serves for the code [DeepImageAnalogy](https://github.com/MingtaoGuo/Deep-image-analogy-TensorFlow)

### DeepImageAnalogy example
|Input|:|Output|::|Output|:|Input|
|-|-|-|-|-|-|-|
|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/A.jpg)|:|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/a2b.jpg)|::|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/b2a.jpg)|:|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/B_prime.jpg)|

![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/patchmatch.jpg)
# Results
|image|reference|
|-|-|
|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/road_.jpg)|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/thomas.jpg)|

|Iteration 0|Iteration 1/6|Iteration 2/6|Iteration 3/6|Iteration 4/6|Iteration 5/6|Iteration 1|
|-|-|-|-|-|-|-|
|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/0.jpg)|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/1.jpg)|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/2.jpg)|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/3.jpg)|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/4.jpg)|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/5.jpg)|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/6.jpg)|

|image|reference|
|-|-|
|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/img.png)|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/ref.png)|

|Iteration 1|Iteration 2|Iteration 5|
|-|-|-|
|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/1_itr.gif)|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/2_itr.gif)|![](https://github.com/MingtaoGuo/Patch_Match_python/blob/master/IMAGE/5_itr.gif)|
