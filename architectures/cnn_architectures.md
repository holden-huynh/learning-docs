# 1. resnet
### structure
#### conv
- original: conv - BN - RELU
- new: BN - RELU - conv

#### residual block:
- basic block: 2 3x3 conv layers
    - same width for 2 layers
- bottleneck block: 3 conv layers
    - 1x1 conv: to compress width (e.g. 256 to 64) to reduce no. params
    - 3x3 conv: compressed width (e.g.64)
    - 1x1 conv: restore width (e.g. 64 to 256)

#### depth = number of conv layers
- always 4 layers
- resnet50 = 1 conv + 3 * (3+4+6+3) + 1 avg_pool

#### residual connection
- if resolution halves, channel doubles
    - spatial: stride 2
    - channel: zero padding or 1x1 conv
### complexity
- resnet34: 3.6B FLOPS
- resnet152: 11.3B FLOPS
- no. params: 
### Evaluation
- imagenet
    - top-1 (val): 
        - resnet34: ~78%
        - resnet50: ~79%
        - resnet152: ~80.6%
    - top-5 (val): 
        - resnet34: ~94.4%
        - resnet50: ~94.75%
        - resnet152: ~95.51%
    - top-5 (test):
        - ensemble: 96.43%
## wide resnet

idea: decrease depth and increas width (no. channels)        

## resnext
Adopts strategy of __repeating layers__ (VGG & Resnet), while exploiting the Inception's <span style="color:red">__split-transform-merge__</span> (multi-path representation) strategy. A module performs a set of transformations on low-dim embeddings; outputs are aggregated by summation
* resnet's bottleneck block: 256-d in -> __[ [256, 1x1, 64] -> [64, 3x3, 64] -> [64, 1x1, 256] ]__ -> 256-d out
* resnext introduce cardinality (group)
    * instead of 64 channels, __split__ to 32 paths, 4 channel each (__[256, 1x1, 4]__)
    * each __transformed__ by __[4, 3x3, 4]__ 
    * back to original dim by __[4, 1x1, 256]__
    * __merge__ by aggregated transformation (sum)
    * residual as normal resnet
    
<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-06_at_4.32.52_PM.png" alt="resnext" width="500"/>

Equivalent forms:
* early concatenation of low-dim embeddings instead of transform back to original dim
    * concatenated 32 paths into 128-d embedding
    * transformed to 256-d by __[128, 1x1, 256]__
* Group convolution
    * 256-d in -> 128-d embedding by __[256, 1x1, 128]__
    * transformed by group conv __[128, 3x3, 128, group=32]__
    * back to original by __[128, 1x1, 256]__
    * looks similar to bottleneck block, but wider (128 vs 64) and sparsely connected
    * references:
        - https://towardsdatascience.com/grouped-convolutions-convolutions-in-parallel-3b8cc847e851
# 2. inception
### Inception v1
Inception module approximates sparse computation by dense conv(s): factorizes simultaneous channel & spatial correlation into a series of independent channel correlations & spatial correlations
* input is <span style="color:red">__split__</span> into lower dimensional embeddings by 1x1
* <span style="color:red">__transformed__</span> by 3x3, 5x5 filters 
* <span style="color:red">__merged__</span> by concat

Details:
* 1 5x5 conv (after 1x1 to reduce computation)
* 1 3x3 conv (after 1x1 to reduce computation)
* 1 1x1 conv
* 1 maxpool (followed by 1x1 to reduce computation since maxpool keeps the same number of filters)
* All concatenated
### Inception v2
* 5x5 conv factorized to 2 3x3 conv(s)
* 3x3 conv factorized to 1x3 and 3x1 conv
* auxilliary classifiers
* grid size reduction with 2 parallel stride-2 blocks
* label smoothing
### xception
* replace all conv in Inception by depthwise separable conv
* residual connections
# 3. mobilenets
all conv are 3x3 depthwise separable conv
### mobilenets v2
inverted residual block
# 4. densenet
### Structure
#### Dense block
* connect all layers (with matching feature-map size) directly with each other
* combine features by concatenating (channel-wise), not (element-wise) summing as in ResNet
* better params efficiency
    * requires fewer parameters, as no need to re-learn redundant feature maps
    * layers are very narrow (few filters per layer), adding only a small set of feature maps
* easy to train
    * every layer has access to grads of loss and original input

#### Transition layers
* conv and pooling

#### Bottleneck layers
* 1x1 conv to reduce input feature maps to improve computational efficiency

### Efficiency

- https://www.reddit.com/r/MachineLearning/comments/67fds7/d_how_does_densenet_compare_to_resnet_and/
- Memory-Efficient Implementation of DenseNets: https://arxiv.org/pdf/1707.06990.pdf

### References:
- https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803
# CSP
# Attention
### SENet
Explicitly models relationship between channels (channel attention)
* feature recalibration -> learn to use global information to selectively emphasize informative features and suppress less useful ones

Squeeze & Excitation block
* Input: feature maps U
* Squeeze: aggregate feature maps across spatial dimensions -> global information of channel-wise feature responses
    * input: feature maps U (C x H x W)
    * output: (C x 1 x 1)
* Excitation: simple self-gating
    * input: embedding (C x 1 x 1)
    * output: per-channel modulation weights (C x 1 x 1)
* Output: weights applied to U

#### References:
* https://blog.paperspace.com/channel-attention-squeeze-and-excitation-networks/
### CBAM
Spatial and channel-wise attention
* Channel attention
    * uses both max-pool and average-pool instead of average-pool only (as in SE)
    * MLP 
* Spatial attention
    * avg-pool and max-pool along channels and concat
    * apply conv to concatenated feature descriptor -> spatial attention map
### ResNeSt
* Standard conv learns a set of filters which __aggregates the neighborhood information__ with __spatial and channel connections__ -> suitable to capture __correlated features__
* Multi-path representtion learns __independent features__ -> encourages the feature exploration by decoupling the input channel connections.

ResNeSt combines channel-wise attention with multi-path

#### Split-Attention Block
Enables feature-map __attention across different feature-map groups__
* Feature-map group: feature divided into several groups (cardinality)
    * cardinality (K)
    * radix (R): number of splits within a cardinal group -> total number of feature groups G=KR
* Split-Attention operations
    * applied to each cardinal group
    * combines multiple splits by fusing via element-wise summation
* Relation to existing attention methods:
    * R=1: SA block = SE on each cardinal group (v.s SENet applys on the entire block)
    * R=2: SA block = SKNet    