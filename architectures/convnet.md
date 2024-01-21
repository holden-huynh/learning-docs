# Convolution
### standard
normal conv: 
- input: $D_i$ channels
- output: $D_o$ channels
- $D_o$ kernels, each with shape $[D_i \times k \times k] -> (D_o \times D_i \times k \times k)$ params
### separable
#### spatial separable
- [k x k] kernel = [1 x k] x [k x 1] kernels

#### depth-wise separable
- $D_i$ [1 x k x k] kernels applied __separately__ to $D_i$ input channels -> $D_i$-channel output
- one 1 x 1 conv kernel [$D_o$ x 1 x 1] -> $D_o$-channel output
- no. params = $D_i$ x k x k + $D_o$
### dilatied
### transposed
### padding & stride
---
# Designs & Scaling
### General design
#### Structure
* stem
* body
    - stages operating at progressively reduced resolutions
    - each stage consists of identical blocks (except 1st block)
* head

#### Principles
* mostly 3x3 conv
* feature maps in the same stage have the same resolution and channels
* cross stage
    - channels doubles
    - resolution halves

### Computation
- actual running time of conv layers on larger feature maps is slower than those on smaller feature maps, when their time complexity is the same (i.e. different no. filters)

### Scaling
Given target FLOPS

- Vary depth
    - VGG
    - ResNet
- Decrease depth & increase width
    - Wide ResNet: 
        - width multiplier k
        - complexity quadratic with k
- Increase cardinality
    - ResNeXt
- Resolution
    - Mobilenets
- Jointly
    - EfficientNet
# Receptive fields

- https://theaisummer.com/receptive-field/
# Visualization
- https://distill.pub/2017/feature-visualization/
- https://www.comet.com/team-comet-ml/cnn-visualizations/reports/visualizations-to-interpret-convolutional-neural-networks
- https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
- https://christophm.github.io/interpretable-ml-book/cnn-features.html