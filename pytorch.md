* https://github.com/vahidk/EffectivePyTorch
# 1. Basics
---
### Tensors

* item(): single **Python** value
* data: attribute of Variable (access to Variable's Tensor) ~ detach
* numel(): num of elements
* dim(): dimension
* contiguous():
    * C-contiguous (vs Fortran-contiguous)
    * https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays/26999092#26999092
* view() v.s reshape():
    * view: same data, new shape, can only operate on contiguous tensors
    * reshape: not guarantee same data or copy, can operate on non-contiguous tensors
* transpose():
* unsqeeze() ~ expand_dims?
* ops with suffix "\_": in-place ops

#### Ops
* tensor slicing:
    * gather: https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
* accessing (read/write) arbitrary elements: 
    * by list of `x`s and list of `y`s
    * T[(x0, x1, x2,..), (y0, y1, y2,..)]
#### Conversion

* numpy to Tensor: Tensor(), torch.as_tensor()
* Tensor to numpy: [.cpu().data].numpy()
### Autograd

* autograd keeps a record of data (tensors) & all executed operations (along with the resulting new tensors) in a DAG
* In a forward pass, autograd does two things simultaneously:
    * run the requested operation to compute a resulting tensor, and
    * maintain the operation’s gradient function in the DAG
* The backward pass kicks off when .backward() is called on the DAG root. autograd then:
    * computes the gradients from each .grad_fn,
    * accumulates them in the respective tensor’s .grad attribute, and
    * using the chain rule, propagates all the way to the leaf tensors.
* graph dynamically created from scratch __after each .backward()__
* grad tape (record operations as they occur and replays them backward) vs symbolic derivatives
* detach()

#### References:
* https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
* [PyTorch Autograd - Towards Data Science](https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95)
* detach: [B. Nikolic: PyTorch .detach() method](http://www.bnikolic.co.uk/blog/pytorch-detach.html)
* https://stackoverflow.com/questions/56816241/difference-between-detach-and-with-torch-nograd-in-pytorch
* https://discuss.pytorch.org/t/clone-and-detach-in-v0-4-0/16861/5
# 2. Data
---
### Datasets
#### map-styled
* \_\_len__, \_\_getitem__

#### iterable-styled
* \_\_iter__
* not random read

#### collate_fn
* torchvision's default: convert list of 2D tuples to tuple of 2 lists (images, targets)

#### transformation
* transform / target_transform
* transforms
* torchvision transforms:
    - ToTensor(): RGB, range [0..1]

#### multi-process
* do not return CUDA tensors, use pin_memory instead

#### References
* https://twitter.com/_ScottCondron/status/1363494433715552259
* https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html
# 3. Model
---
### nn.Module
base class
#### basic:

* forward(\*input)
* parameters() / named_parameters():
    * iterator over all params
* modules() / named_modules():
    * generator of all submodules
# 4. Optimizer
---
## 4.1. basic
* param_group: 
    - list of groups, each is a dict (e.g. {'lr': .., 'weight_decay': .., })
    - params of each group ('lr', 'weight_decay' etc.), if provided, are used instead of optimizer's values
* without zero_grad() or no_grad, grads are always accumulated -> OOM
* step() adjusts each param by its stored .grad
## 4.2. lr scheduler
Update lr for optimizer
### \_LRScheduler
- last_epoch: 
    - last epoch or iteration, depends on context
- init:
    - last_epoch: 
        - default = -1 --> set all optimizer's param groups 'initial_lr' to their current 'lr'
        - if not -1, optimizer's param groups must have 'initial_lr' already
    - base_lrs: 
        - initialized with optimizer's param_group's 'initial_lr' 
    - call step()
- step(epoch=None):
    - must be called AFTER optimizer.step(), at the end of training loop (or iteration)
    - epoch arg is deprecated
    - increase last_epoch & \_step_count THEN calculate next lr(s) via #get_lr()
    - set calculated next lr(s) for each param_group of optimizer 
    - keep calculated lr(s) in self.\_last_lr
- get_lr(): 
    - calculate lr for each param group at CURRENT last_epoch
- get_last_lr():
    - To get the last lr(s) (self.\_last_lr) computed by the scheduler, do not call get_lr() directly
    
#### Examples:
- initial optimizer & scheduler
- load saved optimizer & scheduler and continue training
- load saved optimizer & scheduler and adjust lr before continue training
#### References:
- https://pytorch.org/docs/stable/data.html
# Performance
- Quantization: https://pytorch.org/docs/stable/quantization.html
- https://pytorch.org/tutorials/recipes/quantization.html
- Mixed precision: https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
- torchscript: https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#saving-and-loading-models
- https://stackoverflow.com/questions/70503585/pytorch-model-optimization-automatic-mixed-precision-vs-quantization
# Debugging
* https://pytorch.org/blog/trace-analysis-for-masses/