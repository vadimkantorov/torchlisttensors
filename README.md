# torchlisttensors
- traverses the autograd backward graph and prints all saved-for-backward tensors and leafs
- starting from a provided root
- also tries to discover gc-tracked alive, and non gc-tracked, and gc-tracked non-alive using `gc.get_objects()` and `sys.getobjects()`. Currently (in python <= 3.14) `sys.getobjects()` is only available only when python is built with `--with-trace-refs` build option - according to the Python Discuss thread linked below
- useful for memory/OOM debugging

## References
- https://github.com/pytorch/pytorch/issues/91692 - the main ask/discussion for inclusion of this utility in PyTorch core
- https://github.com/pytorch/pytorch/issues/104247
- https://discuss.python.org/t/list-all-objects-gc-tracked-or-not-of-a-given-type-for-debugging-introspection-and-memory-profiling
- https://www.mail-archive.com/numpy-discussion@scipy.org/msg46120.html
- https://github.com/szagoruyko/pytorchviz/blob/0adcd83af8aa7ab36d6afd139cabbd9df598edb7/torchviz/dot.py#L146

## Example usage:
```python
import torch

import torchlisttensors
    
model = torch.nn.Sequential(torch.nn.Linear(20, 20), torch.nn.Linear(20, 20), torch.nn.Linear(20, 20))
model = torchlisttensors.assign_names(model, 'model')
# model.apply(lambda module: module.register_forward_hook(torchlisttensors.assign_names_output_hook))

x = torch.zeros(4, 20)
x.__name__ = 'x'

loss = model(x).sum()
loss.__name__ = 'loss'

z = torch.zeros(65, 35)
z.__name__ = 'z'

tensors = torchlisttensors.torchlisttensors(loss)
for i, t in tensors.items():
    print(i, '\t', t)
```

```
140022603754192          shape=(20,20)  name=model.0.weight
140022604499456          shape=(20)     name=model.0.bias
140022604499376          shape=(20,20)  name=model.1.weight
140022604499536          shape=(20)     name=model.1.bias
140022604499696          shape=(20,20)  name=model.2.weight
140022604499616          shape=(20)     name=model.2.bias
140022606243040          shape=(4,20)   name=x
140022604499776          shape=()       name=loss
140022604500336          shape=(65,35)  name=z
140022604500256          shape=(4,20)   name=mat1_AddmmBackward0_140021819231392_SAVED
140022604506656          shape=(20,20)  name=mat2_AddmmBackward0_140021819231392_SAVED
140022604500176          shape=(4,20)   name=mat1_AddmmBackward0_140021821040880_SAVED
140022604507136          shape=(20,20)  name=mat2_AddmmBackward0_140021821040880_SAVED
```

Also a nasty warning is printed:
```
/home/vadimkantorov/.local/lib/python3.12/site-packages/torch/__init__.py:1028: FutureWarning: `torch.distributed.reduce_op` is deprecated, please use `torch.distributed.ReduceOp` instead
  return isinstance(obj, torch.Tensor)
/mnt/c/Users/vadim/torchlisttensors/torchlisttensors.py:112: FutureWarning: `torch.distributed.reduce_op` is deprecated, please use `torch.distributed.ReduceOp` instead
  elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
```
