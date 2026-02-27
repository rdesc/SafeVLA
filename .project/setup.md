# setup.md — SafeVLA

## Going through the original SafeVLA code
- They don't seem to support proper resume training. The lagrange multipliers are not saved as part of the checkpoint
- 


## 


```shell
git clone -b real-wip --recurse-submodules https://github.com/rdesc/SafeVLA
```


## Confirm ai2thor is properly installed and vulkan works
```python
from ai2thor.platform import CloudRendering
from ai2thor.controller import Controller
print('imported pkgs, launching controller....')
controller = Controller(platform=CloudRendering, gpu_device=0, server_timeout=300)
print('launched controller')
controller.step(dict(action='Initialize', gridSize=0.25))
print('stepped with controller')
```


## Setup on nibi
