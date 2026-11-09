# Observed CPU reference behaviour

PyTorch 2.11.0+cu130 on CPU. NumPy integer counts accepted in construction and forward; all 12 wrong parameter shapes refused. Boolean refusal classes differ by operand. These observations do not override the order’s ValueError/domain requirements.

| Case | Outcome | Class | Observed result / message |
| --- | --- | --- | --- |
| nondivisible | refused | ValueError | num_channels (6) must be divisible by num_groups (4) |
| num_groups/zero | refused | ZeroDivisionError | integer modulo by zero |
| num_groups/negative | accepted |  | GroupNorm(-2, 6, eps=1e-05, affine=True) |
| num_groups/float | accepted |  | GroupNorm(2.0, 6, eps=1e-05, affine=True) |
| num_groups/true | accepted |  | GroupNorm(True, 6, eps=1e-05, affine=True) |
| num_groups/false | refused | ZeroDivisionError | integer modulo by zero |
| num_groups/string | refused | TypeError | unsupported operand type(s) for %: 'int' and 'str' |
| num_groups/none | refused | TypeError | unsupported operand type(s) for %: 'int' and 'NoneType' |
| num_channels/zero | accepted |  | GroupNorm(2, 0, eps=1e-05, affine=True) |
| num_channels/negative | refused | RuntimeError | Trying to create tensor with negative dimension -2: [-2] |
| num_channels/float | refused | TypeError | empty() received an invalid combination of arguments - got (float, dtype=NoneType, device=NoneType), but expected one of:  * (tuple of ints size, *, tuple of names names, torch.memory_format memory_format = None, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)  * (tuple of ints size, *, torch.memory_format memory_format = None, Tensor out = None, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)  |
| num_channels/true | refused | ValueError | num_channels (True) must be divisible by num_groups (2) |
| num_channels/false | refused | TypeError | empty() received an invalid combination of arguments - got (bool, dtype=NoneType, device=NoneType), but expected one of:  * (tuple of ints size, *, tuple of names names, torch.memory_format memory_format = None, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)  * (tuple of ints size, *, torch.memory_format memory_format = None, Tensor out = None, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)  |
| num_channels/string | refused | TypeError | not all arguments converted during string formatting |
| num_channels/none | refused | TypeError | unsupported operand type(s) for %: 'NoneType' and 'int' |
| rank0 | refused | RuntimeError | Expected at least 2 dimensions for input tensor but received 0 |
| rank1 | refused | RuntimeError | Expected at least 2 dimensions for input tensor but received 1 |
| channels | refused | RuntimeError | Expected weight to be a vector of size equal to the number of channels in input, but got weight of shape [6] and input of shape [2, 4, 3] |
| torch.int64/input | refused | RuntimeError | mixed dtype (CPU): all inputs must share same datatype. |
| torch.int64/weight | refused | RuntimeError | mixed dtype (CPU): expect parameter to have scalar type of Float |
| torch.int64/bias | refused | RuntimeError | expected scalar type Float but found Long |
| torch.bool/input | refused | RuntimeError | mixed dtype (CPU): all inputs must share same datatype. |
| torch.bool/weight | refused | RuntimeError | mixed dtype (CPU): expect parameter to have scalar type of Float |
| torch.bool/bias | refused | RuntimeError | expected scalar type Float but found Bool |
| torch.uint8/input | refused | RuntimeError | mixed dtype (CPU): all inputs must share same datatype. |
| torch.uint8/weight | refused | RuntimeError | mixed dtype (CPU): expect parameter to have scalar type of Float |
| torch.uint8/bias | refused | RuntimeError | expected scalar type Float but found Byte |
| mixed/torch.float32/torch.float16 | refused | RuntimeError | expected scalar type Float but found Half |
| mixed/torch.float16/torch.float32 | refused | RuntimeError | mixed dtype (CPU): expect parameter to have scalar type of Float |
| num_groups/np_int64 | accepted |  | {'shape': [2, 6, 3], 'dtype': 'torch.float32'} |
| num_groups/np_int32 | accepted |  | {'shape': [2, 6, 3], 'dtype': 'torch.float32'} |
| num_groups/np_uint8 | accepted |  | {'shape': [2, 6, 3], 'dtype': 'torch.float32'} |
| num_groups/np_bool_true | refused | TypeError | group_norm(): argument 'num_groups' (position 2) must be int, not numpy.bool |
| num_channels/np_int64 | accepted |  | {'shape': [2, 6, 3], 'dtype': 'torch.float32'} |
| num_channels/np_int32 | accepted |  | {'shape': [2, 6, 3], 'dtype': 'torch.float32'} |
| num_channels/np_uint8 | accepted |  | {'shape': [2, 6, 3], 'dtype': 'torch.float32'} |
| num_channels/np_bool_true | refused | ValueError | num_channels (True) must be divisible by num_groups (2) |
| weight/shape/() | refused | RuntimeError | Expected weight to be a vector of size equal to the number of channels in input, but got weight of shape [] and input of shape [2, 6, 3] |
| weight/shape/(2, 3) | refused | RuntimeError | Expected weight to be a vector of size equal to the number of channels in input, but got weight of shape [2, 3] and input of shape [2, 6, 3] |
| weight/shape/(6, 1) | refused | RuntimeError | Expected weight to be a vector of size equal to the number of channels in input, but got weight of shape [6, 1] and input of shape [2, 6, 3] |
| weight/shape/(1, 6, 1) | refused | RuntimeError | Expected weight to be a vector of size equal to the number of channels in input, but got weight of shape [1, 6, 1] and input of shape [2, 6, 3] |
| weight/shape/(5,) | refused | RuntimeError | Expected weight to be a vector of size equal to the number of channels in input, but got weight of shape [5] and input of shape [2, 6, 3] |
| weight/shape/(7,) | refused | RuntimeError | Expected weight to be a vector of size equal to the number of channels in input, but got weight of shape [7] and input of shape [2, 6, 3] |
| bias/shape/() | refused | RuntimeError | Expected bias to be a vector of size equal to the number of channels in input, but got bias of shape [6] and input of shape [2, 6, 3] |
| bias/shape/(2, 3) | refused | RuntimeError | Expected bias to be a vector of size equal to the number of channels in input, but got bias of shape [6] and input of shape [2, 6, 3] |
| bias/shape/(6, 1) | refused | RuntimeError | Expected bias to be a vector of size equal to the number of channels in input, but got bias of shape [6] and input of shape [2, 6, 3] |
| bias/shape/(1, 6, 1) | refused | RuntimeError | Expected bias to be a vector of size equal to the number of channels in input, but got bias of shape [6] and input of shape [2, 6, 3] |
| bias/shape/(5,) | refused | RuntimeError | Expected bias to be a vector of size equal to the number of channels in input, but got bias of shape [6] and input of shape [2, 6, 3] |
| bias/shape/(7,) | refused | RuntimeError | Expected bias to be a vector of size equal to the number of channels in input, but got bias of shape [6] and input of shape [2, 6, 3] |
