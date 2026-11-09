# Live PyTorch behavior

CPU observation of PyTorch 2.11.0+cu130; A2 requirements govern the product even where PyTorch differs. No GPU queried or allocated.

| Case | PyTorch result |
|---|---|
| nondivisible | ValueError |
| num_groups/zero | ZeroDivisionError |
| num_groups/negative | accepted constructor |
| num_groups/float | accepted constructor |
| num_groups/true | accepted constructor |
| num_groups/false | ZeroDivisionError |
| num_groups/string | TypeError |
| num_groups/none | TypeError |
| num_channels/zero | accepted constructor |
| num_channels/negative | RuntimeError |
| num_channels/float | TypeError |
| num_channels/true | ValueError |
| num_channels/false | TypeError |
| num_channels/string | TypeError |
| num_channels/none | TypeError |
| rank0 | RuntimeError |
| rank1 | RuntimeError |
| channels | RuntimeError |
| torch.int64/input | RuntimeError |
| torch.int64/weight | RuntimeError |
| torch.int64/bias | RuntimeError |
| torch.bool/input | RuntimeError |
| torch.bool/weight | RuntimeError |
| torch.bool/bias | RuntimeError |
| torch.uint8/input | RuntimeError |
| torch.uint8/weight | RuntimeError |
| torch.uint8/bias | RuntimeError |
| mixed/torch.float32/torch.float16 | RuntimeError |
| mixed/torch.float16/torch.float32 | RuntimeError |

Exact exception messages and operands: pytorch_behavior.json. Both mixed configurations refuse during forward; there is no backward outcome to claim.
