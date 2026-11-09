# Complete Python reshape census

37 engine calls on 32 text lines, plus 9 NumPy calls. The reported 30 is an undercount; multiple calls share source lines.

| File:line:column | Call | Check |
|---|---|---|
| tensor_cuda/tensor_cuda/__init__.py:553:35 | `faces.reshape(-1)` | NumPy array operation; NDArray::reshape is never invoked. |
| tensor_cuda/tensor_cuda/__init__.py:555:8 | `np.roll(faces, -1, axis=1).reshape(-1)` | NumPy array operation; NDArray::reshape is never invoked. |
| tensor_cuda/tensor_cuda/__init__.py:1394:13 | `onehot.reshape(logits.shape)` | NumPy array operation; NDArray::reshape is never invoked. |
| tensor_cuda/tensor_cuda/functional.py:189:53 | `tc.causal_softmax(             s.reshape([B_, H_ * L, 1, S])).reshape([B_, H_, L, S])` | Undo causal-softmax flattening: B*H*L*S; output same element count. |
| tensor_cuda/tensor_cuda/functional.py:190:12 | `s.reshape([B_, H_ * L, 1, S])` | Flatten H*L and insert query extent 1; B*H*L*S unchanged. |
| tensor_cuda/tensor_cuda/functional.py:226:23 | `tt.reshape(shape)` | Einsum permutation preserves all operand axes; insert only missing singleton axes. |
| tensor_cuda/tensor_cuda/nn.py:304:15 | `x.reshape([B, L, self.num_heads, self.head_dim])` | B*L*heads*head_dim = B*L*embed_dim; constructor defines head_dim=embed_dim//heads. |
| tensor_cuda/tensor_cuda/nn.py:315:14 | `out.transpose(1, 2).reshape([B, L, self.embed_dim])` | Transpose copies preserve B*L*embed_dim; merge heads back to embed_dim. |
| tensor_cuda/tensor_cuda/nn.py:363:13 | `self.weight.reshape([self.out_ch, K])` | K=in_ch*kh*kw; weight has out_ch*K elements. |
| tensor_cuda/tensor_cuda/nn.py:367:14 | `out.transpose(1, 2).reshape([N, self.out_ch, OH, OW])` | L=OH*OW from im2col; output has N*L*out_ch elements. |
| tensor_cuda/tensor_cuda/nn.py:369:24 | `self.bias.reshape([1, self.out_ch, 1, 1])` | Bias vector length out_ch; singleton axes add no elements. |
| tensor_cuda/tensor_cuda/nn.py:389:15 | `cols.reshape([N, C, self.kh * self.kw, L])` | im2col gives N*(C*kh*kw)*L, split its channel/kernel axis. |
| tensor_cuda/tensor_cuda/nn.py:390:12 | `self.weight.reshape([1, C, self.kh * self.kw, 1])` | Depthwise weight has C*kh*kw elements; add singleton axes. |
| tensor_cuda/tensor_cuda/nn.py:394:14 | `out.reshape([N, C, OH, OW])` | Sum removed the kernel axis, leaving N*C*(OH*OW). |
| tensor_cuda/tensor_cuda/nn.py:396:24 | `self.bias.reshape([1, C, 1, 1])` | Bias length C; add singleton axes. |
| tensor_cuda/tensor_cuda/nn.py:424:13 | `self.weight.reshape([self.in_ch, self.out_ch * self.kh * self.kw])` | Transposed-conv weight in_ch*out_ch*kh*kw; merge last three axes. |
| tensor_cuda/tensor_cuda/nn.py:425:13 | `x.reshape([N, Cin, H * W])` | Flatten H*W; element count N*Cin*H*W unchanged. |
| tensor_cuda/tensor_cuda/nn.py:432:24 | `self.bias.reshape([1, self.out_ch, 1, 1])` | Bias length out_ch; add singleton axes. |
| tensor_cuda/tensor_cuda/nn.py:470:12 | `self.weight.reshape([1, self.C, 1, 1])` | BN2D weight length C; singleton axes only. |
| tensor_cuda/tensor_cuda/nn.py:471:12 | `self.bias.reshape([1, self.C, 1, 1])` | BN2D bias length C; singleton axes only. |
| tensor_cuda/tensor_cuda/nn.py:480:19 | `self.running_mean.reshape([1, self.C, 1, 1])` | BN2D running_mean length C; singleton axes only. |
| tensor_cuda/tensor_cuda/nn.py:481:18 | `self.running_var.reshape([1, self.C, 1, 1])` | BN2D running_var length C; singleton axes only. |
| tensor_cuda/tensor_cuda/nn.py:498:12 | `self.weight.reshape(wshape)` | BN1D parameter length C; wshape=[1,C] or [1,C,1]. |
| tensor_cuda/tensor_cuda/nn.py:498:45 | `self.bias.reshape(wshape)` | BN1D parameter length C; wshape=[1,C] or [1,C,1]. |
| tensor_cuda/tensor_cuda/nn.py:505:19 | `self.running_mean.reshape(wshape)` | BN1D running_mean length C; wshape adds only singleton axes. |
| tensor_cuda/tensor_cuda/nn.py:506:18 | `self.running_var.reshape(wshape)` | BN1D running_var length C; wshape adds only singleton axes. |
| tensor_cuda/tensor_cuda/nn.py:576:12 | `x.reshape([N, self.G, (C // self.G) * math.prod(spatial)])` | GroupNorm validates C%G=0; product is N*C*prod(spatial), also when N/spatial is zero. |
| tensor_cuda/tensor_cuda/nn.py:581:12 | `g.reshape([N, C] + spatial)` | Inverse of grouped layout; product N*C*prod(spatial). |
| tensor_cuda/tensor_cuda/nn.py:585:23 | `self.weight.reshape(wshape)` | Each GroupNorm parameter length C; wshape=[1,C]+[1]*len(spatial). |
| tensor_cuda/tensor_cuda/nn.py:585:53 | `self.bias.reshape(wshape)` | Each GroupNorm parameter length C; wshape=[1,C]+[1]*len(spatial). |
| tensor_cuda/tensor_cuda/nn.py:588:21 | `weight.reshape(wshape)` | Float32-cast parameters retain length C; broadcast singleton shape. |
| tensor_cuda/tensor_cuda/nn.py:588:46 | `bias.reshape(wshape)` | Float32-cast parameters retain length C; broadcast singleton shape. |
| tensor_cuda/tensor_cuda/nn.py:602:20 | `self.weight.reshape([1, self.C, 1, 1])` | InstanceNorm parameter length C; singleton axes only. |
| tensor_cuda/tensor_cuda/nn.py:602:61 | `self.bias.reshape([1, self.C, 1, 1])` | InstanceNorm parameter length C; singleton axes only. |
| tensor_cuda/tensor_cuda/nn.py:613:24 | `x.reshape([N, C, 1, L])` | Conv1D inserts height 1; N*C*L preserved. |
| tensor_cuda/tensor_cuda/nn.py:614:15 | `out.reshape([N, out.shape[1], out.shape[3]])` | Height-one Conv2D output; removing its singleton height preserves count. |
| tensor_cuda/tensor_cuda/nn.py:696:17 | `x.slice(1, t, 1).reshape([B, x.shape[2]])` | RNN time slice length 1; remove singleton time axis, keep B*input_dim. |
| tensor_cuda/tensor_cuda/nn.py:799:19 | `self.pe.slice(0, 0, L).reshape([1, L, self.d_model])` | Position table slice L*d_model; prepend singleton batch axis. |
| tensor_cuda/tensor_cuda/quant.py:124:9 | `R.float().reshape([1, H, D, D])` | R has H*D*D elements; prepend singleton B before expand. |
| tensor_cuda/tensor_cuda/quant.py:186:11 | `main.reshape(list(x.shape[:-1]) + [nb, _ROW_SUM_BLOCK])` | main contains nb*4096 keys per row (tail removed); split last axis into nb and 4096. |
| tensor_cuda/tensor_cuda/quantization/affine.py:156:14 | `w.reshape(out_features, groups, int(group_size))` | NumPy array operation; NDArray::reshape is never invoked. |
| tensor_cuda/tensor_cuda/quantization/affine.py:167:25 | `codes.reshape(out_features, in_features)` | NumPy array operation; NDArray::reshape is never invoked. |
| tensor_cuda/tensor_cuda/quantization/affine.py:195:8 | `q.reshape(rows, groups, int(group_size))` | NumPy array operation; NDArray::reshape is never invoked. |
| tensor_cuda/tensor_cuda/quantization/affine.py:198:11 | `w.reshape(rows, int(in_features))` | NumPy array operation; NDArray::reshape is never invoked. |
| tensor_cuda/tensor_cuda/quantization/affine.py:216:8 | `q.reshape(rows, groups, int(group_size))` | NumPy array operation; NDArray::reshape is never invoked. |
| tensor_cuda/tensor_cuda/quantization/affine.py:218:11 | `w.reshape(rows, int(in_features))` | NumPy array operation; NDArray::reshape is never invoked. |
