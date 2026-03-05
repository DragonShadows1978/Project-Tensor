import numpy as np
import tensor_gpu_v2 as tg
import torch
import sys
import os

def test_conv_transpose2d():
    print("Testing ConvTranspose2D...")
    tg.set_device('cpu')
    
    in_channels = 3
    out_channels = 2
    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 1
    
    # N, C, H, W
    x_np = np.random.randn(1, in_channels, 4, 4).astype(np.float32)
    w_np = np.random.randn(in_channels, out_channels, kernel_size, kernel_size).astype(np.float32)
    b_np = np.random.randn(out_channels).astype(np.float32)
    
    # tg version
    conv_tg = tg.ConvTranspose2D(in_channels, out_channels, kernel_size, stride, padding, output_padding)
    conv_tg.w.data = w_np
    conv_tg.b.data = b_np
    
    x_tg = tg.Tensor(x_np, requires_grad=True)
    y_tg = conv_tg(x_tg)
    loss_tg = y_tg.sum()
    loss_tg.backward()
    
    # torch version
    x_torch = torch.tensor(x_np, requires_grad=True)
    # torch ConvTranspose2d weights are (in_channels, out_channels, kH, kW)
    conv_torch = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
    with torch.no_grad():
        conv_torch.weight.copy_(torch.tensor(w_np))
        conv_torch.bias.copy_(torch.tensor(b_np))
    
    y_torch = conv_torch(x_torch)
    loss_torch = y_torch.sum()
    loss_torch.backward()
    
    # Comparisons
    y_ok = np.allclose(y_tg.data, y_torch.detach().numpy(), atol=1e-5)
    dw_ok = np.allclose(conv_tg.w.grad, conv_torch.weight.grad.numpy(), atol=1e-5)
    db_ok = np.allclose(conv_tg.b.grad, conv_torch.bias.grad.numpy(), atol=1e-5)
    dx_ok = np.allclose(x_tg.grad, x_torch.grad.numpy(), atol=1e-5)
    
    if y_ok and dw_ok and db_ok and dx_ok:
        print("ConvTranspose2D: PASS")
    else:
        print(f"ConvTranspose2D: FAIL (y={y_ok}, dw={dw_ok}, db={db_ok}, dx={dx_ok})")

def test_adaptive_avg_pool2d_nhwc():
    print("Testing AdaptiveAvgPool2D NHWC...")
    tg.set_device('cpu')
    
    # N, H, W, C
    x_np = np.random.randn(1, 8, 8, 3).astype(np.float32)
    output_size = (4, 4)
    
    # tg version
    pool_tg = tg.AdaptiveAvgPool2D(output_size, data_format='NHWC')
    x_tg = tg.Tensor(x_np, requires_grad=True)
    y_tg = pool_tg(x_tg)
    loss_tg = y_tg.sum()
    loss_tg.backward()
    
    # torch version (torch is NCHW)
    x_torch = torch.tensor(x_np.transpose(0, 3, 1, 2), requires_grad=True)
    pool_torch = torch.nn.AdaptiveAvgPool2d(output_size)
    y_torch = pool_torch(x_torch)
    loss_torch = y_torch.sum()
    loss_torch.backward()
    
    # Comparisons
    y_tg_nchw = y_tg.data.transpose(0, 3, 1, 2)
    y_ok = np.allclose(y_tg_nchw, y_torch.detach().numpy(), atol=1e-5)
    
    dx_tg_nchw = x_tg.grad.transpose(0, 3, 1, 2)
    dx_ok = np.allclose(dx_tg_nchw, x_torch.grad.numpy(), atol=1e-5)
    
    if y_ok and dx_ok:
        print("AdaptiveAvgPool2D NHWC: PASS")
    else:
        print(f"AdaptiveAvgPool2D NHWC: FAIL (y={y_ok}, dx={dx_ok})")

if __name__ == "__main__":
    test_conv_transpose2d()
    test_adaptive_avg_pool2d_nhwc()
