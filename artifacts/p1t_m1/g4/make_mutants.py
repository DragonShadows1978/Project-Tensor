"""P1T-M1 G4: create the four registered mutant source trees under /tmp/p1t_m1_mut_<x>/tensor_cuda (each a copy of
the worktree's tensor_cuda/ without build products) and apply exactly one edit each. Mine."""
import pathlib, subprocess, sys
W = pathlib.Path('/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_m1/tensor_cuda')

def edit(path, old, new):
    s = path.read_text()
    assert s.count(old) == 1, (path, old[:60], s.count(old))
    path.write_text(s.replace(old, new))

MUTANTS = {
 'a': ('src/bindings.cpp',  # import copies into a new engine Storage instead of aliasing (capsule left unconsumed)
       '''  Device dev;
  dev.type = DeviceType::CUDA;
  dev.index = cur;
  auto storage = std::make_shared<Storage>(''',
       '''  Device dev;
  dev.type = DeviceType::CUDA;
  dev.index = cur;
  if (true) {  // MUTANT (a): copy instead of alias
    NDArray c = NDArray::empty(shape, dt, dev);
    if (numel > 0)
      cudaMemcpy(c.data_ptr(), static_cast<char*>(dl->data) + dl->byte_offset,
                 static_cast<size_t>(numel) * isz, cudaMemcpyDeviceToDevice);
    return Tensor::make(c, false);
  }
  auto storage = std::make_shared<Storage>('''),
 'b': ('src/bindings.cpp',  # export manager holds a raw Storage*, not the shared_ptr
       '''struct DLPackExportCtx {
  std::shared_ptr<Storage> owner;''',
       '''struct DLPackExportCtx {
  Storage* owner;  // MUTANT (b): raw pointer, no keepalive'''),
 'b2': ('src/bindings.cpp',
       '''  ctx->owner = a.storage;''',
       '''  ctx->owner = a.storage.get();  // MUTANT (b)'''),
 'c': ('src/bindings.cpp',  # export ignores stream: argument checks kept, no event record / wait
       '''  cudaStream_t consumer =
      (s == 2) ? cudaStreamPerThread''',
       '''  if (true) return;  // MUTANT (c): no event wait
  cudaStream_t consumer =
      (s == 2) ? cudaStreamPerThread'''),
 'd': ('src/kernels.cu',  # import Storage calls the deleter without draining the engine stream
       '''    if (device.is_cuda()) cudaStreamSynchronize(0);
    external_release(external_ctx);''',
       '''    // MUTANT (d): no engine-stream drain before the deleter
    external_release(external_ctx);'''),
}
for m in ('a', 'b', 'c', 'd'):
    root = pathlib.Path(f'/tmp/p1t_m1_mut_{m}')
    root.mkdir(exist_ok=True)
    subprocess.run(['rsync', '-a', '--exclude', 'build', '--exclude', '*.so', '--exclude', '__pycache__',
                    f'{W}/', f'{root}/tensor_cuda/'], check=True)
    keys = ['b', 'b2'] if m == 'b' else [m]
    for k in keys:
        rel, old, new = MUTANTS[k]
        edit(root / 'tensor_cuda' / rel, old, new)
    print('mutant', m, 'applied', keys)
