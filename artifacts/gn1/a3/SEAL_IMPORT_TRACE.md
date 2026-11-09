# Import-traced seal

Collected 468 regression IDs; traced/censused 534 module origins (21 local, 513 external). Nine local arm modules are covered by each arm map; the other 12 local modules and all 513 external origins are pinned directly. The trace uses module-execution profiling plus the loaded-module origin census, and the seal pins the trace itself. Child verify_start calls verify_seal before importing its arm.

| Imported local module | SHA-256 | Child pin |
| --- | --- | --- |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/__init__.py | f801ff5bb4b8487b45e79a2c98334f14cc94a80fdb096a578e9441f5a2bf7bf6 | arm map |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so | c9b7dc610421010a8b11e8e3af74d1a9ad99928dabb98835aa6c265754daf8ce | arm map |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/apa_sp2.py | 018a26b6bf7acfd85bc23e5084d0367952e1b2c81898da8585c8ecda302a9f07 | arm map |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/functional.py | ef8d1fa2aa2b4f7e818d01c17377ab026d763f690de23fb8831de67f0b131c6e | arm map |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/nn.py | fea952f3ebeb0694044245d6e2fc98acadbdb04ccad9ddd5790fae89be761ca5 | arm map |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/optim.py | 41693c324d5ea2f081ec7777ce8165fbc65d04a7b557fce27ff6d9b9a24457ee | arm map |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/quant.py | 9bb2093df1fa751022400791579afebfe904e7c616d3cd090e0746266829ecd7 | arm map |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/quantization/__init__.py | d29a048409906800e68bbed774978764045451b4451eb2d31ec9d5343293658f | arm map |
| artifacts/gn1/a3/arms/candidate/tensor_cuda/quantization/affine.py | 8bf31258f8635fa4948d08805fb4909ea6a36189d40f0ee2cc589b796c5e3943 | arm map |
| artifacts/gn1/a3/trace_imports.py | 09f38f92907d8ceef6d609621d995b77713271347a8d9525e85da7b3849290ed | import_modules |
| scripts/apa_om135_core.py | cb2f5dd76c9ab7552bc5b8aca7ee0ff4462acc3f05c97156616bdf661336f849 | import_modules |
| tensor_cuda/tests/test_apa_om135_imm.py | 8ac67998194ce0cca0dd5ad507bcb0811e2fb4f4c220dc5ccc1a3419623fd4b3 | import_modules |
| tensor_cuda/tests/test_attention_phase5.py | 23e533cf0b5742cdef46f1a731cbd96c8535e41f2a05da6c6f71a9fdc28961a7 | import_modules |
| tensor_cuda/tests/test_conv_phase3b.py | 2e225ab94c2ded7b559a3c8d8f4d751be91ca848587e41574b17a3f24d8430a8 | import_modules |
| tensor_cuda/tests/test_ext_phase7.py | 8ec82d0438a28057036fca1721c990e19ced5828b5cb57918762326fd7e47455 | import_modules |
| tensor_cuda/tests/test_groupnorm_gn1.py | 0933bed97114a776832790b4045a06640b92097731f114a5913042c1419f49e7 | import_modules |
| tensor_cuda/tests/test_hy3d_engine_ops.py | 55b30170c63711b44f5efc41f4250e3c69b6e969d58378f619576227931efe9d | import_modules |
| tensor_cuda/tests/test_integration.py | ffc947c62c2028890d47030817a2611e6a936c80fffe57f5af8a58115d1bdf1c | import_modules |
| tensor_cuda/tests/test_nn_phase3.py | 2c0a66f3bff5ec2a920d4939fe1d986fc203036d5bb767dbb12693ef0478b1f5 | import_modules |
| tensor_cuda/tests/test_phase9.py | db0de76dc3d1d5d76cbcfcf6d816c77700836b1e143ae5b718a2904318f37f95 | import_modules |
| tensor_cuda/tests/test_rnn_phase5b.py | 0a87495014abe357185eaf5a589666d9e49722cf3bd21895eb170e394ad54f29 | import_modules |

Complete external origins and digests are in import_trace.json and staging_seal.json. `$HOME/` expands only for verification. G4_extra.json demonstrates the parent, child and scorer refusing a changed helper pin; the read-only helper file itself was never edited.
