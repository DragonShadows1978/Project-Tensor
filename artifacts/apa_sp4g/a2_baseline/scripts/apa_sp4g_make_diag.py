"""Separate instrumented literal kernel copies. Prior art: SP3 diagnostics
(2026), Perry APA z-tail (2026), BLASST/Yuan2025 running-max, Dao2023 FA2,
Milakov/Gimelshein2018 online softmax. Observation only; no production edits.
"""
from apa_sp4g_common import R,BUILD,verify_sources
# Import ONLY the source extractor/generator; substitute its verification scope.
import apa_sp3_make_diag as base
base.verify_sources=verify_sources

def generated():
    s=base.generated().replace('apa_sp3','apa_sp4g')
    s=s.replace('float qr[4]','float qr[16]').replace('t<4','t<16')
    s=s.replace('(((int64_t)b*H+h)*S+j)*D','((int64_t)b*S+j)*D')
    s=s.replace('apa_selective_kernel<T,128,true>','apa_selective_kernel<T,512,true>')
    s=s.replace('scale,z,causal,H,1,','scale,z,causal,1,H,')
    # Literal actual B bulk score, stored alongside its original mask.
    s=s.replace('int KVH, int group, uint8_t* selected)', 'int KVH, int group, uint8_t* selected, float* bscores)')
    s=s.replace('if (lane == 0) selected[(int64_t)row*S+j] = fabsf(bulk)>=thr;', 'if (lane == 0) { selected[(int64_t)row*S+j] = fabsf(bulk)>=thr; bscores[(int64_t)row*S+j]=bulk; }')
    s=s.replace('std::pair<NDArray,NDArray> selective(', 'std::tuple<NDArray,NDArray,NDArray> selective(')
    s=s.replace('NDArray o({B,H,L,VD},q.dtype,q.device),m({B,H,L,S},DType::Uint8,q.device);','NDArray o({B,H,L,VD},q.dtype,q.device),m({B,H,L,S},DType::Uint8,q.device),bs({B,H,L,S},DType::Float32,q.device);')
    s=s.replace('scale,z,causal,1,H,(uint8_t*)m.data_ptr());','scale,z,causal,1,H,(uint8_t*)m.data_ptr(),(float*)bs.data_ptr());')
    s=s.replace('cuda_check_last("apa_sp4g_selective_diagnostic"); return {o,m};','cuda_check_last("apa_sp4g_selective_diagnostic"); return {o,m,bs};')
    return '#include <tuple>\n'+s
if __name__=='__main__':
    BUILD.mkdir(parents=True,exist_ok=True)
    with (BUILD/'diagnostics.cu').open('x') as f:f.write(generated())
    s=(R/'scripts/apa_sp3_diag_bindings.cpp').read_text().replace('apa_sp3','apa_sp4g').replace('SP3','SP4G').replace('!=96','!=512')
    s=s.replace('qs[1]!=ks[1]','(qs[1]!=16 || ks[1]!=1)')
    s=s.replace('std::pair<NDArray,NDArray> selective','std::tuple<NDArray,NDArray,NDArray> selective')
    s=s.replace('return py::make_tuple(Tensor::make(o.first,false),Tensor::make(o.second,false));\n });\n}', 'return py::make_tuple(Tensor::make(std::get<0>(o),false),Tensor::make(std::get<1>(o),false),Tensor::make(std::get<2>(o),false));\n });\n}')
    with (BUILD/'diag_bindings.cpp').open('x') as f:f.write('#include <tuple>\n'+s)
