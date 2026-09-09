#include <tuple>
// Prior art: pybind11 (Jakob et al., 2016) C++/Python bindings, existing
// Project-Tensor Tensor ABI. SP4G adds checked, separate diagnostic wrappers.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tc/autograd.h"
#include <cmath>
namespace py=pybind11;
using namespace tc;
namespace apa_sp4g {
NDArray bulk_scores(const NDArray&,const NDArray&,float);
std::pair<NDArray,NDArray> blend(const NDArray&,const NDArray&,float,int,int);
std::tuple<NDArray,NDArray,NDArray> selective(const NDArray&,const NDArray&,const NDArray&,const NDArray&,float,float,bool);
}
static void check(Tensor& t) {
 const auto& a=t.data();
 if(a.ndim()!=4 || !a.device.is_cuda() ||
    (a.dtype!=DType::Float32 && a.dtype!=DType::Float16 && a.dtype!=DType::BFloat16))
   throw std::runtime_error("SP4G diagnostics need rank4 CUDA float tensors");
 for(auto d:a.shape) if(d<=0 || d>32768) throw std::runtime_error("SP4G diagnostic extent");
}
PYBIND11_MODULE(_apa_sp4g_diag,m) {
 m.def("bulk_scores",[](Tensor& q,Tensor& kq,double scale){
   check(q);check(kq);
   auto qs=q.shape(),ks=kq.shape();
   if(qs[3]!=512 || ks[3]!=512 || qs[0]!=ks[0] || (qs[1]!=16 || ks[1]!=1) ||
      q.data().dtype!=kq.data().dtype || q.data().device.index!=kq.data().device.index ||
      !std::isfinite(scale) || scale<=0)
     throw std::runtime_error("SP4G bulk score contract");
   return Tensor::make(apa_sp4g::bulk_scores(q.data(),kq.data(),scale),false);
 });
 m.def("blend",[](Tensor& b,Tensor& r,double z,int Lq,int row0){
   check(b); check(r);
   if(b.shape()!=r.shape() || b.data().dtype!=r.data().dtype ||
      b.data().device.index!=r.data().device.index || !std::isfinite(z) ||
      Lq<0 || row0<0 || (Lq>0 && row0+b.shape()[2]>Lq))
     throw std::runtime_error("SP4G blend contract");
   auto o=apa_sp4g::blend(b.data(),r.data(),z,Lq,row0);
   return py::make_tuple(Tensor::make(o.first,false),Tensor::make(o.second,false));
 });
 m.def("selective",[](Tensor& q,Tensor& k,Tensor& kq,Tensor& v,double scale,double z,bool causal){
   for(auto t:{&q,&k,&kq,&v}) check(*t);
   auto qs=q.shape(),ks=k.shape(),vs=v.shape();
   if(qs[3]!=512 || vs[3]!=512 || ks!=kq.shape() || ks[3]!=512 ||
      qs[0]!=ks[0] || (qs[1]!=16 || ks[1]!=1) || vs[0]!=ks[0] || vs[1]!=ks[1] || vs[2]!=ks[2] ||
      qs[2]<=1 || (causal && ks[2]<qs[2]) || !std::isfinite(scale) || scale<=0 || !std::isfinite(z))
     throw std::runtime_error("SP4G selective requires D=VD=96 MHA prefill");
   for(auto t:{&k,&kq,&v}) if(t->data().dtype!=q.data().dtype || t->data().device.index!=q.data().device.index)
     throw std::runtime_error("SP4G diagnostic dtype/device mismatch");
   auto o=apa_sp4g::selective(q.data(),k.data(),kq.data(),v.data(),scale,z,causal);
   return py::make_tuple(Tensor::make(std::get<0>(o),false),Tensor::make(std::get<1>(o),false),Tensor::make(std::get<2>(o),false));
 });
}
