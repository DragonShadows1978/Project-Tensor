// PASS iff aggregate retained-byte accounting does not overflow on individually validated metadata.
// Prior art: C++17 signed-overflow rules; GCC UBSan (2013+); checked addition (GCC5,2015).
// Taken: unchanged actual Operand, validation and note_push; ours: two fp32 descriptor boundary cases.
// No CUDA call, allocation or kernel. Synthetic descriptors do not establish real resident GPU allocations.
#include <pybind11/embed.h>
#include <iostream>
#include "dlpack_operand.cpp"
int main(int argc,char**argv) {
 pybind11::scoped_interpreter python{};
 int64_t shape=(int64_t{1}<<61)-1;
 DLManagedTensor x{},y{};
 for(auto* m:{&x,&y}) {m->dl_tensor.shape=&shape;m->dl_tensor.ndim=1;m->dl_tensor.dtype={kDLFloat,32,1};m->dl_tensor.device={kDLCUDA,0};}
 x.dl_tensor.data=reinterpret_cast<void*>(uintptr_t{4});
 y.dl_tensor.data=reinterpret_cast<void*>(uintptr_t{1}<<63);
 auto cap=[](DLManagedTensor* m){return pybind11::reinterpret_steal<pybind11::object>(PyCapsule_New(m,"dltensor",nullptr));};
 tcs::Operand X(cap(&x),"x",1,false),Y(cap(&y),"y",1,true);
 tcs::expect_disjoint(Y,X);
 std::cout<<"validated disjoint operands, bytes each="<<X.numel()*int64_t{4}<<std::endl;
 tcs::PendingEntry e;
 if(argc>1 && std::string(argv[1])=="queue") {
   tcs::PendingState st; st.bytes=4; st.managers=1;
   e.bytes=X.numel()*int64_t{4};
   tcs::note_push(st,e); // unchanged production function, actual pending-list accounting
   std::cout<<st.bytes<<std::endl;
 } else {
   // Verbatim arithmetic statement from RetainUntilComplete destructor, dlpack_operand.cpp:731.
   for(tcs::Operand* op:{&X,&Y}) {
     e.bytes += op->numel() * static_cast<int64_t>(sizeof(float));
   }
   std::cout<<e.bytes<<std::endl;
 }
}
