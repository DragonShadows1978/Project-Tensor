"""Scoped source edits. Prior art is annotated at each inserted code site."""
from pathlib import Path
p=Path('tensor_cuda/src/apa_spo/eb3.cuh');s=p.read_text()
a=s.index('// ---- persistent merge tickets ');b=s.index('// ---- variant table ',a)
s=s[:a]+'''// ---- persistent per-device tickets (APA-P1-S3 r2b) -------------------------
// Contract: ONE stream, the CUDA legacy default stream (0), on each device.
// A launch on any other stream is out of contract. atomicInc/fence arithmetic
// is inherited unchanged from CUDA Samples threadFenceReduction (NVIDIA).
// Prior art: CUDA stream-ordered lifetime APIs (cudaMallocAsync, NVIDIA 2021)
// and std::mutex/map ownership (ISO C++11, 2011). Taken: stream ordering and
// last-writer tickets; ours: device-keyed merge/choose slots and a launch lock.
// Literature unverified -- lead to check: CUDA Samples threadFenceReduction;
// CUDA 11.2 stream ordered memory allocator 2021. No new reduction.
// APA_P1_HOST_TICKETS_BEGIN (also compiled by the CPU ownership test)
struct TicketSlot { unsigned* buffer=nullptr; int capacity=0; };
struct DeviceTickets { TicketSlot merge,choose; int64_t l2=-1; };
inline std::recursive_mutex& ticket_mutex(){static std::recursive_mutex m;return m;}
inline std::unordered_map<int,DeviceTickets>& device_tickets(){
  // Process lifetime, as before. Do not call CUDA from static destructors.
  static std::unordered_map<int,DeviceTickets> states;return states;
}
class TicketDeviceGuard {
  int previous_=0;bool changed_=false;
public:
  explicit TicketDeviceGuard(int device){
    if(cudaGetDevice(&previous_)!=cudaSuccess)throw std::runtime_error("apa tickets: get device");
    changed_=previous_!=device;
    if(changed_ && cudaSetDevice(device)!=cudaSuccess)throw std::runtime_error("apa tickets: set device");
  }
  ~TicketDeviceGuard() noexcept {if(changed_)cudaSetDevice(previous_);}
  TicketDeviceGuard(const TicketDeviceGuard&)=delete;
  TicketDeviceGuard& operator=(const TicketDeviceGuard&)=delete;
};
inline unsigned* ticket_buffer(bool choose,int n,int minimum,int64_t* bytes){
  std::lock_guard<std::recursive_mutex> lock(ticket_mutex());
  int device=0;
  if(cudaGetDevice(&device)!=cudaSuccess)throw std::runtime_error("apa tickets: get device");
  auto& state=device_tickets()[device];auto& slot=choose?state.choose:state.merge;
  if(n>slot.capacity){
    const int want=std::max(n,minimum);unsigned* next=nullptr;
    if(cudaMalloc(&next,sizeof(unsigned)*(size_t)want)!=cudaSuccess)throw std::runtime_error("apa tickets: allocation");
    if(cudaMemset(next,0,sizeof(unsigned)*(size_t)want)!=cudaSuccess){
      cudaFree(next);throw std::runtime_error("apa tickets: zero");
    }
    if(slot.buffer){
      // Only the CURRENT (guarded operand) device is synchronized. The launch
      // lock remains held from lookup through enqueue, so no caller can hold
      // an old pointer that has not yet been submitted when growth frees it.
      if(cudaDeviceSynchronize()!=cudaSuccess){cudaFree(next);throw std::runtime_error("apa tickets: growth sync");}
      if(cudaFree(slot.buffer)!=cudaSuccess){cudaFree(next);throw std::runtime_error("apa tickets: growth free");}
    }
    slot.buffer=next;slot.capacity=want;
  }
  if(bytes)*bytes=(int64_t)slot.capacity*(int64_t)sizeof(unsigned);
  return slot.buffer;
}
inline unsigned* counters(int n,int64_t* bytes=nullptr){
  return ticket_buffer(false,n,tune::COUNTER_MIN_ENTRIES,bytes);
}
inline int64_t l2_bytes(){
  std::lock_guard<std::recursive_mutex> lock(ticket_mutex());
  int device=0;
  if(cudaGetDevice(&device)!=cudaSuccess)throw std::runtime_error("apa tickets: L2 device");
  auto& cached=device_tickets()[device].l2;
  if(cached<0){
    int value=0;
    if(cudaDeviceGetAttribute(&value,cudaDevAttrL2CacheSize,device)!=cudaSuccess)throw std::runtime_error("apa tickets: L2 size");
    cached=value;
  }
  return cached;
}
// APA_P1_HOST_TICKETS_END

''' + s[b:]
# Lock spans lookup/growth AND enqueue. Device guard spans allocations/launches.
needle='  int np,pk;apa_int4_fixed_partition_layout((int)S,np,pk);'
assert s.count(needle)==1
s=s.replace(needle,'''  // Prior art: CUDA legacy-default-stream ordering and RAII (C++11, 2011).
  // Serialize host submission through growth; no steady-state device sync.
  std::lock_guard<std::recursive_mutex> ticket_launch_lock(eb3::ticket_mutex());
  eb3::TicketDeviceGuard operand_device(q.device.index);
'''+needle)
p.write_text(s)
p=Path('tensor_cuda/src/apa_spo/eb4.cuh');s=p.read_text();a=s.index('// ---- persistent choose tickets ');b=s.index('} // namespace eb4',a)
s=s[:a]+'''// ---- persistent choose tickets, separate from merge on EACH CUDA device ---
// Contract: one stream, CUDA legacy default stream; every other stream is out
// of contract. Prior art: CUDA Samples threadFenceReduction, CUDA stream-ordered
// lifetime APIs (NVIDIA 2021), ISO C++11 mutex/map (2011). Taken: existing
// tickets/order; ours: per-device choose ownership via EB3's locked registry.
// Unverified -- lead to check: CUDA 11.2 stream ordered memory allocator 2021.
inline unsigned* counters(int n,int64_t* bytes=nullptr){
  return eb3::ticket_buffer(true,n,tune::COUNTER_MIN_ENTRIES,bytes);
}
''' + s[b:]
assert s.count(needle)==1
s=s.replace(needle,'''  // Same locked enqueue interval as EB3, including choose and merge slots.
  // Prior art: legacy-default-stream ordering and C++11 RAII (2011).
  std::lock_guard<std::recursive_mutex> ticket_launch_lock(eb3::ticket_mutex());
  eb3::TicketDeviceGuard operand_device(q.device.index);
'''+needle)
p.write_text(s)
print('S1 edited EB3/EB4; device code unchanged')
