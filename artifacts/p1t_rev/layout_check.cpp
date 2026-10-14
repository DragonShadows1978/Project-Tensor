// Prior art: C++ object ABI layout and DLPack ABI. No GPU operations.
#include "tc/core.h"
#include <dlpack/dlpack.h>
#include <cstddef>
#include <iostream>
int main() {
 std::cout << "Storage=" << sizeof(tc::Storage) << " revision=" << offsetof(tc::Storage,revision) << '\n';
 std::cout << "DLTensor=" << sizeof(DLTensor) << " DLManagedTensor=" << sizeof(DLManagedTensor)
 << " DLManagedTensorVersioned=" << sizeof(DLManagedTensorVersioned)
 << " versioned_tensor=" << offsetof(DLManagedTensorVersioned,dl_tensor) << '\n';
}
