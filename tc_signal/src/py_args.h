// P1T-FIX6, sweep (g) "user Python run from inside a native frame": how every _tc_signal binding receives its
// arguments.
//
// Defect class (round-6 review V6-02 found it in core; this sweep found the same class in every typed parameter of the
// add-on). pybind11's typed argument casters run conversions in the dispatcher, before the function body: an int64_t
// parameter calls a foreign object's __index__, a double its __float__ / __index__, a bool its __bool__, a
// std::vector its __len__ / __getitem__ and each item's conversion. That user Python runs on a native stack. If it
// releases the GIL (time.sleep, I/O) on a daemon thread while the interpreter starts finalizing, CPython 3.12 exits the
// thread when it re-takes the GIL (pthread_exit, a forced unwind). The unwind runs the dispatcher's C++ cleanup: when a
// keyword argument filled a named parameter, the dispatcher holds the only reference to its private copy of the kwargs
// dict (pybind11 v2.12 pybind11.h:880-883, PyDict_Copy), and that dict is deallocated with the exiting thread's state
// already freed: SIGSEGV (receipt: scratch experiment `_C.empty(shape=SeqLen())` rc 139 2/2 on acd42fa, positional
// rc 0; ledger 07:32-07:40). pybind11's own failure paths run user Python too: when no overload matches it builds its
// TypeError with repr() of every argument (pybind11.h:1122-1160), and it looks keyword names up with dict lookups that
// may call a str subclass's __eq__.
//
// The fix (refusal before invocation, the order's option (ii), for every binding of this module): every binding is
// declared as (*args, **kwargs). For that signature the dispatcher copies no argument, looks up no keyword name, has no
// overload to mismatch, and never converts (receipt: scratch module /tmp/p1t_fix6_scratch/pbtest; a method's self
// arrives as args[0]). CallArgs then matches positions and keyword names with C-level comparisons only
// (PyUnicode_CompareWithASCIIString reads the string's data, also for a str subclass: no __eq__, no __hash__), and the
// converters below accept exactly the types whose value can be read without calling any Python-level method:
//   * int parameters: int and its subclasses (bool, IntEnum) -- PyLong_AsLongLongAndOverflow reads a PyLong_Check
//     object's value directly (CPython 3.12 Objects/longobject.c: _PyNumber_Index only for non-ints), so an
//     overridden __index__ / __int__ is never called; out of the int64 range -> OverflowError;
//   * real parameters: float and its subclasses (PyFloat_AsDouble reads PyFloat_Check objects directly), int and its
//     subclasses (PyLong_AsDouble, likewise); anything else -> TypeError;
//   * flags: bool, or an int (nonzero = true, read as above); anything else -> TypeError (no __bool__);
//   * string lists: a list or tuple (PyList_GET_ITEM / PyTuple_GET_ITEM, which ignore a subclass's __len__ /
//     __getitem__) of str (PyUnicode_AsUTF8AndSize reads the data);
//   * operands: passed through untouched as py::object; every Python call on them happens inside to_capsule's admitted
//     producer-callback section, which tc_signal's exit hook waits for (P1T-FIX5).
// A refusal names the parameter and the argument's type (tp_name, a C string): no repr(), no __str__, no __format__.
// The public Python wrappers (tc_signal.conv1d, lstm, norms, small ops) already converted their arguments in Python
// frames (operator.index, float), so they pass exact ints and floats and behave exactly as before.
//
// Prior art: CPython's own C argument parser (Python/getargs.c, keyword matching by identity, then a C-level string
// comparison, no __eq__; unverified here, lead to check "CPython getargs.c find_keyword _PyUnicode_Equal"); PyTorch
// 2.11's Tensor.__dlpack__, which refuses a non-int stream with `type(stream) is not int` -> TypeError instead of
// converting it (read in the installed torch/_tensor.py); the Python array API's `__dlpack__` signature (stream: int |
// None, max_version: tuple[int, int] | None, dl_device: tuple | None, copy: bool | None; Consortium for Python Data API
// Standards, 2021-2023); pybind11's py::args / py::kwargs (Jakob et al., 2015-). Taken: C-level argument matching and
// type refusal instead of conversion. Ours: applying it to every binding so that no user Python can run inside the
// dispatcher or a binding frame of this module.
#pragma once

#include <pybind11/pybind11.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace tcs {
namespace py = pybind11;

struct Param {
  const char* name;
  bool required = true;  // false: may be omitted (the binding supplies the default)
};

class CallArgs {
 public:
  static constexpr size_t kMaxParams = 16;

  // `positional_max`: how many leading parameters may be given by position (the rest are keyword-only).
  CallArgs(const char* fn, const py::args& args, const py::kwargs& kwargs, std::initializer_list<Param> params,
           size_t positional_max = kMaxParams)
      : fn_(fn) {
    if (params.size() > kMaxParams) throw std::logic_error("tc_signal: CallArgs: too many parameters");
    for (const Param& p : params) params_[n_params_++] = p;
    PyObject* a = args.ptr();
    const size_t n_pos = static_cast<size_t>(PyTuple_GET_SIZE(a));
    const size_t pos_ok = positional_max < n_params_ ? positional_max : n_params_;
    if (n_pos > pos_ok)
      throw py::type_error(prefix() + "takes at most " + std::to_string(pos_ok) + " positional argument(s) (" +
                           std::to_string(n_pos) + " given)");
    for (size_t i = 0; i < n_pos; ++i) vals_[i] = PyTuple_GET_ITEM(a, static_cast<Py_ssize_t>(i));
    PyObject* k = kwargs.ptr();
    if (k != nullptr && PyDict_Check(k)) {
      Py_ssize_t pos = 0;
      PyObject* key = nullptr;
      PyObject* value = nullptr;
      while (PyDict_Next(k, &pos, &key, &value)) {
        if (!PyUnicode_Check(key)) throw py::type_error(prefix() + "keywords must be strings");
        size_t j = 0;
        while (j < n_params_ && PyUnicode_CompareWithASCIIString(key, params_[j].name) != 0) ++j;
        if (j == n_params_) throw py::type_error(prefix() + "got an unexpected keyword argument '" + utf8(key) + "'");
        if (vals_[j] != nullptr)
          throw py::type_error(prefix() + "got multiple values for argument '" + params_[j].name + "'");
        vals_[j] = value;
      }
    }
    for (size_t j = 0; j < n_params_; ++j)
      if (params_[j].required && vals_[j] == nullptr)
        throw py::type_error(prefix() + "missing required argument '" + params_[j].name + "' (position " +
                             std::to_string(j) + ")");
  }

  bool given(size_t i) const { return vals_[i] != nullptr; }
  PyObject* raw(size_t i) const { return vals_[i]; }
  // The argument as a new reference, or None when an optional parameter was omitted. No conversion.
  py::object object(size_t i) const {
    return vals_[i] ? py::reinterpret_borrow<py::object>(vals_[i]) : py::none();
  }
  [[noreturn]] void refuse(size_t i, const char* expected) const {
    throw py::type_error(prefix() + "argument '" + params_[i].name + "' must be " + expected + ", not " +
                         Py_TYPE(vals_[i])->tp_name +
                         " (P1T-FIX6: refused without calling any of its Python methods)");
  }
  std::string prefix() const { return std::string("tc_signal._C.") + fn_ + "(): "; }
  const char* name(size_t i) const { return params_[i].name; }

 private:
  static std::string utf8(PyObject* s) {
    Py_ssize_t n = 0;
    const char* p = PyUnicode_AsUTF8AndSize(s, &n);  // reads the data; a lone surrogate fails (no Python called)
    if (p == nullptr) {
      PyErr_Clear();
      return "?";
    }
    return std::string(p, static_cast<size_t>(n));
  }
  const char* fn_;
  Param params_[kMaxParams] = {};
  PyObject* vals_[kMaxParams] = {};
  size_t n_params_ = 0;
};

// int and its subclasses only; value read without calling __index__ / __int__ (see above).
inline int64_t arg_int64(const CallArgs& c, size_t i) {
  PyObject* o = c.raw(i);
  if (!PyLong_Check(o)) c.refuse(i, "an int");
  int overflow = 0;
  const long long v = PyLong_AsLongLongAndOverflow(o, &overflow);
  if (overflow != 0)
    throw std::overflow_error(c.prefix() + "argument '" + c.name(i) + "' does not fit in a signed 64-bit integer");
  if (v == -1 && PyErr_Occurred()) throw py::error_already_set();  // not reached for a PyLong (kept for safety)
  return static_cast<int64_t>(v);
}

inline int arg_int32(const CallArgs& c, size_t i) {
  const int64_t v = arg_int64(c, i);
  if (v < INT_MIN || v > INT_MAX)
    throw std::overflow_error(c.prefix() + "argument '" + c.name(i) + "' does not fit in a C int");
  return static_cast<int>(v);
}

// float (and subclasses) or int (and subclasses); read without calling __float__ / __index__.
inline double arg_real(const CallArgs& c, size_t i) {
  PyObject* o = c.raw(i);
  if (PyFloat_Check(o)) return PyFloat_AsDouble(o);
  if (PyLong_Check(o)) {
    const double v = PyLong_AsDouble(o);
    if (v == -1.0 && PyErr_Occurred()) throw py::error_already_set();  // OverflowError for a huge int
    return v;
  }
  c.refuse(i, "a real number (float or int)");
}

// bool, or an int read as nonzero (no __bool__ is called).
inline bool arg_flag(const CallArgs& c, size_t i) {
  PyObject* o = c.raw(i);
  if (o == Py_True) return true;
  if (o == Py_False) return false;
  if (PyLong_Check(o)) {
    int overflow = 0;
    const long long v = PyLong_AsLongLongAndOverflow(o, &overflow);
    return overflow != 0 || v != 0;
  }
  c.refuse(i, "a bool");
}

// A list or tuple of str; items read with the C macros (a subclass's __len__ / __getitem__ is never called).
inline std::vector<std::string> arg_str_list(const CallArgs& c, size_t i) {
  PyObject* o = c.raw(i);
  const bool is_list = PyList_Check(o);
  if (!is_list && !PyTuple_Check(o)) c.refuse(i, "a list or tuple of str");
  const Py_ssize_t n = is_list ? PyList_GET_SIZE(o) : PyTuple_GET_SIZE(o);
  std::vector<std::string> out;
  out.reserve(static_cast<size_t>(n));
  for (Py_ssize_t j = 0; j < n; ++j) {
    PyObject* s = is_list ? PyList_GET_ITEM(o, j) : PyTuple_GET_ITEM(o, j);
    if (!PyUnicode_Check(s)) c.refuse(i, "a list or tuple of str");
    Py_ssize_t len = 0;
    const char* p = PyUnicode_AsUTF8AndSize(s, &len);
    if (p == nullptr) throw py::error_already_set();
    out.emplace_back(p, static_cast<size_t>(len));
  }
  return out;
}

// P1T-FIX6: the argument of a RAW CPython function (METH_VARARGS | METH_KEYWORDS) whose body waits with the GIL released
// (tc_signal's exit wait, _exit_close_admission). Through pybind11 such a body always has a Python object owned only by
// the dispatcher on the stack (a (*args, **kwargs) binding called positionally gets a NEW empty kwargs dict, pybind11.h
// step 4b; a typed binding called with a keyword gets a PyDict_Copy, step 2); if CPython exits the waiting thread (a
// non-main thread re-taking the GIL after finalization began) the unwind deallocates it with the thread's state freed
// (core's version of this wait: rc -11 in 3 of 5 runs, ledger 08:40-09:05). A raw function's args / kwargs are owned
// by the caller's frame. The value is read at C level, as above. Returns false with a Python error set.
inline bool raw_one_real_arg(const char* fn, const char* name, PyObject* args, PyObject* kwargs, double* out) {
  PyObject* v = nullptr;
  const Py_ssize_t n = args ? PyTuple_GET_SIZE(args) : 0;
  if (n > 1) {
    PyErr_Format(PyExc_TypeError, "%s(): takes at most 1 positional argument (%zd given)", fn, n);
    return false;
  }
  if (n == 1) v = PyTuple_GET_ITEM(args, 0);
  if (kwargs != nullptr && PyDict_Check(kwargs)) {
    Py_ssize_t pos = 0;
    PyObject* key = nullptr;
    PyObject* value = nullptr;
    while (PyDict_Next(kwargs, &pos, &key, &value)) {
      if (!PyUnicode_Check(key)) {
        PyErr_Format(PyExc_TypeError, "%s(): keywords must be strings", fn);
        return false;
      }
      if (PyUnicode_CompareWithASCIIString(key, name) != 0) {  // %U copies the str's data: no __str__ is called
        PyErr_Format(PyExc_TypeError, "%s(): got an unexpected keyword argument '%U'", fn, key);
        return false;
      }
      if (v != nullptr) {
        PyErr_Format(PyExc_TypeError, "%s(): got multiple values for argument '%s'", fn, name);
        return false;
      }
      v = value;
    }
  }
  if (v == nullptr) {
    PyErr_Format(PyExc_TypeError, "%s(): missing required argument '%s'", fn, name);
    return false;
  }
  if (PyFloat_Check(v)) {
    *out = PyFloat_AsDouble(v);
    return true;
  }
  if (PyLong_Check(v)) {
    *out = PyLong_AsDouble(v);
    return !(*out == -1.0 && PyErr_Occurred());
  }
  PyErr_Format(PyExc_TypeError,
               "%s(): argument '%s' must be a real number (float or int), not %s (P1T-FIX6: refused without calling "
               "any of its Python methods)",
               fn, name, Py_TYPE(v)->tp_name);
  return false;
}

}  // namespace tcs
