"""Non-string allocation/conversion ordering. Prior art: boundary/spying tests (no specific prior art known to me); original ENG2 F1 unconvertible operand pattern. No CUDA call is permitted by the C++ boundary sentinel."""
from pathlib import Path
from collections import UserString
import json,sys
import numpy as np
ROOT=Path.cwd();OUT=ROOT/"artifacts/eng2/rev1";sys.path.insert(0,str(ROOT/"tensor_cuda"));import tensor_cuda as tc
class Boundary(Exception):pass
class Unconvertible:
 def __array__(self,*a,**kw):raise AssertionError("data conversion reached")
rows=[];original=tc._C.tensor
try:
 def boundary(*a,**kw):raise Boundary("C++ boundary reached; call suppressed")
 tc._C.tensor=boundary
 for name in ["randn","rand"]:
  np.random.seed(2214);before=np.random.get_state();assert np.any(before[1]!=0) and np.ptp(before[1])>0
  try:getattr(tc,name)(2,3,dtype=UserString("float32"))
  except Boundary as e:exception=str(e)
  after=np.random.get_state();assert np.any(after[1]!=0) and np.ptp(after[1])>0
  changed=not(np.array_equal(before[1],after[1]) and before[2:]==after[2:]);assert changed
  rows.append(dict(entry=name,non_string="UserString(float32)",random_state_changed=changed,result=exception))
 try:tc.tensor(Unconvertible(),dtype=UserString("float32"))
 except AssertionError as e:rows.append(dict(entry="tensor",non_string="UserString(float32)",result=str(e)))
 else:raise AssertionError("conversion probe did not execute")
finally:tc._C.tensor=original
(OUT/"nonstring_random_review.json").write_text(json.dumps(rows,indent=2)+"\n");print(json.dumps(rows,indent=2))
