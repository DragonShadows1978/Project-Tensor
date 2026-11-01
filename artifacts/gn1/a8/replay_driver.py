"""Final-seal G4 foreground replay. Prior art: A3 bounded gate driver."""
import os
from pathlib import Path
import subprocess
import sys
TOOL=Path(__file__).resolve().parent
ROOT=TOOL.parents[2]
for gate,script,args,bound in (
    ('hostile_gate','hostile_scorer.py',[],600),
    ('runner_gate','runner_controls.py',[],400),
    ('rev2_gate','rev2_controls.py',['a','c','d'],600),
    ('extra_gate','extra_controls.py',[],220),
    ('a4_controls_gate','a4_controls.py',[],580)):
    result=subprocess.run(['timeout',str(bound+10)+'s',sys.executable,'-B',str(TOOL/'run_cpu_gate.py'),
                           gate,str(bound),sys.executable,'-B',str(TOOL/script),*args],cwd=ROOT,env=os.environ)
    assert result.returncode==0,gate
