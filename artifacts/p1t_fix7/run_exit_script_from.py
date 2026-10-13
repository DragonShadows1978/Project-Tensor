import sys, subprocess, os
testfile, pkgroot = sys.argv[1], sys.argv[2]
src = open(testfile).read()
start = src.index("_EXIT_SCRIPT = r'''"); end = src.index("'''.replace('TESTS_DIR'", start)
script = src[start+len("_EXIT_SCRIPT = r'''"):end].replace('TESTS_DIR', repr(pkgroot + '/tc_signal/tests'))
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=pkgroot + '/tc_signal')
r = subprocess.run([sys.executable, '-c', script, 'deleter'], capture_output=True, text=True, env=env, timeout=60)
out = r.stdout
print('rc', r.returncode, 'ordered', 0 <= out.find('CALLBACK finished') < out.find('FINALIZING True'), 'garbage-name' if 'is not a DLPack capsule' in r.stderr else '')
