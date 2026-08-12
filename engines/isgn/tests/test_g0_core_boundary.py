import subprocess, sys, os


def test_import_core_does_not_import_device_or_benchmark_modules():
    code = r'''
import sys, isgn
blocked=['isgn.bridges.ehss','isgn.real_motor','isgn.simulation','isgn.phm_simulation']
print([x for x in blocked if x in sys.modules])
assert not [x for x in blocked if x in sys.modules]
assert not hasattr(isgn,'load_ehss_scenario')
'''
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True, env={**os.environ, 'PYTHONPATH': str((__import__('pathlib').Path(__file__).resolve().parents[1]/'src')) + os.pathsep + os.environ.get('PYTHONPATH','')})


def test_builtin_adapters_are_lazy_descriptors():
    code = r'''
import sys, isgn
names=isgn.available_adapters()
assert 'ehss' in names and 'euv' in names and 'uci-hydraulic' in names
assert 'isgn.bridges.ehss' not in sys.modules
assert 'isgn.bridges.euv' not in sys.modules
assert 'isgn.bridges.uci_hydraulic' not in sys.modules
'''
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True, env={**os.environ, 'PYTHONPATH': str((__import__('pathlib').Path(__file__).resolve().parents[1]/'src')) + os.pathsep + os.environ.get('PYTHONPATH','')})
