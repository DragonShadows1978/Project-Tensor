"""Check the shipped pytest suite's counts and its GPU module provenance."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

TEST = 'tensor_cuda/tests/test_apa_pf2_nonfinite.py'
INDEX = 'test_actual_diagnostic_tail_helper'
EXPECTED = Counter({
    INDEX: 1,
    'test_nonfinite_sink': 16,
    'test_nonfinite_stored_norm': 16,
    'test_decode_partition_raw_nonfinite_norm': 44,
})


def check(xml, gpu):
    cases = ET.parse(xml).findall('.//testcase')
    names = Counter(case.get('name').split('[', 1)[0] for case in cases)
    assert names == EXPECTED, names
    passed, skipped = [], []
    for case in cases:
        name = case.get('name')
        assert case.find('failure') is None and case.find('error') is None, name
        skip = case.find('skipped')
        if skip is None:
            passed.append(name)
        else:
            assert skip.get('message') == 'CUDA unavailable', (name, skip.attrib)
            skipped.append(name)
    assert (len(passed), len(skipped)) == ((77, 0) if gpu else (1, 76))
    assert INDEX in passed
    return dict(status='PASS', collected=len(cases), passed=len(passed), skipped=len(skipped),
                index_passed=True, skip_reasons={'CUDA unavailable': len(skipped)} if skipped else {})


def main():
    mode, xml = sys.argv[1:3]
    if mode == 'cpu':
        assert 'CUDA_VISIBLE_DEVICES' not in os.environ
        print(json.dumps(check(xml, gpu=False), indent=2))
        return
    assert mode == 'gpu' and os.environ['FORGE_GPU_DRY_RUN'] == '0'
    assert os.environ['APA_PF2F3_LEASE'] == '1' and os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    import pytest
    code = pytest.main(['-q', '-rs', '-p', 'no:cacheprovider', TEST, '--junitxml', xml,
                        '--basetemp', sys.argv[3]])
    assert code == 0, code
    result = check(xml, gpu=True)
    module = Path(sys.modules['tensor_cuda']._C.__file__).resolve()
    assert module == Path(sys.argv[4]).resolve(), module
    with module.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == sys.argv[5]
    result.update(module=str(module), module_sha256=digest)
    with (Path(xml).parent / 'RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
