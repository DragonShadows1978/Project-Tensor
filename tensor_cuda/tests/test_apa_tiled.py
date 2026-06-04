"""Superseded by test_apa_selective.py.

The tiled-blend approach was abandoned in favor of the fused sparse selective
kernel (apa_selective_attention), which is what actually extends context. See
test_apa_selective.py for the validated selective-APA tests.
"""
import pytest

pytestmark = pytest.mark.skip(reason="superseded by test_apa_selective.py")
