from io import StringIO

from antio.utils.config import sys_info


def test_sys_info():
    """Test info-showing utility."""
    out = StringIO()
    sys_info(fid=out)
    value = out.getvalue()
    out.close()
    assert "Platform:" in value
    assert "Executable:" in value
    assert "CPU:" in value
    assert "Physical cores:" in value
    assert "Logical cores" in value
    assert "RAM:" in value
    assert "SWAP:" in value

    assert "numpy" in value
    assert "psutil" in value

    assert "ruff" not in value
    assert "pytest" not in value

    out = StringIO()
    sys_info(fid=out, developer=True)
    value = out.getvalue()
    out.close()

    assert "ruff" in value
    assert "pytest" in value
