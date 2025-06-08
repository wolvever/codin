import pytest
from codin.sandbox.local import LocalSandbox

@pytest.mark.asyncio
async def test_run_c_code(tmp_path):
    sandbox = LocalSandbox(workdir=str(tmp_path))
    await sandbox.up()
    try:
        code = "#include <stdio.h>\nint main(){printf(\"hi\\n\");return 0;}"
        result = await sandbox.run_code(code, language='c')
        assert result.exit_code == 0
        assert 'hi' in result.stdout
    finally:
        await sandbox.down()
