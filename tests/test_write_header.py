from __future__ import annotations

import pytest

from PAOFLOW_QTpy.write_header import write_header


class TestWriteHeader:

    def test_long_messages(self):

        msg = "*" * 80

        with pytest.raises(ValueError) as _:
            write_header(msg)

    @pytest.mark.parametrize("msg", ["hello", "goodbye"])
    def test_output(self, capfd, msg):

        write_header(msg)

        out, _ = capfd.readouterr()
        lines = out.split("\n")

        for line, expected in zip(lines, self._get_expected_output(msg)):
            assert line == expected

    def _get_expected_output(self, msg: str) -> tuple[str, ...]:
        return (
            f"  {'='*70}",
            f"  =  {msg:^66s}=",
            f"  {'='*70}",
        )
