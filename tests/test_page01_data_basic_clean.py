from __future__ import annotations

import pyarrow as pa

from pages.page01_data_basic_clean import build_clean_comparison_frame


def test_clean_comparison_frame_has_arrow_safe_text_columns():
    frame = build_clean_comparison_frame(
        {
            "raw_rows": 1500,
            "raw_columns": 25,
            "duplicate_rows_removed": 3,
            "clean_rows": 1497,
            "clean_columns": 24,
        }
    )

    table = pa.Table.from_pandas(frame, preserve_index=False)

    assert table.column("Raw").type == pa.string()
    assert table.column("Basic clean").type == pa.string()
    assert frame.loc[0, "Raw"] == "1,500"
    assert frame.loc[4, "Raw"] == "job_id present"
    assert frame.loc[4, "Basic clean"] == "job_id removed"
