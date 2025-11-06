from pathlib import Path

import pandas as pd
import pandera as pa
from pandas._testing import assert_frame_equal
from pandera.typing import Series

from nebelung.types import CoercedDataFrame
from nebelung.utils import parse_workflow_inputs, type_data_frame


class Model(CoercedDataFrame):
    id: Series[pd.Int64Dtype]
    comment: Series[pd.StringDtype] = pa.Field(nullable=True)


class ModelAddMissing(CoercedDataFrame):
    class Config(CoercedDataFrame.Config):
        add_missing_columns = True

    id: Series[pd.Int64Dtype]
    comment: Series[pd.StringDtype] = pa.Field(nullable=True)
    opt: Series[pd.BooleanDtype] = pa.Field(nullable=True, default=False)


class TestTypeDataFrame:
    def test_type_model(self):
        df = pd.DataFrame(
            [
                {"id": 1, "comment": "test"},
                {"id": 2, "comment": None},
            ]
        )

        observed = type_data_frame(
            df, pandera_schema=Model, reorder_cols=False, remove_unknown_cols=False
        )

        expected = pd.DataFrame(
            [
                {"id": 1, "comment": "test"},
                {"id": 2, "comment": None},
            ]
        )
        expected["id"] = expected["id"].astype("Int64")
        expected["comment"] = expected["comment"].astype("string")

        assert_frame_equal(observed, expected)

    def test_type_model_reorder_cols(self):
        df = pd.DataFrame(
            [
                {"comment": "test", "id": 1},
                {"comment": None, "id": 2},
            ]
        )

        observed = type_data_frame(
            df, pandera_schema=Model, reorder_cols=True, remove_unknown_cols=False
        )

        expected = pd.DataFrame(
            [
                {"id": 1, "comment": "test"},
                {"id": 2, "comment": None},
            ]
        )
        expected["id"] = expected["id"].astype("Int64")
        expected["comment"] = expected["comment"].astype("string")

        assert_frame_equal(observed, expected)

    def test_type_model_dont_reorder_cols(self):
        df = pd.DataFrame(
            [
                {"comment": "test", "id": 1},
                {"comment": None, "id": 2},
            ]
        )

        observed = type_data_frame(
            df, pandera_schema=Model, reorder_cols=False, remove_unknown_cols=False
        )

        expected = pd.DataFrame(
            [
                {"comment": "test", "id": 1},
                {"comment": None, "id": 2},
            ]
        )
        expected["id"] = expected["id"].astype("Int64")
        expected["comment"] = expected["comment"].astype("string")

        assert_frame_equal(observed, expected)

    def test_type_model_remove_unknown_cols(self):
        df = pd.DataFrame(
            [
                {"id": 1, "comment": "test", "extra": "a"},
                {"id": 2, "comment": None, "extra": None},
            ]
        )

        observed = type_data_frame(
            df, pandera_schema=Model, reorder_cols=False, remove_unknown_cols=True
        )

        expected = pd.DataFrame(
            [
                {"id": 1, "comment": "test"},
                {"id": 2, "comment": None},
            ]
        )
        expected["id"] = expected["id"].astype("Int64")
        expected["comment"] = expected["comment"].astype("string")

        assert_frame_equal(observed, expected)

    def test_type_model_dont_remove_unknown_cols(self):
        df = pd.DataFrame(
            [
                {"id": 1, "comment": "test", "extra": "a"},
                {"id": 2, "comment": None, "extra": None},
            ]
        )

        observed = type_data_frame(
            df, pandera_schema=Model, reorder_cols=False, remove_unknown_cols=False
        )

        expected = pd.DataFrame(
            [
                {"id": 1, "comment": "test", "extra": "a"},
                {"id": 2, "comment": None, "extra": None},
            ]
        )
        expected["id"] = expected["id"].astype("Int64")
        expected["comment"] = expected["comment"].astype("string")

        assert_frame_equal(observed, expected)

    def test_type_model_reorder_and_remove_unknown_cols(self):
        df = pd.DataFrame(
            [
                {"comment": "test", "id": 1, "extra": "a"},
                {"comment": None, "id": 2, "extra": None},
            ]
        )

        observed = type_data_frame(
            df, pandera_schema=Model, reorder_cols=True, remove_unknown_cols=True
        )

        expected = pd.DataFrame(
            [
                {"id": 1, "comment": "test"},
                {"id": 2, "comment": None},
            ]
        )
        expected["id"] = expected["id"].astype("Int64")
        expected["comment"] = expected["comment"].astype("string")

        assert_frame_equal(observed, expected)

    def test_type_model_add_missing(self):
        df = pd.DataFrame(
            [
                {"id": 1, "comment": "test"},
                {"id": 2, "comment": None},
            ]
        )

        observed = type_data_frame(
            df,
            pandera_schema=ModelAddMissing,
            reorder_cols=False,
            remove_unknown_cols=False,
        )

        expected = pd.DataFrame(
            [
                {"id": 1, "comment": "test", "opt": False},
                {"id": 2, "comment": None, "opt": False},
            ]
        )
        expected["id"] = expected["id"].astype("Int64")
        expected["comment"] = expected["comment"].astype("string")
        expected["opt"] = expected["opt"].astype("boolean")

        assert_frame_equal(observed, expected)


class TestParseWorkflowInputs:
    def test_workflow_inputs(self):
        observed = parse_workflow_inputs(
            Path("tests/nebelung/data/workflow_inputs.json")
        )

        expected = {
            "workflow.negative_decimal": "-1.328",
            "workflow.positive_decimal": "0.15",
            "workflow.small_decimal": "0.00001",
            "workflow.large_decimal": "999999999.999999999",
            "workflow.negative_integer": "-2",
            "workflow.positive_integer": "5",
            "workflow.large_integer": "100000000000",
            "workflow.bool": "false",
            "workflow.string": '"test"',
            "workflow.escaped_string": '"(FILTER=\\\\\\"PASS\\\\\\"|FILTER=\\\\\\"MaxDepth\\\\\\") && (SUM(FORMAT/PR[0:1]+FORMAT/SR[0:1]) >= 5) && (CHROM!=\\\\\\"chrM\\\\\\") && (ALT!~\\\\\\"chrM\\\\\\")"',
            "workflow.array": '"[this.foo, this.bar]"',
        }

        assert observed == expected
