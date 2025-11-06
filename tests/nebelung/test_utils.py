import pandas as pd
import pandera as pa
from pandas._testing import assert_frame_equal
from pandera.typing import Series

from nebelung.types import CoercedDataFrame
from nebelung.utils import type_data_frame


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
