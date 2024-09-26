import logging
from functools import partial
from math import ceil, sqrt
from time import sleep
from typing import Any, Callable, Generator, Iterable, ParamSpec, Type

import pandas as pd
import requests

from nebelung.types import PanderaBaseSchema, T, TypedDataFrame


def list_comprehender(x: list[T], i1: int, i2: int) -> list[T]:
    """
    Perform list comprehension to get a batch of items from a list.

    :param x: the list to extract a batch from
    :param i1: the lower index
    :param i2: the higher index
    :return: a batch from the list
    """

    return x[i1:i2]


def df_comprehender(x: pd.DataFrame, i1: int, i2: int) -> pd.DataFrame:
    """
    Perform list comprehension to get a batch of rows from a data frame.

    :param x: the data frame to extract a batch from
    :param i1: the lower index
    :param i2: the higher index
    :return: a batch from the data frame
    """

    return x.iloc[i1:i2, :]


def batch_evenly(
    items: Iterable[T] | pd.DataFrame, max_batch_size: int
) -> Generator[list[T] | pd.DataFrame, None, None]:
    """
    Yields evenly sized batches from an iterable or data frame such that each batch has
    at most `max_batch_size` items.

    :param items: the iterable or DataFrame to be batched
    :param max_batch_size: the maximum size of each batch
    :return: a generator yielding batches from the input items
    """

    if isinstance(items, pd.DataFrame):
        batchable_items = items
        comprehender = df_comprehender
    else:
        try:
            batchable_items = list(items)
            comprehender = list_comprehender
        except TypeError as e:
            raise TypeError(f"Cannot batch items of type {type(items)}: {e}")

    n_items = len(batchable_items)
    n_batches = 1 + n_items // max_batch_size
    batch_size = n_items / n_batches

    for i in range(n_batches):
        i1 = ceil(i * batch_size)
        i2 = ceil((i + 1) * batch_size)
        yield comprehender(batchable_items, i1, i2)  # pyright:ignore


def type_data_frame(
    df: pd.DataFrame,
    pandera_schema: Type[PanderaBaseSchema],
    remove_unknown_cols: bool = False,
) -> TypedDataFrame[PanderaBaseSchema]:
    """
    Coerce a data frame into one specified by a Pandera schema and optionally remove
    unknown columns.

    :param df: a data frame
    :param pandera_schema: a Pandera schema
    :param remove_unknown_cols: remove columns not specified in the schema
    :return: a data frame validated with the provided Pandera schema
    """

    if len(df) == 0:
        # make an empty data frame that conforms to the Pandera schema
        s = pandera_schema.to_schema()

        # `example` doesn't know how to instantiate dicts, so do that manually
        dict_cols = []

        for c in s.columns:
            if s.columns[c].dtype.type is dict:
                dict_cols.append(c)
                s = s.remove_columns([c])

        df = pd.DataFrame(s.example(size=0))

        if len(dict_cols) > 0:
            for c in dict_cols:
                df[c] = {}

    elif remove_unknown_cols:
        df_cols = pandera_schema.to_schema().columns.keys()
        df = df.loc[:, df_cols]

    return TypedDataFrame[pandera_schema](df)


def expand_dict_columns(
    df: pd.DataFrame,
    except_cols: list[str] | None = None,
    sep: str = "__",
    name_columns_with_parent: bool = True,
    parent_key: str = "",
    col_name_formatter: Callable[[str], str] = lambda _: _,
) -> pd.DataFrame:
    """
    Recursively expand columns in a data frame containing dictionaries into separate
    columns.

    :param df: a data frame
    :param except_cols: an optional list of columns to exclude from expansion
    :param sep: a separator character to use between `parent_key` and its column names
    :param name_columns_with_parent: whether to "namespace" nested column names using
    their parents' column names
    :param parent_key: the name of the parent column, applicable only if
    `name_columns_with_parent` is `True` (for recursion)
    :param col_name_formatter: an optional function to format resulting column names
    :return: a widened data frame
    """

    if except_cols is None:
        except_cols = []

    flattened_dict = {}

    # iterate as (column, series) tuples
    for c, s in df.items():
        # get the index of the first non-NA value in this series (if there is one)
        fvi = s.first_valid_index()

        # check if that first non-NA value is a dict and the column is expandable
        if (
            str(c) not in except_cols
            and fvi is not None
            and isinstance(s.loc[fvi], dict)
        ):
            # recursively flatten this dictionary column
            nested_df = pd.json_normalize(s.tolist(), sep=sep)
            nested_df.index = df.index

            if name_columns_with_parent:
                # e.g. if current column `c` is "foo" and the nested data contains a
                # field "bar", the resulting column name is "foo__bar"

                nested_df.columns = [
                    sep.join(
                        [
                            parent_key,
                            col_name_formatter(str(c)),
                            col_name_formatter(str(col)),
                        ]
                    )
                    if parent_key != ""
                    else sep.join(
                        [col_name_formatter(str(c)), col_name_formatter(str(col))]
                    )
                    for col in nested_df.columns
                ]

            # recurse on the nested data
            flattened_dict.update(
                expand_dict_columns(
                    nested_df,
                    except_cols=except_cols,
                    sep=sep,
                    name_columns_with_parent=name_columns_with_parent,
                    parent_key=col_name_formatter(str(c)),
                )
            )

        else:
            # if not a dictionary, add the column as is
            flattened_dict[c] = s

    df = pd.DataFrame(flattened_dict)

    if parent_key == "":
        # make sure there are no duplicate column names after all expansion is done
        col_name_counts = df.columns.value_counts()

        if col_name_counts.gt(1).any():
            dup_names = set(
                col_name_counts[col_name_counts.gt(1)].index,  # pyright: ignore
            )
            raise NameError(
                f"Column names {dup_names} are duplicated. Try calling "
                "`expand_dict_columns` with `name_columns_with_parent=True`."
            )

    return df


def generalized_fibonacci(n: int, *, f0: float = 1.0, f1: float = 1.0) -> float:
    """
    Calculate the nth number in a generalized Fibonacci sequence given two starting
    nonnegative real numbers. This generates a gradually increasing sequence that
    provides a good balance between linear and exponential functions for use as a
    backoff.

    :param n: the nth Fibonacci number to compute
    :param f0: the first starting value for the sequence
    :param f1: the second starting value for the sequence
    :return: the nth Fibonacci number
    """

    assert f0 >= 0, "f0 must be at least 0.0"
    assert f1 >= 0, "f1 must be at least 0.0"

    # compute constants for closed-form of Fibonacci sequence recurrence relation
    sqrt5 = sqrt(5)
    phi = (1 + sqrt5) / 2
    psi = 1 - phi
    a = (f1 - f0 * psi) / sqrt5
    b = (f0 * phi - f1) / sqrt5

    return max([0, a * phi**n + b * psi**n])


P = ParamSpec("P")


def maybe_retry(
    func: Callable[P, T],
    retryable_exceptions: tuple[Type[Exception], ...] = tuple([Exception]),
    max_retries: int = 0,
    waiter: Callable[..., float] = partial(generalized_fibonacci, f0=1.0, f1=1.0),
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """
    Call a function and optionally retry (at most `max_retries` times) if it raises
    certain exceptions.

    :param func: a function
    :param retryable_exceptions: a tuple of retryable exceptions
    :param max_retries: the maximum number of times to retry (can be 0 if not retrying)
    :param waiter: a function that returns the number of seconds to wait given how many
    tries have already happened
    :param kwargs: keyword arguments to `func`
    :return: the return value from `func`
    """

    if max_retries == 0:
        return func(*args, **kwargs)

    n_retries = 0

    while True:
        try:
            return func(*args, **kwargs)

        except retryable_exceptions as e:
            if n_retries == max_retries:
                raise e

            wait_seconds = round(waiter(n_retries + 1), 1)
            logging.error(f"{e} (retrying in {wait_seconds}s)")
            sleep(wait_seconds)
            n_retries += 1


def call_firecloud_api(
    func: Callable, max_retries: int = 2, *args: Any, **kwargs: Any
) -> Any:
    """
    Call a Firecloud API endpoint and check the response for a valid HTTP status code.

    :param func: a `firecloud.api` method
    :param max_retries: an optional maximum number of times to retry
    :param args: arguments to `func`
    :param kwargs: keyword arguments to `func`
    :return: the API response, if any
    """

    res = maybe_retry(
        func,
        retryable_exceptions=(requests.ConnectionError, requests.ConnectTimeout),
        max_retries=max_retries,
        *args,
        **kwargs,
    )

    if 200 <= res.status_code <= 299:
        try:
            return res.json()
        except requests.JSONDecodeError:
            return res.text

    try:
        raise requests.RequestException(f"HTTP {res.status_code} error: {res.json()}")
    except Exception as e:
        # it's returning HTML or we can't parse the JSON
        logging.error(f"Error getting response as JSON: {e}")
        logging.error(f"Response text: {res.text}")
        raise requests.RequestException(f"HTTP {res.status_code} error")
