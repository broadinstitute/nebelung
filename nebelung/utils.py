import json
import logging
import random
from decimal import Decimal
from functools import partial
from math import ceil, sqrt
from os import PathLike
from time import sleep
from typing import Any, Callable, Dict, Generator, Iterable, ParamSpec, Type

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
    n_batches = ceil(n_items / max_batch_size)
    batch_size = n_items / n_batches

    for i in range(n_batches):
        i1 = ceil(i * batch_size)
        i2 = min(ceil((i + 1) * batch_size), n_items)
        yield comprehender(batchable_items, i1, i2)  # pyright:ignore


def type_data_frame(
    df: pd.DataFrame,
    pandera_schema: Type[PanderaBaseSchema],
    reorder_cols: bool = True,
    remove_unknown_cols: bool = False,
) -> TypedDataFrame[PanderaBaseSchema]:
    """
    Coerce a data frame into one specified by a Pandera schema and optionally reorder
    columns and remove unknown columns.

    :param df: a data frame
    :param pandera_schema: a Pandera schema
    :param reorder_cols: reorder columns as specified in the schema
    :param remove_unknown_cols: remove columns not specified in the schema
    :return: a data frame validated with the provided Pandera schema
    """

    if len(df) == 0:
        # make an empty data frame that conforms to the Pandera schema
        s = pandera_schema.to_schema()

        # `example` doesn't know how to instantiate columns with structured data
        dict_cols = []
        list_cols = []

        for c in s.columns:
            if s.columns[c].dtype.type is dict:
                dict_cols.append(c)
                s = s.remove_columns([c])
            elif s.columns[c].dtype.type is list:
                list_cols.append(c)
                s = s.remove_columns([c])

        df = pd.DataFrame(s.example(size=1))

        if len(dict_cols) > 0:
            for c in dict_cols:
                df[c] = [{}] * len(df)

        if len(list_cols) > 0:
            for c in list_cols:
                df[c] = [[]] * len(df)

        df = df.iloc[:0]
        return TypedDataFrame[pandera_schema](df)

    if not remove_unknown_cols and not reorder_cols:
        # can type and return
        return TypedDataFrame[pandera_schema](df)

    # we need to collect the current columns and schema columns (in original orders)
    current_cols = list(df.columns)
    schema_cols = list(pandera_schema.to_schema().columns.keys())

    if remove_unknown_cols:
        # drop excess columns (if any)
        excess_cols = list(set(current_cols) - set(schema_cols))

        if len(excess_cols) > 0:
            df = df.drop(columns=excess_cols)
            current_cols = list(df.columns)

    # `df` might contain extra columns, but we can still type it now
    df = TypedDataFrame[pandera_schema](df)

    if reorder_cols:
        # put columns in schema order, with extra columns in original order at the end
        all_cols = schema_cols.copy()
        all_cols.extend(current_cols)
        all_cols = list(dict.fromkeys(all_cols))
        df = df.loc[:, all_cols]

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
    jitter: bool = True,
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
    :param jitter: whether to add random jitter to the number of seconds to wait
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
                raise

            wait_seconds = waiter(n_retries + 1)

            if jitter:
                wait_seconds += random.uniform(wait_seconds / -2, wait_seconds / 2)

            logging.error(f"{e} (retrying in {round(wait_seconds, 1)}s)")
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


def parse_workflow_inputs(path: PathLike) -> Dict[str, Any]:
    """
    Parse a workflow inputs JSON file into a dictionary with formatted values suitable
    for upload as a Terra method config.

    :param path: Path to the JSON file containing workflow inputs.
    :return: Dictionary mapping input names to formatted string values.
    """

    def custom_object_hook(obj: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        for k, v in obj.items():
            if isinstance(v, str):
                result[k] = '"' + json.dumps(v)[1:-1] + '"'
            elif isinstance(v, Decimal):
                result[k] = str(v)
            else:
                result[k] = json.dumps(v)

        return result

    with open(path, "r") as f:
        return json.load(f, object_hook=custom_object_hook, parse_float=Decimal)
