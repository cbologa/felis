# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Load and dump small configuration objects.

This module provides thin wrappers around YAML or JSON loading and JSON dumping for
configuration dictionaries used across the FELIS workflows.

Notes:
    - Loading is performed with `yaml.safe_load` to avoid executing arbitrary
      Python objects.
    - Dumping is performed with `json.dump` for deterministic, portable output.
"""

import json
import logging
import math
from decimal import Decimal, InvalidOperation
from typing import Any, TextIO

import yaml

logger = logging.getLogger(__name__)


def finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    """Normalize a numeric config value, including YAML's string form of 1e-8."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be a {qualifier} number")
    return result


def finite_int(value: Any, name: str) -> int:
    """Accept exact integral numbers without truncation or boolean coercion."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    if not isinstance(value, (float, str)):
        raise ValueError(f"{name} must be an integer")
    try:
        number = Decimal(value.strip() if isinstance(value, str) else str(value))
    except InvalidOperation as error:
        raise ValueError(f"{name} must be an integer") from error
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"{name} must be a finite integer")
    return int(number)


def numeric_list(value: Any, name: str, converter, *, lengths=None) -> list:
    if not isinstance(value, (list, tuple)) or (lengths is not None and len(value) not in lengths):
        raise ValueError(f"{name} must be a numeric list of the expected length")
    return [converter(item, f"{name}[{index}]") for index, item in enumerate(value)]


def load_config(file_handle: TextIO) -> Any:
    """Load a YAML or JSON configuration from an open file handle.

    Args:
        file_handle (TextIO): Open file-like object positioned at the start of
            a YAML or JSON document.

    Returns:
        Any: Parsed YAML or JSON content. This is typically a mapping for FELIS
            configuration files, but may also be `None` for an empty file.
    """
    return yaml.safe_load(file_handle)


def dump_config(data: dict, file_handle: TextIO, indent: int | None = None) -> None:
    """Dump a configuration mapping as JSON to an open file handle.

    Args:
        data (dict): Configuration mapping to serialize.
        file_handle (TextIO): Open file-like object to write JSON into.
        indent (int): JSON indentation level. Use a negative value (default)
            or `None` to produce compact JSON.

    Raises:
        TypeError: If `data` contains non-JSON-serializable objects.
        OSError: If writing to `file_handle` fails.
    """
    if indent is None or indent < 0:
        json.dump(data, file_handle)
    else:
        json.dump(data, file_handle, indent=indent)
