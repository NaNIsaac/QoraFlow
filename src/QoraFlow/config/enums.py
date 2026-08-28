"""
This module defines enumerations for various experiment configurations.
It includes enums for detectors, hutch settings, X-ray types, and Hertz settings,
providing a structured approach to handle these configuration options.

Classes:
    - `Detector`: Enum representing different types of detectors.
    - `Hutch`: Enum representing different hutch settings.
    - `Xray`: Enum representing different X-ray types.
    - `Hertz`: Enum representing different Hertz settings.

Each enum class provides a string representation of the enum value.
"""

from enum import Enum


class Detector(Enum):
    """
    Enum representing different types of detectors.

    To add a new detector, follow these steps:

    1. Add the detector name in all uppercase:
        <DETECTOR_NAME> = '<detector_name>'
    """

    JUNGFRAU1 = "jungfrau1"
    JUNGFRAU2 = "jungfrau2"


class Hutch(Enum):
    """
    Enum representing different hutch settings.

    To add a new hutch, follow these steps:

    1. Add the hutch name in all uppercase:
        <HUTCH_NAME> = '<hutch_name>'
    """

    EH1 = "eh1"
    EH2 = "eh2"


class Xray(Enum):
    """
    Enum representing different X-ray types.
    """

    SOFT = "SX"
    HARD = "HX"


class Hertz(Enum):
    """
    Enum representing different Hertz settings.
    """

    ZERO = "0HZ"
    TEN = "10HZ"
    FIFTEEN = "15HZ"
    TWENTY = "20HZ"
    THIRTY = "30HZ"
    SIXTY = "60HZ"
