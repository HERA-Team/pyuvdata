# -*- mode: python; coding: utf-8 -*-
# Copyright (c) 2018 Radio Astronomy Software Group
# Licensed under the 2-clause BSD License

"""
Base class for objects with UVParameter attributes.

Subclassed by UVData and Telescope.
"""
import copy
import warnings

from astropy.time import Time
import numpy as np
from astropy.units import Quantity

from . import parameter as uvp
from .utils import _get_iterable
from . import __version__

__all__ = ["UVBase"]


def _warning(msg, *a, **kwargs):
    """Improve the printing of user warnings."""
    return str(msg) + "\n"


class UVBase(object):
    """
    Base class for objects with UVParameter attributes.

    This class is intended to be subclassed and its init method should be
    called in the subclass init after all associated UVParameter attributes are
    defined. The init method of this base class creates properties
    (named using UVParameter.name) from all the UVParameter attributes on the subclass.
    AngleParameter and LocationParameter attributes also have extra convenience
    properties defined:

    AngleParameter:

        UVParameter.name+'_degrees'

    LocationParameter:

        UVParameter.name+'_lat_lon_alt'
        UVParameter.name+'_lat_lon_alt_degrees'
    """

    def __init__(self):
        """Create properties from UVParameter attributes."""
        warnings.formatwarning = _warning

        # set any UVParameter attributes to be properties
        for p in self:
            this_param = getattr(self, p)
            attr_name = this_param.name
            setattr(
                self.__class__,
                attr_name,
                property(self.prop_fget(p), self.prop_fset(p)),
            )
            if isinstance(this_param, uvp.AngleParameter):
                setattr(
                    self.__class__,
                    attr_name + "_degrees",
                    property(self.degree_prop_fget(p), self.degree_prop_fset(p)),
                )
            elif isinstance(this_param, uvp.LocationParameter):
                setattr(
                    self.__class__,
                    attr_name + "_lat_lon_alt",
                    property(
                        self.lat_lon_alt_prop_fget(p), self.lat_lon_alt_prop_fset(p)
                    ),
                )
                setattr(
                    self.__class__,
                    attr_name + "_lat_lon_alt_degrees",
                    property(
                        self.lat_lon_alt_degrees_prop_fget(p),
                        self.lat_lon_alt_degrees_prop_fset(p),
                    ),
                )

        # String to add to history of any files written with this version of pyuvdata
        self.pyuvdata_version_str = (
            f"  Read/written with pyuvdata version: {__version__ }."
        )

    def prop_fget(self, param_name):
        """Getter method for UVParameter properties."""
        # Create function to return
        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.value

        return fget

    def prop_fset(self, param_name):
        """Setter method for UVParameter properties."""
        # Create function to return
        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.value = value
            setattr(self, param_name, this_param)

        return fset

    def degree_prop_fget(self, param_name):
        """Degree getter method for AngleParameter properties."""
        # Create function to return
        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.degrees()

        return fget

    def degree_prop_fset(self, param_name):
        """Degree setter method for AngleParameter properties."""
        # Create function to return
        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.set_degrees(value)
            setattr(self, param_name, this_param)

        return fset

    def lat_lon_alt_prop_fget(self, param_name):
        """Lat/lon/alt getter method for LocationParameter properties."""
        # Create function to return
        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.lat_lon_alt()

        return fget

    def lat_lon_alt_prop_fset(self, param_name):
        """Lat/lon/alt setter method for LocationParameter properties."""
        # Create function to return
        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.set_lat_lon_alt(value)
            setattr(self, param_name, this_param)

        return fset

    def lat_lon_alt_degrees_prop_fget(self, param_name):
        """Lat/lon/alt degree getter method for LocationParameter properties."""
        # Create function to return
        def fget(self):
            this_param = getattr(self, param_name)
            return this_param.lat_lon_alt_degrees()

        return fget

    def lat_lon_alt_degrees_prop_fset(self, param_name):
        """Lat/lon/alt degree setter method for LocationParameter properties."""
        # Create function to return
        def fset(self, value):
            this_param = getattr(self, param_name)
            this_param.set_lat_lon_alt_degrees(value)
            setattr(self, param_name, this_param)

        return fset

    def __iter__(self, uvparams_only=True):
        """Iterate over all UVParameter attributes."""
        attribute_list = [
            a
            for a in dir(self)
            if not a.startswith("__") and not callable(getattr(self, a))
        ]
        param_list = []
        for a in attribute_list:
            attr = getattr(self, a)
            if uvparams_only:
                if isinstance(attr, uvp.UVParameter):
                    param_list.append(a)
            else:
                param_list.append(a)
        for a in param_list:
            yield a

    def required(self):
        """Iterate over all required UVParameter attributes."""
        attribute_list = [
            a
            for a in dir(self)
            if not a.startswith("__") and not callable(getattr(self, a))
        ]
        required_list = []
        for a in attribute_list:
            attr = getattr(self, a)
            if isinstance(attr, uvp.UVParameter):
                if attr.required:
                    required_list.append(a)
        for a in required_list:
            yield a

    def extra(self):
        """Iterate over all non-required UVParameter attributes."""
        attribute_list = [
            a
            for a in dir(self)
            if not a.startswith("__") and not callable(getattr(self, a))
        ]
        extra_list = []
        for a in attribute_list:
            attr = getattr(self, a)
            if isinstance(attr, uvp.UVParameter):
                if not attr.required:
                    extra_list.append(a)
        for a in extra_list:
            yield a

    def __eq__(self, other, check_extra=True):
        """
        Equal if classes match and parameters are equal.

        If check_extra is True, include all parameters, otherwise only include
        required parameters.
        """
        if isinstance(other, self.__class__):
            # only check that required parameters are identical
            self_required = []
            other_required = []
            for p in self.required():
                self_required.append(p)
            for p in other.required():
                other_required.append(p)
            if set(self_required) != set(other_required):
                print(
                    "Sets of required parameters do not match. "
                    f"Left is {self_required},"
                    f" right is {other_required}."
                )
                return False

            if check_extra:
                self_extra = []
                other_extra = []
                for p in self.extra():
                    self_extra.append(p)
                for p in other.extra():
                    other_extra.append(p)
                if set(self_extra) != set(other_extra):
                    print(
                        "Sets of extra parameters do not match. "
                        f"Left is {self_extra},"
                        f" right is {other_extra}."
                    )
                    return False

                p_check = self_required + self_extra
            else:
                p_check = self_required

            p_equal = True
            for p in p_check:
                self_param = getattr(self, p)
                other_param = getattr(other, p)
                if self_param != other_param:
                    print(
                        f"parameter {p} does not match. Left is {self_param.value},"
                        f" right is {other_param.value}."
                    )
                    p_equal = False
            return p_equal
        else:
            print("Classes do not match")
            return False

    def __ne__(self, other):
        """Not equal."""
        return not self.__eq__(other)

    def check(
        self, check_extra=True, run_check_acceptability=True, ignore_requirements=False
    ):
        """
        Check that required parameters exist and have the correct shapes.

        Optionally, check that the values are acceptable.

        Parameters
        ----------
        check_extra : bool
            If true, check shapes and values on all parameters,
            otherwise only check required parameters.
        run_check_acceptability : bool
            Option to check if values in parameters are acceptable.
        ignore_requirements : bool
            Do not error if a required parameter isn't set.
            This allows the user to run the shape/acceptability checks
            on parameters in a partially-defined UVData object.

        """
        if check_extra:
            p_check = list(self.required()) + list(self.extra())
        else:
            p_check = list(self.required())

        for p in p_check:
            param = getattr(self, p)
            # Check required parameter exists
            if param.value is None:
                if ignore_requirements:
                    continue
                if param.required is True:
                    raise ValueError(f"Required UVParameter {p} has not been set.")
            else:
                # Check parameter shape
                eshape = param.expected_shape(self)
                # default value of eshape is ()
                if eshape == "str" or (eshape == () and param.expected_type == "str"):
                    # Check that it's a string
                    if not isinstance(param.value, str):
                        raise ValueError(
                            f"UVParameter {p} expected to be string, but is not."
                        )
                else:
                    # Check the shape of the parameter value. Note that np.shape
                    # returns an empty tuple for single numbers.
                    # eshape should do the same.
                    if isinstance(param.value, Time):
                        this_shape = np.shape(param.value.value)
                    else:
                        this_shape = np.shape(param.value)

                    if not this_shape == eshape:
                        raise ValueError(
                            f"UVParameter {param.name} is not expected shape. "
                            f"Parameter shape is {this_shape}, expected shape is "
                            f"{eshape}."
                        )
                    # Quantity objects complicate things slightly
                    # Do a separate check with warnings until a quantity based
                    # parameter value is created
                    if isinstance(param.value, Quantity):
                        # check if user put expected type as a type of quantity
                        # not a more generic type of number.
                        if any(
                            issubclass(param_type, Quantity)
                            for param_type in _get_iterable(param.expected_type)
                        ):
                            # Verify the param is an instance
                            # of the specific Quantity type
                            if not isinstance(param.value, param.expected_type):
                                raise ValueError(
                                    f"UVParameter {p} is a Quantity object "
                                    "but not the appropriate type. "
                                    f"Is {type(param.value)} but "
                                    f"expected {param.expected_type}."
                                )
                            else:
                                # matches expected type
                                continue  # pragma: no cover
                        else:
                            # Expected type is not a Quantity subclass
                            # Assuming it is a data type like float, int, etc
                            # continuing with check below
                            warnings.warn(
                                f"Parameter {p} is a Quantity object, "
                                "but the expected type is a precision identifier: "
                                f"{param.expected_type}. "
                                "Testing the precision of the value, but this "
                                "check will fail in a future version."
                            )
                            check_vals = [param.value.item(0).value]

                    elif eshape == ():
                        # Single element
                        check_vals = [param.value]
                    else:
                        if isinstance(param.value, (list, tuple)):
                            # List & tuples needs to be handled differently than array
                            # list values may be different types, so they all
                            # need to be checked
                            check_vals = list(param.value)
                        else:
                            # numpy array
                            check_vals = [param.value.item(0)]

                    for val in check_vals:
                        if not isinstance(val, param.expected_type):
                            raise ValueError(
                                f"UVParameter {p} is not the appropriate"
                                f" type. Is:  {type(val)}. "
                                f"Should be: {param.expected_type}."
                            )

                if run_check_acceptability:
                    accept, message = param.check_acceptability()
                    if not accept:
                        raise ValueError(
                            f"UVParameter {p} has unacceptable values. {message}"
                        )

        return True

    def copy(self):
        """Make and return a copy of the object."""
        return copy.deepcopy(self)
