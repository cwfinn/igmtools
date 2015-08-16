"""
Data representation, including units and uncertainty propagation. This
module is still experimental.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.units import Quantity, Unit, dimensionless_unscaled, UnitsError

import numpy as np

import operator

__all__ = ['Data', 'Uncertainty']


class Data(Quantity):
    """
    An experimantal class for data representation. Assumes Gaussian
    uncertainties.

    Parameters
    ----------
    value : float, array, `astropy.units.Quantity`, or `astro.data.Data`
        The data value.

    uncertainty : float, array, or `astropy.units.Quantity`
        Uncertainty (standard deviation) on the data value.

    unit : `astropy.units.UnitBase` or str, optional
        The data unit.

    Notes
    -----
    Use with caution. This still requires work, and only basic arithmetic is
    supported (+, -, /, **). Logarithms are not currently handled. Works with
    NumPy arrays and indexing, but ufuncs such as np.sum() or np.prod() will
    not propagate uncertainties at present.

    """

    def __new__(cls, value, uncertainty, unit=None):

        return Quantity.__new__(cls, value, unit)

    def __init__(self, value, uncertainty, unit=None):

        if isinstance(value, self.__class__):
            self._uncertainty = value.uncertainty

        else:
            if isinstance(value, Quantity):
                if unit is not None:
                    raise ValueError('cannot use the unit argument when '
                                     '`value` is a Quantity')

            else:
                self._unit = (dimensionless_unscaled if unit is None else
                              Unit(unit))

            if hasattr(uncertainty, 'unit'):

                if (uncertainty.unit != dimensionless_unscaled and
                        self._unit != dimensionless_unscaled):
                    try:
                        uncertainty = uncertainty.to(self._unit)
                    except:
                        raise UnitsError('cannot convert unit of uncertainty '
                                         'to unit of the `Data` instance')

                if (self._unit == dimensionless_unscaled and
                        uncertainty.unit != dimensionless_unscaled):
                    raise UnitsError('cannot assign an uncertainty with units '
                                     'to a `Data` instance without a unit')

                uncertainty = Uncertainty(uncertainty)

            else:
                uncertainty = Uncertainty(uncertainty, self._unit)

            if uncertainty.size != self.size:
                raise ValueError('uncertainty must have the same shape as '
                                 'the `Data` instance')

            self._uncertainty = uncertainty
            self._uncertainty.parent_data = self

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self._unit = getattr(obj, '_unit', None)
        self._uncertainty = getattr(obj, '_uncertainty', None)

    def __str__(self):

        return '{0} +/- {1}{2}'.format(
            self.value, self.uncertainty.value, self._unitstr)

    def __repr__(self):

        prefix = '<' + self.__class__.__name__ + ' '

        return '{0}{1} +/- {2}{3:s}>'.format(
            prefix, self.value, self.uncertainty.value, self._unitstr)

    def __getitem__(self, item):

        uncertainty = self.uncertainty.__getitem__(item)
        out = super(Data, self).__getitem__(item)
        out.uncertainty = uncertainty

        return out

    @property
    def uncertainty(self):
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value):

        if value is not None:

            if not isinstance(value, Uncertainty):
                raise TypeError('uncertainty must be an instance of an '
                                '`Uncertainty` object')

            if value.size != self.size:
                raise ValueError('uncertainty must have the same shape as '
                                 'the `Data` instance')

            if self.unit != dimensionless_unscaled and value.unit:
                try:
                    value = value.to(self.unit)
                except:
                    raise UnitsError('cannot convert unit of uncertainty to '
                                     'unit of the `Data` instance')

            if self.unit == dimensionless_unscaled and value.unit:
                raise UnitsError('cannot assign an uncertainty with units to '
                                 'a `Data` instance without a unit')

        self._uncertainty = value
        self._uncertainty.parent_data = self

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):

        if self._unit != dimensionless_unscaled:
            raise AttributeError('the unit has already been set. Use the '
                                 '.convert_unit_to() method to change the '
                                 'unit and scale the data appropriately')

        elif value is None:
            self._unit = dimensionless_unscaled

        else:
            self._unit = Unit(value)

    def copy(self):
        return self.__class__(self.value, self.uncertainty, self.unit)

    def _arithmetic(self, operand, propagate_uncertainties, operation,
                    reverse=False):
        """
        {description}

        Parameters
        ----------
        operand : `igmtools.data.Data`, `astropy.units.Quantity`, float or int
            Either a or b in the operation a {operator} b

        propagate_uncertainties : str
            The name of one of the propagation rules defined by the
            `igmtools.data.Uncertainty` class.

        operation : `numpy.ufunc`
            The function that performs the operation.

        reverse : bool, optional (default=False)
            Sets the operand order for the operation. Set to True to place
            `self` after the operation.

        Returns
        -------
        result : `igmtools.data.Data` or `astropy.units.Quantity`
            The data or quantity resulting from the arithmetic.

        Notes
        -----
        Uncertainties are propagated, although correlated errors are not
        supported.

        """

        if isinstance(operand, (int, float)):
            operand = Quantity(operand)

        if reverse:
            try:
                result = operation(operand.value * operand.unit,
                                   self.value * self.unit)
            except:
                raise UnitsError('operand units do not match')

        else:
            try:
                result = operation(self.value * self.unit,
                                   operand.value * operand.unit)
            except:
                raise UnitsError('operand units do not match')

        # If we are not propagating uncertainties, we should just return the
        # result here:
        if not propagate_uncertainties:
            return result

        unit = result.unit
        value = result.value

        # If the operation is addition or subtraction then we need to ensure
        # that the operand is in the same units as the result:
        if (operation in (operator.add, operator.sub)
                and unit != operand.unit):
            operand = operand.to(unit)

        method = getattr(self.uncertainty, propagate_uncertainties)

        if propagate_uncertainties in ('propagate_add', 'propagate_subtract'):
            uncertainty = method(operand)

        else:
            uncertainty = method(operand, result)

        result = self.__class__(value, uncertainty.value, unit)

        return result

    def __neg__(self, propagate_uncertainties=True):

        return self.__rsub__(0, propagate_uncertainties)

    def __add__(self, operand, propagate_uncertainties=True):

        if propagate_uncertainties:
            propagate_uncertainties = 'propagate_add'

        else:
            propagate_uncertainties = None

        return self._arithmetic(
            operand, propagate_uncertainties, np.add)

    __add__.__doc__ = _arithmetic.__doc__.format(
        description='Add another value (`operand`) to this one.',
        operator='+')

    def __iadd__(self, operand, propagate_uncertainties=True):

        if propagate_uncertainties:
            propagate_uncertainties = 'propagate_add'

        else:
            propagate_uncertainties = None

        return self._arithmetic(
            operand, propagate_uncertainties, np.add)

    __iadd__.__doc__ = _arithmetic.__doc__.format(
        description='Add another value (`operand`) to this one.',
        operator='+')

    def __radd__(self, operand, propagate_uncertainties=True):

        if propagate_uncertainties:
            propagate_uncertainties = 'propagate_add'

        else:
            propagate_uncertainties = None

        return self._arithmetic(
            operand, propagate_uncertainties, np.add, reverse=True)

    __radd__.__doc__ = _arithmetic.__doc__.format(
        description='Add this value to another (`operand`).',
        operator='+')

    def __sub__(self, operand, propagate_uncertainties=True):

        if propagate_uncertainties:
            propagate_uncertainties = 'propagate_subtract'

        else:
            propagate_uncertainties = None

        return self._arithmetic(
            operand, propagate_uncertainties, np.subtract)

    __sub__.__doc__ = _arithmetic.__doc__.format(
        description='Subtract another value (`operand`) from this one.',
        operator='-')

    def __isub__(self, operand, propagate_uncertainties=True):

        if propagate_uncertainties:
            propagate_uncertainties = 'propagate_subtract'

        else:
            propagate_uncertainties = None

        return self._arithmetic(
            operand, propagate_uncertainties, np.subtract)

    __isub__.__doc__ = _arithmetic.__doc__.format(
        description='Subtract another value (`operand`) from this one.',
        operator='-')

    def __rsub__(self, operand, propagate_uncertainties=True):

        if propagate_uncertainties:
            propagate_uncertainties = 'propagate_subtract'

        else:
            propagate_uncertainties = None

        return self._arithmetic(
            operand, propagate_uncertainties, np.subtract, reverse=True)

    __rsub__.__doc__ = _arithmetic.__doc__.format(
        description='Subtract this value from another (`operand`).',
        operator='-')

    def __mul__(self, operand, propagate_uncertainties=True):

        if propagate_uncertainties:
            propagate_uncertainties = 'propagate_multiply'

        else:
            propagate_uncertainties = None

        return self._arithmetic(
            operand, propagate_uncertainties, np.multiply)

    __mul__.__doc__ = _arithmetic.__doc__.format(
        description='Multiply another value (`operand`) by this one.',
        operator='*')

    def __imul__(self, operand, propagate_uncertainties=True):

        if propagate_uncertainties:
            propagate_uncertainties = 'propagate_multiply'

        else:
            propagate_uncertainties = None

        return self._arithmetic(
            operand, propagate_uncertainties, np.multiply)

    __imul__.__doc__ = _arithmetic.__doc__.format(
        description='Multiply another value (`operand`) by this one.',
        operator='*')

    def __rmul__(self, operand, propagate_uncertainties=True):

        if propagate_uncertainties:
            propagate_uncertainties = 'propagate_multiply'

        else:
            propagate_uncertainties = None

        return self._arithmetic(
            operand, propagate_uncertainties, np.multiply, reverse=True)

    __rmul__.__doc__ = _arithmetic.__doc__.format(
        description='Multiply this value by another (`operand`).',
        operator='*')

    def __div__(self, operand, propagate_uncertainties=True):

        if propagate_uncertainties:
            propagate_uncertainties = 'propagate_divide'

        else:
            propagate_uncertainties = None

        return self._arithmetic(
            operand, propagate_uncertainties, np.divide)

    __div__.__doc__ = _arithmetic.__doc__.format(
        description='Divide this value by another (`operand`).',
        operator='/')

    def __idiv__(self, operand, propagate_uncertainties=True):

        if propagate_uncertainties:
            propagate_uncertainties = 'propagate_divide'

        else:
            propagate_uncertainties = None

        return self._arithmetic(
            operand, propagate_uncertainties, np.divide)

    __idiv__.__doc__ = _arithmetic.__doc__.format(
        description='Divide this value by another (`operand`).',
        operator='/')

    def __rdiv__(self, operand, propagate_uncertainties=True):

        if propagate_uncertainties:
            propagate_uncertainties = 'propagate_divide'

        else:
            propagate_uncertainties = None

        return self._arithmetic(
            operand, propagate_uncertainties, np.divide, reverse=True)

    __rdiv__.__doc__ = _arithmetic.__doc__.format(
        description='Divide another value (`operand`) by this one.',
        operator='/')

    def __pow__(self, operand, propagate_uncertainties=True):

        if hasattr(operand, 'unit'):
            if operand.unit != dimensionless_unscaled:
                raise UnitsError('can only raise something to a '
                                 'dimensionless quantity')

        elif propagate_uncertainties:
            propagate_uncertainties = 'propagate_power_left'

        else:
            propagate_uncertainties = None

        return self._arithmetic(
            operand, propagate_uncertainties, np.power)

    __pow__.__doc__ = _arithmetic.__doc__.format(
        description='Raise this value to the power of another (`operand`)',
        operator='^')

    def __ipow__(self, operand, propagate_uncertainties=True):

        if hasattr(operand, 'unit'):
            if operand.unit != dimensionless_unscaled:
                raise UnitsError('can only raise something to a '
                                 'dimensionless quantity')

        elif propagate_uncertainties:
            propagate_uncertainties = 'propagate_power_left'

        else:
            propagate_uncertainties = None

        return self._arithmetic(
            operand, propagate_uncertainties, np.power)

    __ipow__.__doc__ = _arithmetic.__doc__.format(
        description='Raise this value to the power of another (`operand`)',
        operator='^')

    def __rpow__(self, operand, propagate_uncertainties=True):

        if self.unit != dimensionless_unscaled:
            raise UnitsError('can only raise something to a dimensionless '
                             'quantity')

        elif propagate_uncertainties:
            propagate_uncertainties = 'propagate_power_right'

        else:
            propagate_uncertainties = None

        return self._arithmetic(
            operand, propagate_uncertainties, np.power, reverse=True)

    __rpow__.__doc__ = _arithmetic.__doc__.format(
        description='Raise another value (`operand`) to the power of this one',
        operator='^')

    def to(self, unit, equivalencies=None):
        """
        Returns a new Data object whose values have been converted to a new
        unit.

        Parameters
        ----------
        unit : `astropy.units.UnitBase` instance or str
            The unit to convert to.

        equivalencies : list, optional
           A list of equivalence pairs to try if the units are not
           directly convertible.

        Returns
        -------
        result : `igmtools.data.Data`
            The resulting data value.

        Raises
        ------
        UnitsError
            If the units are inconsistent.

        """

        if self.unit is None:
            raise UnitsError('no unit specified on source data')

        data = self.unit.to(unit, self.value, equivalencies)

        uncertainty_value = self.unit.to(unit, self.uncertainty, equivalencies)
        uncertainty = self.uncertainty.__class__(uncertainty_value)

        result = self.__class__(data, uncertainty, unit)

        return result


class Uncertainty(Quantity):
    """
    A class for standard deviation uncertainties.

    """

    def __new__(cls, value, unit=None):

        return Quantity.__new__(cls, value, unit)

    def __init__(self, value, unit=None):

        if isinstance(value, self.__class__):
            self._parent_data = value._parent_data

        else:
            self._parent_data = None

    def __array_finalize__(self, obj):

        if obj is None:
            return

        self._parent_data = getattr(obj, '_parent_data', None)

    @property
    def parent_data(self):

        try:
            if self._parent_data is None:
                raise AttributeError('uncertainty is not associated with a '
                                     'Data object')

            else:
                return self._parent_data

        except:
            raise AttributeError('uncertainty is not associated with a Data '
                                 'object')

    @parent_data.setter
    def parent_data(self, value):

        self._parent_data = value

    def propagate_add(self, other_data):
        """
        Propagates uncertainties for addition.

        Parameters
        ----------
        other_data : `igmtools.data.Data` or `astropy.units.Quantity`
            The data/quantity a or b in a + b.

        Returns
        -------
        uncertainty : `igmtools.data.Uncertainty`
            The resulting uncertainty.

        """

        if not isinstance(other_data, Data):
            uncertainty = self

        else:
            uncertainty = np.sqrt(self ** 2 + other_data.uncertainty ** 2)

        return uncertainty

    def propagate_subtract(self, other_data):
        """
        Propagates uncertainties for subtraction.

        Parameters
        ----------
        other_data : `igmtools.data.Data` or `astropy.units.Quantity`
            The data/quantity a or b in a - b.

        Returns
        -------
        uncertainty : `igmtools.data.Uncertainty`
            The resulting uncertainty.

        """

        if not isinstance(other_data, Data):
            uncertainty = self

        else:
            uncertainty = np.sqrt(self ** 2 + other_data.uncertainty ** 2)

        return uncertainty

    def propagate_multiply(self, other_data, result):
        """
        Propagates uncertainties for multiplication.

        Parameters
        ----------
        other_data : `igmtools.data.Data` or `astropy.units.Quantity`
            The data/quantity a or b in a * b.

        result : `astropy.units.Quantity`
            The result of the multiplication.

        Returns
        -------
        uncertainty : `igmtools.data.Uncertainty`
            The resulting uncertainty.

        """

        if not isinstance(other_data, Data):
            if isinstance(other_data, Quantity):
                uncertainty = self.value * other_data.value
            else:
                uncertainty = self.value * other_data

        else:
            uncertainty = (
                np.sqrt(
                    (self.value / self.parent_data.value) ** 2 +
                    (other_data.uncertainty.value / other_data.value) ** 2) *
                np.fabs(result.value))

        return self.__class__(uncertainty, result.unit)

    def propagate_divide(self, other_data, result):
        """
        Propagates uncertainties for division.

        Parameters
        ----------
        other_data : `igmtools.data.Data` or `astropy.units.Quantity`
            The data/quantity a or b in a / b.

        result : `astropy.units.Quantity`
            The result of the division.

        Returns
        -------
        uncertainty : `igmtools.data.Uncertainty`
            The resulting uncertainty.

        """

        if not isinstance(other_data, Data):
            if isinstance(other_data, Quantity):
                uncertainty = self.value / other_data.value
            else:
                uncertainty = self.value / other_data

        else:
            uncertainty = (
                np.sqrt(
                    (self.value / self.parent_data.value) ** 2 +
                    (other_data.uncertainty.value / other_data.value) ** 2) *
                np.fabs(result.value))

        return self.__class__(uncertainty, result.unit)

    def propagate_power_left(self, other_data, result):
        """
        Propagates uncertainties for raising to a power when `self`
        represents the uncertainty in the data left of the operator.

        Parameters
        ----------
        other_data : `igmtools.data.Data` or `astropy.units.Quantity`
            The data/quantity b in a ^ b.

        result : `astropy.units.Quantity`
            The result of raising to the power.

        Returns
        -------
        uncertainty : `igmtools.data.Uncertainty`
            The resulting uncertainty.

        """

        if not isinstance(other_data, Data):
            if isinstance(other_data, Quantity):
                uncertainty = np.fabs(
                    result.value * other_data.value * self.value /
                    self.parent_data.value)
            else:
                uncertainty = np.fabs(
                    result.value * other_data * self.value /
                    self.parent_data.value)

        else:
            uncertainty = (
                np.sqrt((other_data.value * self.value /
                         self.parent_data.value) ** 2 +
                        (np.log(self.parent_data.value) *
                         other_data.uncertainty.value) ** 2) *
                np.fabs(result.value))

        return self.__class__(uncertainty, result.unit)

    def propagate_power_right(self, other_data, result):
        """
        Propagates uncertainties for raising to a power when `self`
        represents the uncertainty in the data right of the operator.

        Parameters
        ----------
        other_data : `igmtools.data.Data` or `astropy.units.Quantity`
            The data/quantity a in a ^ b.

        result : `astropy.units.Quantity`
            The result of raising to the power.

        Returns
        -------
        uncertainty : `igmtools.data.Uncertainty`
            The resulting uncertainty.

        """

        if not isinstance(other_data, Data):
            if isinstance(other_data, Quantity):
                uncertainty = np.fabs(
                    result.value * np.log(other_data.value) *
                    self.uncertainty.value)
            else:
                uncertainty = np.fabs(
                    result.value * np.log(other_data) *
                    self.uncertainty.value)

        else:
            uncertainty = (
                np.sqrt((self.parent_data.value *
                         other_data.uncertainty.value /
                         other_data.value) ** 2 +
                        (np.log(other_data.value) *
                         self.uncertainty.value) ** 2) *
                np.fabs(result.value))

        return self.__class__(uncertainty, result.unit)
