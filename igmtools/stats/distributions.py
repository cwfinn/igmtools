"""
Functions for dealing with statistical distributions.

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

__all__ = ['ProbabilityDistribution']


class ProbabilityDistribution(object):
    """
    Represents an arbitrary discrete distribution as a probability mass
    function.

    Parameters
    ----------
    x : array
        Values sampling the continuous random variable.

    distribution : array
        The value of the discrete distribution function at each x.

    Attributes
    ----------
    pmf : array
        The probability mass function.

    cmf : array
        The cumulative probability mass function.

    x : array
        Values sampling the continuous random variable.


    Notes
    -----
    This finds the mass density function and the cumulative mass density
    function.

    Make sure the input distribution is sampled densely enough (i.e. there
    are enough x values to properly define the shape of both the CDF and its
    inverse), because linear interpolation is used between the provided x
    values and CDF to infer new x values when generating the random numbers.
    A log sampling of x values is appropriate for distributions like inverse
    power laws, for example.

    """

    def __init__(self, x, distribution):

        # Normalise such that area under PDF is 1:
        self.pmf = distribution / np.trapz(distribution, x=x)

        # Cumulative probability distribution:
        self.cmf = distribution.cumsum()
        self.cmf /= self.cmf[-1]
        self.x = x

    def random(self, n=1, seed=None):
        """ Generate a random set drawn from the inferred probability
        density function.

        Parameters
        ----------
        N : int
            Number of random values to be drawn from the inferred probability
            density function.

        seed : int, optional
            Random seed.

        """

        if seed:
            np.random.seed(seed)

        i = np.random.rand(n)
        y = np.interp(i, self.cmf, self.x)

        return y
