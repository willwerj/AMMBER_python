import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit


class TernaryIsothermalDiscretePhase:
    """
    Class representing a single phase at one temperature described by a set of points in ternary composition-Gibbs free energy space
    """

    def __init__(self, x1data, x2data, Gdata):
        """
        Constructor. Initializes the phase object from points in ternary composition-Gibbs free energy space.

        Parameters
        ----------
        x1data : list, array
            compositions of the first component
        x2data : list, array
            compositions of the second component
        Gdata : list, array
            Gibbs free energies corresponding to compositions (x1, x2)
        """
        # Ensure the inputs are unique and sort them accordingly
        x1data, ordering = np.unique(x1data, return_index=True)
        x2data = np.array(x2data)[ordering]
        self.Gdata = np.array(Gdata)[ordering]
        self.x1data = x1data
        self.x2data = x2data
        self.x3data = 1 - x1data - x2data  # x3 inferred

    def resample(self, x1points, x2points):
        """
        Returns a new TernaryIsothermalDiscretePhase object by interpolating between points onto (x1points, x2points). (cubic spline)

        Parameters
        ----------
        x1points : list, array
            compositions of the first component to be resampled over
        x2points : list, array
            compositions of the second component to be resampled over

        Returns
        -------
        TernaryIsothermalDiscretePhase object
        """
        # Filter out points within valid range
        valid_points = np.logical_and(
            np.logical_and(x1points >= self.x1data[0], x1points <= self.x1data[-1]),
            np.logical_and(x2points >= self.x2data[0], x2points <= self.x2data[-1])
        )
        new_x1 = x1points[valid_points]
        new_x2 = x2points[valid_points]

        if len(new_x1) >= 3 and len(new_x2) >= 3:
            spl = CubicSpline(self.x1data, self.Gdata)
            new_G = spl(new_x1)  # Interpolate over x1
            spl2 = CubicSpline(self.x2data, new_G)
            final_G = spl2(new_x2)  # Then interpolate over x2

            return TernaryIsothermalDiscretePhase(new_x1, new_x2, final_G)
        # Return the current phase without resampling if conditions not met
        return self

    def resample_near_xpoint(self, x1, x2, xdist=0.04, npts=101):
        """
        Returns a new TernaryIsothermalDiscretePhase with points resampled near (x1, x2).

        Parameters
        ----------
        x1 : float
            composition of the first component to be resampled near
        x2 : float
            composition of the second component to be resampled near
        xdist : float
            range of compositions from (x1, x2) to be resampled
        npts : positive integer
            number of resampled points

        Returns
        -------
        TernaryIsothermalDiscretePhase object
        """
        x1points = np.linspace(x1 - xdist, x1 + xdist, npts)
        x2points = np.linspace(x2 - xdist, x2 + xdist, npts)
        return self.resample(x1points, x2points)

    def free_energy(self, x1, x2):
        """
        Returns the estimated free energy by interpolating at compositions (x1, x2). (cubic spline)

        Parameters
        ----------
        x1 : float
            composition of the first component
        x2 : float
            composition of the second component

        Returns
        -------
        (float) free energy at composition (x1, x2)
        """
        if len(self.x1data) > 1 and len(self.x2data) > 1:
            spl = CubicSpline(self.x1data, self.Gdata)
            G_interpolated_x1 = spl(x1)
            spl2 = CubicSpline(self.x2data, G_interpolated_x1)
            return spl2(x2)
        elif self.x1data[0] == x1 and self.x2data[0] == x2:
            return self.Gdata[0]
        else:
            print("Attempted to evaluate free energy of a compound outside its range. This will cause errors.")
            return None


class TernaryIsothermal2ndOrderPhase:
    """
    Class representing a single phase at one temperature described by a 2nd order polynomial (parabola)
    in a ternary system.
    """

    def __init__(self, fmin=0.0, kwell=(1.0, 1.0, 1.0), cmin=(0.33, 0.33, 0.33), discrete=None, kwellmax=1e9):
        """
        Constructor.

        Parameters
        ----------
        cmin : tuple (float, float, float)
            composition of the parabolic free energy minimum for each component (x1, x2, x3).
        fmin : float
            corresponding free energy value at (cmin1, cmin2, cmin3)
        kwell : tuple (float, float, float)
            the parabolic curvatures for each component (kwell1, kwell2, kwell3)
        kwellmax : float
            kwell to be used when fitting line compounds
        """
        if discrete is None:
            self.fmin = fmin
            self.kwell = kwell
            self.cmin = cmin
        else:
            self.fit_phase(discrete.xdata1, discrete.xdata2, discrete.Gdata, kwellmax=kwellmax)

    def fit_phase(self, xdata1, xdata2, Gdata, kwellmax=1e9):
        """
        Fits this phase object to a set of points in composition-energy space.

        Parameters
        ----------
        xdata1 : list (float)
            composition values of the first component to be fit
        xdata2 : list (float)
            composition values of the second component to be fit
        Gdata : list (float)
            free-energy values corresponding to (xdata1, xdata2)
        kwellmax : float
            kwell to be used when fitting a line compound
        """
        if len(xdata1) > 2 and len(xdata2) > 2:
            # Perform curve fitting for a ternary system (2 composition variables)
            (a1, a2, b1, b2, c), _ = curve_fit(self.functional_form, np.array([xdata1, xdata2]).T, Gdata,
                                               bounds=([0, 0, -np.inf, -np.inf, -np.inf],
                                                       [2.0 * kwellmax, 2.0 * kwellmax, np.inf, np.inf, np.inf]))

            # Extract parabolic parameters for each component
            self.kwell = (2.0 * a1, 2.0 * a2, 2.0 * (a1 + a2))  # kwell for x1, x2, and inferred x3
            self.cmin = (-b1 / self.kwell[0], -b2 / self.kwell[1], 1 - (-b1 / self.kwell[0] + -b2 / self.kwell[1]))
            self.fmin = c - (self.kwell[0] / 2.0 * self.cmin[0] ** 2 + self.kwell[1] / 2.0 * self.cmin[1] ** 2)
        elif len(xdata1) == 1 and len(xdata2) == 1:
            # If only one point is provided, directly assign values
            self.fmin = Gdata[0]
            self.kwell = (kwellmax, kwellmax, kwellmax)
            self.cmin = (xdata1[0], xdata2[0], 1 - xdata1[0] - xdata2[0])
        else:
            print("Too few points to fit functional form")

    @staticmethod
    def functional_form(x, a1, a2, b1, b2, c):
        """
        Standard form of a second order polynomial for ternary systems. This function is used in fit_phase.
        """
        x1, x2 = x.T
        return a1 * x1 ** 2 + a2 * x2 ** 2 + b1 * x1 + b2 * x2 + c

    def free_energy(self, x1, x2):
        """
        Evaluates the free energy at (x1, x2) using the parabolic parameters.

        Parameters
        ----------
        x1 : float
            composition of the first component to be sampled
        x2 : float
            composition of the second component to be sampled

        Returns
        -------
        (float) free energy at composition (x1, x2)
        """
        x3 = 1 - x1 - x2
        return self.fmin + (self.kwell[0] / 2.0 * (x1 - self.cmin[0]) ** 2 +
                            self.kwell[1] / 2.0 * (x2 - self.cmin[1]) ** 2 +
                            self.kwell[2] / 2.0 * (x3 - self.cmin[2]) ** 2)

    def discretize(self, xdata1=None, xdata2=None, xrange1=(1e-14, 1.0 - 1e-14), xrange2=(1e-14, 1.0 - 1e-14),
                   npts=1001):
        """
        Returns a TernaryIsothermalDiscretePhase object by sampling the free energy function on a range of compositions.
        Provide either xdata, or a range and number of points.

        Parameters
        ----------
        xdata1 : list (float)
            compositions of the first component to be sampled
        xdata2 : list (float)
            compositions of the second component to be sampled
        xrange1 : tuple (min, max)
            range of compositions for the first component to be linearly sampled
        xrange2 : tuple (min, max)
            range of compositions for the second component to be linearly sampled
        npts : int
            number of points to be sampled from in each range

        Returns
        -------
        TernaryIsothermalDiscretePhase object
        """
        if xdata1 is None:
            xdata1 = np.linspace(*xrange1, npts)
        if xdata2 is None:
            xdata2 = np.linspace(*xrange2, npts)

        x1_grid, x2_grid = np.meshgrid(xdata1, xdata2)
        G_values = np.array([self.free_energy(x1, x2) for x1, x2 in zip(x1_grid.flatten(), x2_grid.flatten())])

        return TernaryIsothermalDiscretePhase(x1_grid.flatten(), x2_grid.flatten(), G_values)
