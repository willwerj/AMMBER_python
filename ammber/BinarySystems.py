from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
import numpy as np
from pycalphad import calculate
from PRISMS_PF_fileGen import write_binary_isothermal_parabolic_parameters
class BinaryIsothermalDiscretePhase:
    "Class representing a single phase at one temperature described by a set of points in composition-Gibbs free energy space"
    def __init__(self, xdata, Gdata):
        """
        Constructor. Initializes the phase object from points in composition-Gibbs free energy space.

        Parameters
        ----------
        xdata : list, array
            compositions
        Gdata : list, array
            Gibbs free energies corresponding to compositions
        """
        self.xdata, ordering = np.unique(xdata, return_index=True)
        self.Gdata = Gdata[ordering]

    def resample(self, xpoints):
        """
        Returns a new BinaryIsothermalDiscretePhase object by interpolating between points onto xpoints. (cubic spline)
        
        Parameters
        ----------
        xpoints : list, array
            compositions to be resampled over

        Returns
        -------
        BinaryIsothermalDiscretePhase object
        """
        newx = xpoints[np.logical_and(xpoints >= self.xdata[0], xpoints <= self.xdata[-1])]
        spl = CubicSpline(self.xdata, self.Gdata)
        return BinaryIsothermalDiscretePhase(newx, spl(newx))

    def resample_near_xpoint(self, x, xdist=0.04, npts=101):
        """
        Returns a new BinaryIsothermalDiscretePhase with only points near x.
        
        Parameters
        ----------
        x : float
            composition to be resampled near
        xdist : float
            range of compositions from x to be resampled
        npts : positive integer
            number of resampled points
        
        Returns
        -------
        BinaryIsothermalDiscretePhase object
        """
        xpoints = np.linspace(x - xdist, x + xdist, npts)
        return self.resample(xpoints)

    def free_energy(self, x):
        """
        Returns the estimated free energy by interpolating at a composition x. (cubic spline)
        
        Parameters
        ----------
        x : float
            composition to be sampled

        Returns
        -------
        (float) free energy at composition x
        """
        spl = CubicSpline(self.xdata, self.Gdata)
        return spl(x)

class BinaryIsothermal2ndOrderPhase:
    "Class representing a single phase at one temperature described by a 2nd order polynomial (parabola)."
    def __init__(self, fmin=0.0, kwell=1.0, cmin=0.5, discrete=None, kwellmax=1e9):#todo
        """
        Constructor.
        
        Parameters
        ----------
        cmin : float
            composition of the parabolic free energy minimum
        fmin : float
            corresponding free energy value at cmin
        kwell : float
            the parabolic curvature
        kwellmax : float
            kwell to be used when fitting line compounds

        Returns
        -------
        BinaryIsothermal2ndOrderPhase object
        """
        if discrete is None:
            self.fmin = fmin
            self.kwell = kwell
            self.cmin = cmin
        else:
            self.fit_phase(discrete.xdata, discrete.Gdata, kwellmax=kwellmax)

    def fit_phase(self, xdata, Gdata, kwellmax=1e9):
        """
        Fits this phase object to a set of points in composition-energy space.
        
        Parameters
        ----------
        xdata : list (float)
            composition values of energy to be fit
        Gdata : list(float)
            free-energy values corresponding to xdata
        kwellmax : float
            kwell to be used when fitting a line compound
        """
        if len(xdata) > 2:
            (a, b, c), _ = curve_fit(self.functional_form, xdata, Gdata,
                                     bounds=([0, -np.inf, -np.inf], [2.0 * kwellmax, np.inf, np.inf]))
            print(a, b, c)
            self.kwell = 2.0*a
            self.cmin = -b / self.kwell
            self.fmin = c - self.kwell/2.0 * self.cmin**2
        elif len(xdata) == 1:
            self.fmin = Gdata[0]
            self.kwell = kwellmax
            self.cmin = xdata[0]
            print(Gdata[0], kwellmax, xdata[0])
        else:
            print("too few points to fit functional form")

    @staticmethod
    def functional_form(x, a, b, c):
        "Standard form of a second order polynomial. This function is used in fit_phase"
        return a * x ** 2 + b * x + c

    def free_energy(self, x):
        """
        Evaluates the free energy at x using the parabolic parameters.
        
        Parameters
        ----------
        x : float
            composition to be sampled

        Returns
        -------
        (float) free energy at composition x
        """
        return self.fmin + self.kwell / 2.0 * (x - self.cmin)**2

    def discretize(self, xdata=None, xrange=(1e-14, 1.0 - 1e-14), npts=1001):
        """
        Returns a BinaryIsothermalDiscretePhase object by sampling the free energy function on a range of compositions
        Provide either xdata, or a range and number of points.

        Parameters
        ----------
        xdata : list (float)
            compositions to be sampled
        xrange : tuple (min, max)
            range of compositions to be linearly sampled
        npts : number of points to be sampled from in xrange

        Returns
        -------
        BinaryIsothermalDiscretePhase object
        """
        if xdata is None:
            xdata = np.linspace(*xrange, npts)
        return BinaryIsothermalDiscretePhase(xdata, self.free_energy(xdata))


class BinaryIsothermalDiscreteSystem:
    "Class representing a seet of phases at one temperature described by a set of points in composition-Gibbs free energy space"
    def __init__(self):
        "Constructor. Initializes with an empty set of phases."
        self.phases = {}
        self.lc_hull = None
        self._is_equilibrium_system = False

    def fromTDB(self, db, elements, component, temperature, phase_list=None):
        """
        Constructs discrete phase from all the phases specified in a pycalphad database

        Parameters
        ----------
        db : pycalphad database
        elements : list (string)
            The space of compositions of interest (Binary). Element abbreviations must be all-caps.
        componenet : string
            Element abbreviation for element corresponding to x=1.
        phase_list : list (string)
            If specified, only listed phases will be constructed, otherwise, all available phases will be constructed.
        """
        if "VA" not in elements:
            elements.append("VA")
        if phase_list is None:
            phase_list = list(db.phases.keys())
        for phase_name in phase_list:
            result = calculate(db, elements, phase_name, P=101325, T=temperature, output='GM', pdens=1001)
            self.phases[phase_name] = BinaryIsothermalDiscretePhase(
                result.X.sel(component=component).data[0][0][0],
                result.GM.data[0][0][0])

    def add_phase(self, name, xdata, Gdata):
        """
        Constructs a new BinaryIsothermalDiscretePhase object from points in composition-Gibbs free energy space.

        Parameters
        ----------
        name : string
            name of phase to be added
        xdata : list, array
            compositions
        Gdata : list, array
            Gibbs free energies corresponding to compositions
        """
        self.phases[name] = BinaryIsothermalDiscretePhase(xdata, Gdata)

    def get_lc_hull(self, recalculate=False):#todo
        """
        Returns a the set of x-G points that lie on the convex hull of all points in all phases
        
        Parameters
        ----------
        recalculate : bool
            if true, performs calculation
        
        Returns
        -------
        numpy array: points on the lower convex hull of all phases
        """
        if self.lc_hull is None or recalculate:
            xdata_list = []
            Gdata_list = []
            for phase in self.phases:
                xdata_list.append(self.phases[phase].xdata)
                Gdata_list.append(self.phases[phase].Gdata)
            allpoints = np.stack((np.concatenate(tuple(xdata_list)),
                                  np.concatenate(tuple(Gdata_list))), axis=-1)
            self.lc_hull = get_lower_convex_hull(allpoints)
        return self.lc_hull

    def get_equilibrium_system(self, miscibility_gap_threshold=0.1):#todo
        """
        Returns the phases that may be present at any composition at equilibrium
        
        Parameters
        ----------
        miscibility_gap_threshold : float
            (optional) Threshold to separate miscibility gap phase into multiple phases

        Returns
        -------
        BinaryIsothermalDiscreteSystem object
        """
        lc_hull = self.get_lc_hull()
  #      print(lc_hull)
        equilibrium_system = BinaryIsothermalDiscreteSystem()
        equilibrium_system._is_equilibrium_system = True
        for phase in self.phases:
            mask = np.logical_and(np.isin(self.phases[phase].xdata, lc_hull[:, 0]),
                                  np.isin(self.phases[phase].Gdata, lc_hull[:, 1]))
            if np.count_nonzero(mask) > 0:
                equilibrium_system.phases[phase] = BinaryIsothermalDiscretePhase(
                    self.phases[phase].xdata[mask],
                    self.phases[phase].Gdata[mask])
        # break apart phases that phase separate
        for phase in self.phases:
            # find indices where TDB phase free energy is discontinuous wrt x, gap between points exceeds deltax_thresh
            gap_inds = []
            if phase in equilibrium_system.phases:
                gap_inds = np.nonzero(equilibrium_system.phases[phase].xdata[1:] -
                            equilibrium_system.phases[phase].xdata[:-1] > miscibility_gap_threshold)[0]
            #print(gap_inds)
            if len(gap_inds) > 0:
                phase_temp = equilibrium_system.phases.pop(phase)
                for i in range(0, len(gap_inds)+1):
                    if i == 0:
                        start = 0
                    else:
                        start = gap_inds[i-1]+1
                    if i == len(gap_inds):
                        end = phase_temp.xdata.size
                    else:
                        end = gap_inds[i]+1
                    #print(start,end)
                    if end - start >= 1:
                        # phase is represented by multiple points
                        equilibrium_system.phases[phase +'_'+ str(i)] = BinaryIsothermalDiscretePhase(
                            phase_temp.xdata[start:end], phase_temp.Gdata[start:end])
                    elif(start == end):
                        # phase is represented by single point
                        print("phase "+phase+'_'+str(i)+" is represented by a single point, a finer discretization may be needed")
                        equilibrium_system.phases[phase +'_'+ str(i)] = BinaryIsothermalDiscretePhase(
                            phase_temp.xdata[start], phase_temp.Gdata[start])
        return equilibrium_system

    def get_equilibrium(self, x):#todo
        """
        Returns the composition or phase-boundary compositions of the 1 or 2 phases that exist at equilibrium at composition x
        
        Parameters
        ----------
        x : float
            composition to be sampled

        Returns
        -------
        dict {string phase_name : array (float) compositions }
            compositions of the phases present at equilibrium at x
        """
        lc_hull = self.get_lc_hull()
        if self._is_equilibrium_system:
            equilibrium_system = self
        else:
            equilibrium_system = self.get_equilibrium_system()
        xind = np.searchsorted(lc_hull[:, 0], x)
        rightpt = lc_hull[xind, :]
        leftpt = lc_hull[xind - 1, :]
        equilibrium = {}
        for phase in equilibrium_system.phases:
            if not ((equilibrium_system.phases[phase].xdata[0] > rightpt[0]) or
                    (equilibrium_system.phases[phase].xdata[-1] < leftpt[0])):
                mask = np.isin(equilibrium_system.phases[phase].Gdata,
                               [rightpt[1], leftpt[1]])
                if np.count_nonzero(mask) == 2:
                    equilibrium[phase] = x
                    return equilibrium
                elif np.count_nonzero(mask) == 1:
                    equilibrium[phase] = equilibrium_system.phases[phase].xdata[mask]
            if len(equilibrium) > 1:
                break
        return equilibrium

    def resample_near_equilibrium(self, x, xdist=0.001, npts=101):
        """
        Returns a system that only contains points and phases near compositions of the phases present at equilibrium at x
        
        Parameters
        ----------
        x : float
            composition to be sampled
        xdist : float
            (optional) the range of compositions to be sampled around the points of interest
        npts : positive int
            number of points to sample from each phase

        Returns
        -------
        BinaryIsothermalDiscreteSystem object
        """
        equilibrium = self.get_equilibrium(x)
        new_system = BinaryIsothermalDiscreteSystem()
        for phase in equilibrium:
            if phase in self.phases:
                new_system.phases[phase] = self.phases[phase].resample_near_xpoint(
                    equilibrium[phase], xdist=xdist, npts=npts)
            # if the equilibrium represents 2 concentrations of the same phase, check
            # if a 2-character extension was added to the phase name, e.g., '_1'
            elif phase[:-2] in self.phases:
                new_system.phases[phase] = self.phases[phase[:-2]].resample_near_xpoint(
                    equilibrium[phase], xdist=xdist, npts=npts)
        return new_system


class BinaryIsothermal2ndOrderSystem:
    "Class representing a seet of phases at one temperature described by a second order polynomial (parabola)"
    def __init__(self, phases=None):
        """
        Constructor.
        
        Parameters
        ----------
        phases : dict {string phase_name : BinaryIsothermal2ndOrderPhase phase}
            (optional) composition to be sampled
        """
        self.phases = {}
        if phases is not None:
            self.phases = phases

    def from_discrete(self, discrete_system, kwellmax=1e6):
        """
        Builds parabolic phases from a discrete system using a fit.
        
        Parameters
        ----------
        discrete_system : BinaryIsothermalDiscreteSystem
            discrete system to be fit
        kwellmax : float
            (optional) kwell to be used when fitting line compounds
        """
        for phase_name in discrete_system.phases.keys():
            self.phases[phase_name] = BinaryIsothermal2ndOrderPhase()
            self.phases[phase_name].fit_phase(discrete_system.phases[phase_name].xdata,
                discrete_system.phases[phase_name].Gdata, kwellmax=kwellmax)

    def from_discrete_near_equilibrium(self, discrete_system, x, kwellmax=1e6, xdist=0.001, npts=101):#todo
        """
        Builds parabolic phases from a discrete system by fitting near phase-boundaries.
        
        Parameters
        ----------
        discrete_system : BinaryIsothermalDiscreteSystem
            discrete system to be fit
        x : float
            composition to detemine equilibrium compositions from
        kwellmax : float
            (optional) kwell to be used when fitting line compounds
        """
        neareq_system = discrete_system.resample_near_equilibrium(x, xdist=0.001, npts=101)
        self.from_discrete(neareq_system, kwellmax=kwellmax)

    def to_discrete(self, xdata=None, xrange=(1e-14, 1.0 - 1e-14), npts=1001):
        """
        Builds a discrete system of phases by evaluating free energies on a range
        Provide either xdata, or a range and number of points.

        Parameters
        ----------
        xdata : list (float)
            compositions to be sampled
        xrange : tuple (min, max)
            range of compositions to be linearly sampled
        npts : positive int
            number of points to be sampled from in xrange

        Returns
        -------
        BinaryIsothermalDiscretePhase object
        """
        discrete_system = BinaryIsothermalDiscreteSystem()
        for phase_name in self.phases.keys():
            discrete_system.phases[phase_name] = self.phases[phase_name].discretize(
                xdata=xdata, xrange=xrange, npts=npts)
        return discrete_system


def get_lower_convex_hull(inputpoints):
    """
        Returns a the set of points that lie on the convex hull of inputpoints
        
        Parameters
        ----------
        inputpoints : numpy array (Nx2)
            points to take convex hull
        
        Returns
        -------
        numpy array: points on the lower convex hull
        """
    # from https://stackoverflow.com/questions/76838415/lower-convex-hull
    # modified to check to make sure leftmost point has lowest y-axis value
    hull = ConvexHull(inputpoints)
    points = inputpoints[hull.vertices]
    minx = np.argmin(points[:, 0])
    maxx = np.argmax(points[:, 0])
#    if abs(maxx - minx) <= 1:
#        lower_convex_hull = np.stack([points[minx],points[maxx]])
#    else:
    maxx = maxx+1
    if minx >= maxx:
        lower_convex_hull = np.concatenate([points[minx:], points[:maxx]])
    else:
        lower_convex_hull = points[minx:maxx]
    return lower_convex_hull

class BinaryDiscreteSystem:
    def __init__(self, temperature_list=[], system_list=[]):
        self.isothermal_systems = dict(zip(temperature_list, system_list))

    def add_temperature(self, temperature, system):
        self.isothermal_systems[temperature] = system

    def fromTDB(self, db, elements, component, temperature_list, phase_list=None):
        for temperature in temperature_list:
            system = BinaryIsothermalDiscreteSystem()
            self.isothermal_systems[temperature] = system.fromTDB(
                db, elements, component, temperature, phase_list=phase_list)

class Binary2ndOrderSystem:
    def __init__(self, temperature_list=[], system_list=[]):
        self.phases = {}
        if phases is not None:
            self.phases = phases

    def from_discrete(self, discrete_system, kwellmax=1e6):
        for phase_name in discrete_system.phases.keys():
            self.phases[phase_name] = BinaryIsothermal2ndOrderPhase()
            self.phases[phase_name].fit_phase(discrete_system.phases[phase_name].xdata,
                discrete_system.phases[phase_name].Gdata, kwellmax=kwellmax)

    def to_discrete(self, xdata=None, xrange=(1e-14, 1.0 - 1e-14), npts=1001):
        discrete_system = BinaryIsothermalDiscreteSystem()
        for phase_name in self.phases.keys():
            discrete_system.phases[phase_name] = self.phases[phase_name].discretize(
                xdata=xdata, xrange=xrange, npts=npts)
        return discrete_system


class Binary2ndOrderPhase:
    def __init__(self, parameter_dict={"a0":1.0,"a1":0.0,"b0":0.0,"b1":1.0,"c0":0.0,"c1":0.0},
                     discrete_system=None, phase_name=None, kwellmax=1e9,
                     isothermal_list=None, temperature_list=None):
        if discrete_system is not None:
            if phase_name is None:
                # just use first phase in discrete system
                phase_name = next(iter(next(iter(discrete_system)).phases))

            self.a1 = a1
            self.a0 = a0
            self.b1 = b1
        else:
            self.fit_phase(discrete.xdata, discrete.Gdata, kwellmax=kwellmax)

    def fit_from_isothermal(self, xdata, Gdata, kwellmax=1e9):
        if len(xdata) > 2:
            (a, b, c), _ = curve_fit(self.functional_form, xdata, Gdata,
                                     bounds=([0, -np.inf, -np.inf], [2.0 * kwellmax, np.inf, np.inf]))
            print(a, b, c)
            self.kwell = 2.0*a
            self.cmin = -b / self.kwell
            self.fmin = c - self.kwell/2.0 * self.cmin**2
        elif len(xdata) == 1:
            self.fmin = Gdata[0]
            self.kwell = kwellmax
            self.cmin = xdata[0]
            print(Gdata[0], kwellmax, xdata[0])
        else:
            print("too few points to fit functional form")

