import numpy as np

class Fracture3D:
    """
    A class representing a 3D fracture.

    Attributes:
        grid_dim (tuple): (Nx, Ny, Nz) numbers of grid points for each axis
        unit_dim (tuple): (dx, dy, dz) lengths of unit volume for each axis
        fracture (dict): dictionary containing information of each subvolume
    """

    def __init__(self, grid_dim, unit_dim):
        # Verify the valid attributes
        assert isinstance(grid_dim, tuple) == True
        assert len(grid_dim) == 3
        assert isinstance(unit_dim, tuple) == True
        assert len(unit_dim) == 3
        for i in range(len(grid_dim)):
            assert isinstance(grid_dim[i], int) == True
            assert grid_dim[i] >= 2
            assert isinstance(unit_dim[i], float) == True

        self.__grid_dim = grid_dim
        self.__unit_dim = unit_dim

        # Intialize two dictionaries of points for the fracture object: staggered grid and main grid
        # Each main grid point consists of position, associated physical properties, and the wavefield vector of 9 components
        # Each staggered grid point consists of position and the component of velocity or stress
        # Each key is a tuple of half-integer indices. Each value is a point object with the corresponding physical position.
        self.__staggered_grid_points = {}
        self.__main_grid_points = {}
        # Also, store staggered grid points separately for each component of velocity and stress for easy access
        for z in range(grid_dim[2]):
            for y in range(grid_dim[1]):
                for x in range(grid_dim[0]):

                    # Add staggered grid points
                    if x+0.5 <= grid_dim[0]-1:
                        self.__staggered_grid_points[(x+0.5, y, z)] = StaggeredGridPoint(((x+0.5)*unit_dim[0], y*unit_dim[1], z*unit_dim[2]), 'v_x')
                    if y+0.5 <= grid_dim[1]-1:
                        self.__staggered_grid_points[(x, y+0.5, z)] = StaggeredGridPoint((x*unit_dim[0], (y+0.5)*unit_dim[1], z*unit_dim[2]), 'v_y')
                    if z+0.5 <= grid_dim[2]-1:
                        self.__staggered_grid_points[(x, y, z+0.5)] = StaggeredGridPoint((x*unit_dim[0], y*unit_dim[1], (z+0.5)*unit_dim[2]), 'v_z')
                    self.__staggered_grid_points[(x, y, z)] = StaggeredGridPoint((x*unit_dim[0], y*unit_dim[1], z*unit_dim[2]), 's_xx')
                    self.__staggered_grid_points[(x, y, z)] = StaggeredGridPoint((x*unit_dim[0], y*unit_dim[1], z*unit_dim[2]), 's_yy')
                    self.__staggered_grid_points[(x, y, z)] = StaggeredGridPoint((x*unit_dim[0], y*unit_dim[1], z*unit_dim[2]), 's_zz')
                    if x+0.5 <= grid_dim[0]-1 and y+0.5 <= grid_dim[1]-1:
                        self.__staggered_grid_points[(x+0.5, y+0.5, z)] = StaggeredGridPoint(((x+0.5)*unit_dim[0], (y+0.5)*unit_dim[1], z*unit_dim[2]), 's_xy')
                    if x+0.5 <= grid_dim[0]-1 and z+0.5 <= grid_dim[2]-1:
                        self.__staggered_grid_points[(x+0.5, y, z+0.5)] = StaggeredGridPoint(((x+0.5)*unit_dim[0], y*unit_dim[1], (z+0.5)*unit_dim[2]), 's_xz')
                    if y+0.5 <= grid_dim[1]-1 and z+0.5 <= grid_dim[2]-1:
                        self.__staggered_grid_points[(x, y+0.5, z+0.5)] = StaggeredGridPoint((x*unit_dim[0], (y+0.5)*unit_dim[1], (z+0.5)*unit_dim[2]), 's_yz')

                    # Add main grid point
                    self.__main_grid_points[(x, y, z)] = MainGridPoint((x*unit_dim[0], y*unit_dim[1], z*unit_dim[2]))
                    #self.__main_grid_points[(x, y, z)] = MainGridPoint((x*unit_dim[0], y*unit_dim[1], z*unit_dim[2]))

    @property
    def gid_dim(self):
        return self.__grid_dim

    @property
    def unit_dim(self):
        return self.__unit_dim

    def staggeredGridSize(self):
        '''
        Return the total number of staggered grid points for velocity and stress components in the fracture.
        '''

        Nx = self.__grid_dim[0]
        Ny = self.__grid_dim[1]
        Nz = self.__grid_dim[2]
        N_main = Nx * Ny * Nz
        N_vx = (Nx-1) * Ny * Nz
        N_vy = Nx * (Ny-1) * Nz
        N_vz = Nx * Ny * (Nz-1)
        N_sxy = (Nx-1) * (Ny-1) * Nz
        N_sxz = (Nx-1) * Ny * (Nz-1)
        N_syz = Nx * (Ny-1) * (Nz-1)
        N_vel = N_vx + N_vy + N_vz
        N_stress = 3*N_main + N_sxy + N_sxz + N_syz
        N_total = N_vel + N_stress
        return N_total

    def translation(self, dist):
        '''
        Translate a fracture with given distances along each axis

        :param dist (tuple): (dist_x, dist_y, dist_z) distance to translate along x-axis, y-axis, and z-axis, respectively
        '''

        assert isinstance(dist, tuple) == True
        assert len(dist) == 3
        for i in range(len(dist)):
            assert isinstance(dist[i], float) == True

        pass

class MainGridPoint:
    '''
    A class representing a point of main grid in a Fracture3D object.

    Attributes:
        position (tuple): (i, j, k) position of the grid point along x-axis, y-axis, and z-axis, respectively
        properties (dict): physical properties of the subvolume associated with the grid point
        wavefield (numpy array): wavefield vector of 9 components at the grid point
    '''

    def __init__(self, position):

        # Verify the valid attributes
        assert isinstance(position, tuple) == True
        assert len(position) == 3

        self.__position = position
        self.__properties = {}
        self.__wavefield = {
            'v_x': None,
            'v_y': None,
            'v_z': None,
            's_xx': None,
            's_yy': None,
            's_zz': None,
            's_xy': None,
            's_xz': None,
            's_yz': None
        }

    @property
    def position(self):
        return self.__position

    @property
    def properties(self):
        return self.__properties

    @property
    def wavefield(self):
        return self.__wavefield

    def translation(self, dist):
        '''
        Translate a grid point with given distances along each axis

        :param dist (tuple): (dist_x, dist_y, dist_z) distance to translate the point along x-axis, y-axis, and z-axis, respectively
        '''

        assert isinstance(dist, tuple) == True
        assert len(dist) == 3
        for i in range(len(dist)):
            assert isinstance(dist[i], float) == True

        self.__position

class StaggeredGridPoint:
    '''
    A class representing a point of staggered grid in a Fracture3D object.

    Attributes:
        position (tuple): (i, j, k) position of the grid point along x-axis, y-axis, and z-axis, respectively
        component (str): center of volume [c], velocity in [v_x, v_y, v_z], or stress in [s_xx, s_yy, s_zz, s_xy, s_xz, s_yz]
    '''

    def __init__(self, position, component):

        # Verify the valid attributes
        assert isinstance(position, tuple) == True
        assert len(position) == 3
        assert component in ['v_x', 'v_y', 'v_z', 's_xx', 's_yy', 's_zz', 's_xy', 's_xz', 's_yz']

        self.__position = position
        self.__component = component
        self.__value = 0

    @property
    def position(self):
        return self.__position

    @property
    def component(self):
        return self.__component

    @property
    def value(self):
        return self.__value

    def associated_main_grid_point(self, dx, dy, dz):
        '''
        Return the position of the main grid point associated with the staggered grid point.
        '''

        position = self.__position
        component = self.__component
        if component in ['s_xx', 's_yy', 's_zz']:
            return position
        elif component == 'v_x':
            return (position[0]-0.5*dx, position[1], position[2])
        elif component == 'v_y':
            return (position[0], position[1]-0.5*dy, position[2])
        elif component == 'v_z':
            return (position[0], position[1], position[2]-0.5*dz)
        elif component == 's_xy':
            return (position[0]-0.5*dx, position[1]-0.5*dy, position[2])
        elif component == 's_xz':
            return (position[0]-0.5*dx, position[1], position[2]-0.5*dz)
        elif component == 's_yz':
            return (position[0], position[1]-0.5*dy, position[2]-0.5*dz)

    def translation(self, dist):
        '''
        Translate a grid point with given distances along each axis

        :param dist (tuple): (dist_x, dist_y, dist_z) distance to translate the point along x-axis, y-axis, and z-axis, respectively
        '''

        assert isinstance(dist, tuple) == True
        assert len(dist) == 3
        for i in range(len(dist)):
            assert isinstance(dist[i], float) == True

        self.__position = tuple(pos + trans for pos, trans in zip(self.__position, dist))

        return self.__position