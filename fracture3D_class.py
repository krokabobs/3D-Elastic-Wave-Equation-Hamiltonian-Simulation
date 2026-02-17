import numpy as np

class Fracture3D:
    """
    A class representing a 3D fracture.

    Attributes:
        grid_dim (tuple): (Nx, Ny, Nz) numbers of grid points for each axis
        cube_length (tuple): (dx, dy, dz) lengths of one cube for each axis
        fracture (dict): dictionary containing information of each subvolume
    """

    def __init__(self, grid_dim, cube_length):
        # Verify the valid attributes
        assert isinstance(grid_dim, tuple) == True
        assert len(grid_dim) == 3
        assert isinstance(cube_length, tuple) == True
        assert len(cube_length) == 3
        for i in range(len(grid_dim)):
            assert isinstance(grid_dim[i], int) == True
            assert grid_dim[i] >= 2
            assert isinstance(cube_length[i], float) == True
        
        self.__grid_dim = grid_dim
        self.__cube_length = cube_length

        self.__staggered_grid_points = []
        self.__main_grid_points = []
        self.__cubes = {}

        # Intialize two lists of points for the fracture object: staggered grid and main grid
        # Each main grid point consists of position, associated physical properties, and the wavefield vector of 9 components
        # Each staggered grid point consists of position and the component of velocity or stress
        staggered_x = 2*grid_dim[0] - 1
        staggered_y = 2*grid_dim[1] - 1
        staggered_z = 2*grid_dim[2] - 1
        for z in range(staggered_z+1):
            for y in range(staggered_y+1):
                for x in range(staggered_x+1):
                    if x % 2 == 0 and y % 2 == 0 and z % 2 == 0:
                        self.__main_grid_points.append(MainGridPoint((x//2, y//2, z//2)))

        
        # Initialize the fracture dictionary with subvolume information
        # Each subvolume has the key (a,b,c) denoting ath cube along x-axis, bth along y-axis, and cth along z-axis.
        for z in range(grid_dim[2]-1):
            for y in range(grid_dim[1]-1):
                for x in range(grid_dim[0]-1):
                    self.__cubes[(x, y, z)] = {
                        'center': ((x+0.5)*cube_length[0],
                                   (y+0.5)*cube_length[1],
                                   (z+0.5)*cube_length[2]),
                        'main_grid_covered': [MainGridPoint(x, y, z),
                                              MainGridPoint(x+cube_length[0], y, z),
                                              MainGridPoint(x, y+cube_length[1], z),
                                              MainGridPoint(x, y, z+cube_length[2]),
                                              MainGridPoint(x+cube_length[0], y+cube_length[1], z),
                                              MainGridPoint(x+cube_length[0], y, z+cube_length[2]),
                                              MainGridPoint(x, y+cube_length[1], z+cube_length[2]),
                                              MainGridPoint(x+cube_length[0], y+cube_length[1], z+cube_length[2])],
                        'staggered_grid_covered': [StaggeredGridPoint(x+0.5, y, z, 'v_x'),
                                                   StaggeredGridPoint(x, y+0.5, z, 'v_y]
                    }

        
    @property
    def gid_dim(self):
        return self.__grid_dim
        
    @property
    def cube_length(self):
        return self.__cube_length
    
    def staggeredGridSize(self):
        '''
        Return the total number of staggered grid points for velocity and stress components in the fracture.
        '''
        Nx = self.__vol_dim[0] + 1
        Ny = self.__vol_dim[1] + 1
        Nz = self.__vol_dim[2] + 1
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
        self.__wavefield = np.zeros(9)

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

    @property
    def position(self):
        return self.__position
        
    @property
    def component(self):
        return self.__component
        
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