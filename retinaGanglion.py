import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image, ImageOps
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata

class ThalamusInput:
    """creates an thalamic input. An image is read in and filtered by
    the mexican hat function"""
    
    def __init__(self, L4_input_params, mexican_hat_params, extent=4):
        #Dictionary with control parameters
        self.params_dict = L4_input_params
        self.mexican_hat_dict = mexican_hat_params

        #In Class used and calculated properties
        """gets the start parameters for the L4 input
        self.retina_neuron_x : np.array() x-coordinate of jittered lgn neuron
        self.retina_neuron_y : np.array() y-coordinate of jittered lgn neuron
        image : np.array( [ [] ] ) 2d quadratic array of a grayscale image
        jitter : np.array() 1d array of int values
        retina_neuron_coordinates : tupel of x-coordinate array and y-coordinate array in the retina
        lgn_neuron_coordinates : tupel of x-coordinate array and y-coordinate array in the thalamus
        lgn_output_matrix : 2d numpy array"""
        self.extent = extent
        self.retina_neuron_x = None
        self.retina_neuron_y = None
        self.lgn_neuron_x = None
        self.lgn_neuron_y = None
        self.image = None
        self.jitter = 0
        self.retina_neuron_coordinates = None
        self.lgn_neuron_coordinates = None
        self.lgn_output_matrix = None

        return

    def image_readout(self, picture_name, path_addition=None):
        """reads in the image and cuts it to a quadratic form"""
        # checks, if the programm is used by the network.py(internal) or directly by the user(external)
        if os.path.isdir(os.getcwd() + "/code/core/simulation/pictures/"):
            path = os.getcwd() + "/code/core/simulation/pictures/"
        elif os.path.isdir(os.getcwd() + "/pictures/"):
            path = os.getcwd() + "/pictures/"
        elif path_addition != None:
            path = os.path.join(path, path_addition)
        else:
            path = os.getcwd()

        # only quadratic images are valid
        # extra if for the a image to make it square
        image = Image.open(os.path.join(path, picture_name))
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image = image.convert("L") #only greyscale is accounted for
        image = np.array(image)
        self.image = image

        return image

    def __mexican_hat(self, x):
        """defines the gaussian function and calculates the mexican hat function
        based on it. sigmac and sigmas need to be scaled to the thalamus size L"""
        sc = self.mexican_hat_dict["sigmac"] * self.extent * 10 ** -3 / 4
        ss = self.mexican_hat_dict["sigmas"] * self.extent * 10 ** -3 / 4

        ''' a gaussian has the form exp(-r**2 / sigma**2) we feed in x = r**2
        therefore the x is not squared in this expression'''
        def gaussian(x, w, sigma):
            return (w / sigma**2) * np.exp(-x / (2 * sigma**2))
        
        mexican_hat = (gaussian(x, self.mexican_hat_dict["wc"], sc) -
                        gaussian(x, self.mexican_hat_dict["ws"], ss))
        return mexican_hat

    def build_retina_and_lgn(self):
        """places the the lgn neurons equidistant on a N_L**2 grid.
        ganglion_x and ganglion_y are 1-d arrays, which represent the
        coordinates of the lgn neurons in the thalamus in a 2-d array.
        The x direction runs from left to right and has just to be repeated.
        The y direction runs from top to bottom and therefore needs a reshape"""

        if self.lgn_neuron_coordinates == None:

            #number of neurons per row and free spaces between two neurons
            N_row = int(np.sqrt(self.params_dict["N"]))

            neuron_x = np.linspace(-self.extent/2, self.extent/2, N_row)
            neuron_x = np.tile(neuron_x, N_row)

            neuron_y = np.linspace(-self.extent/2, self.extent/2, N_row)
            neuron_y = np.tile(neuron_y,(N_row, 1))
            neuron_y = np.reshape(neuron_y, self.params_dict["N"], order="F")

            self.retina_neuron_x = np.copy(neuron_x)
            self.retina_neuron_y = np.copy(neuron_y)
            self.retina_neuron_coordinates = self.retina_neuron_x, self.retina_neuron_y

        self.lgn_neuron_x = np.copy(self.retina_neuron_x)
        self.lgn_neuron_y = np.copy(self.retina_neuron_y)
        self.lgn_neuron_coordinates = self.lgn_neuron_x, self.lgn_neuron_y

        return

    def __calc_jitter_size(self):
        """Calculates a random jittering for every the lgn cells.
        The dictionary jitter value is given by a continuous number but is converted to
        discrete grid steps to to match the discrete grid."""
        self.jitter = self.params_dict["jitter"] * self.extent
        random_jitter = np.random.uniform(-self.jitter, self.jitter, self.params_dict["N"])
        return random_jitter

    def jitter_lgn(self):

        self.lgn_neuron_x += (self.__calc_jitter_size())
        self.lgn_neuron_y += (self.__calc_jitter_size())
        self.lgn_neuron_coordinates = self.retina_neuron_x, self.retina_neuron_y

        return

    def __distance_matrix_stack(self, neuron_coordinates, size):
        """This function is used in combination with the filter function to create a convolution like
        operation. First this function creates the distance squared 2d matrix stack with the lgn neurons as center.
        Then the filter function is applied on every 2d matrix."""
        
        """Creates len(neuron_x) matrices. Given neuron_x[i] and neuron_y[i] 
        values as coordinates, each matrix element represents the squared 
        distance to these coordinates."""
        
        """coordinates : np.array, np.array
           neuron_x : np.array  x-coordinate of the thalamic neuron
           neuron_y : np.array  y-coordinate of the thalamic neuron
           size     : integer   1-d gridsize of possible neuron positions"""

        neuron_x, neuron_y = neuron_coordinates

        x = np.linspace(-self.extent / 2, self.extent / 2, size)
        y = np.linspace(-self.extent / 2, self.extent / 2, size)
        xx, yy = np.meshgrid(x, y)

        xx = (xx[:, :] - neuron_x[:, np.newaxis, np.newaxis])
        yy = (yy[:, :] - neuron_y[:, np.newaxis, np.newaxis])

        zz = yy**2 + xx**2

        return zz


    def lgn_rates(self):
        """Calculates the filtered image in the lgn plane"""
        
        """Uses retina_neuron_coordinates to create a lgn population. Then
        creates the distance_matrix_stack applies the linear filter to it and
        multiplies with the image. The stack is summed up to get a 2-d function
        and at last the lgn neurons are jittered."""
        
        """retina_neuron_coordinates : np.array([int]), np.array([int]) grid coordinates of lgn neurons
           M : int column length of the image
           delta : float scaling factor when changing from integral to sum"""

        self.retina_neuron_coordinates = self.retina_neuron_x, self.retina_neuron_y

        M = len(self.image[0][:])

        delta = self.extent * 10 ** -4 / M
    
        #Filtering the image
        distances_matrices = self.__distance_matrix_stack(self.retina_neuron_coordinates, M)
        spatial_filter_kernels = self.__mexican_hat(distances_matrices * (self.extent * 10 ** -4)**2)

        # how the filter function looks in space
        self.example_filter_kernel = spatial_filter_kernels[int((self.params_dict["N"] + int(np.sqrt(self.params_dict["N"]))) / 2), :, :]

        single_kernel_convolutions = spatial_filter_kernels * self.image[np.newaxis, :, :]
        # delta**2 makes the pixel intensity independent of the image pixel size
        convolved_image = delta**2 * np.sum(single_kernel_convolutions, axis=(1, 2))
        self.lgn_output = convolved_image
        return

    def heatmap(self, matrix, title):
        """Heatmap function for visualization. takes a
         2d array as argument"""
        """matrix : np.array([ [] ])"""
        x=["x"]
        y=["y"]
        plt.xticks(ticks=np.arange(len(x)),labels=x)
        plt.yticks(ticks=np.arange(len(x)),labels=y)
        hm=plt.imshow(matrix, cmap="plasma", interpolation="gaussian", origin='lower')
        plt.colorbar(hm)
        plt.title(title)
        plt.show()
        return 0

    def lgn_rate_plotting(self, gridsize):
        rates = self.lgn_output

        # create heatmap
        fig = plt.figure(figsize=(15, 15))

        ax = fig.add_subplot()
        ax.set_title('lgn rate difference')
        ax.set_ylabel('y in mm', size=10)
        ax.set_xlabel("x in mm", size=10)

        # define the grid onto which to interpolate the data
        xi = np.linspace(-self.extent / 2, self.extent / 2, gridsize)
        yi = np.linspace(-self.extent / 2, self.extent / 2, gridsize)
        Xi, Yi = np.meshgrid(xi, yi)
        # interpolate the data onto the grid
        zi = griddata((self.lgn_neuron_x, self.lgn_neuron_y), rates, (Xi, Yi), method='nearest')

        # create the plot using imshow
        plot = ax.imshow(zi, extent=(xi.min(), xi.max(), yi.min(), yi.max()),
                         cmap='plasma', vmin=min(rates), vmax=max(rates), origin='lower')
        fig.colorbar(plot, label='$\Delta$rates')
        plt.show()

if __name__ == "__main__":
    L4_input_params = {
        "N": 100 ** 2,  # Number of lgn neurons

        "jitter": 0.01,  # relative jittering of the neurons i.e. extent of 4mm and jitter 0.01 => 0.04mm jittering

        "image": "test.png",
        }

    # wc and ws should always have the same sign
    mexican_hat_ON = {
        "wc": 0.2,  # sets values for the Filter constants
        "ws": 0.2,
        "sigmac": 0.2 / 6,  # both sigmas are in (0.1, 1) * L/6
        "sigmas": 0.4 / 6,
    }

    mexican_hat_OFF = {
        "wc": -0.2,  # sets values for the Filter constants
        "ws": -0.2,
        "sigmac": 0.25 / 6,  # both sigmas are in (0.1, 1) * L/6
        "sigmas": 0.45 / 6,
    }

    thalamus = ThalamusInput(L4_input_params, mexican_hat_OFF)
    thalamus.build_retina_and_lgn()
    thalamus.jitter_lgn()
    thalamus.image_readout('grating_30_0.png')
    thalamus.lgn_rates()

    thalamus.heatmap(thalamus.image, 'Image fed into the retina')
    thalamus.heatmap(thalamus.example_filter_kernel, 'Mexican hat kernel function')
    thalamus.lgn_rate_plotting(gridsize=300)