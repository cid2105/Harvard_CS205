import numpy as np
import matplotlib.pyplot as plt
import math
plt.ion()         # Allow interactive updates to the plots

class data_transformer:
  '''A class to transform a line of attenuated data into a back-projected image.
  Construct on the number of data points in a line of data and the number of
  pixels in the resulting square image. This precomputes the
  back-projection operator.
  Once constructed, call the transform method on a line of attenuated data and
  the angle that data represents to retrieve the back-projected image.'''
  def __init__(self, sample_size, image_size):
    '''Perform the required precomputation for the back-projection step.'''
    [self.X,self.Y] = np.meshgrid(np.linspace(-1,1,image_size),
                                  np.linspace(-1,1,image_size))
    self.proj_domain = np.linspace(-1,1,sample_size)
    self.f_scale = abs(np.fft.fftshift(np.linspace(-1,1,sample_size+1)[0:-1]))

  def transform(self, data, phi):
    '''Transform a data line taken at an angle phi to its back-projected image.
    Input: data, an array of sample_size values.
    Output: an image_size x image_size array -- the back-projected image'''
    # Compute the Fourier filtered data
    filtered_data = np.fft.ifft(np.fft.fft(data) * self.f_scale).real
    # Interpolate the data to the rotated image domain
    result = np.interp(self.X*np.cos(phi) + self.Y*np.sin(phi),
                       self.proj_domain, filtered_data)
    return result


if __name__ == '__main__':

  # TODO