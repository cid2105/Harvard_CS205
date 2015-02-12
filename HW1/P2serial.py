import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

def get_data(filepath):
    numrows, numcols = 2048, 6144
    dt = np.dtype(float) 
    data = np.fromfile(filepath, dtype=dt).reshape((numrows, numcols)) 
    return data

if __name__ == '__main__':
    sample_size, image_size = 6144, 512
    data = get_data("PA1Distro/TomoData.bin")
    transformer = data_transformer(sample_size, image_size)
    result = reduce(lambda image, k: image + transformer.transform(data[k-1, :], -k*np.pi/sample_size), np.arange(2048))
    fig = plt.figure(figsize=(15,10),facecolor='w') 
    plt.imshow(result, cmap='bone')
    plt.imsave('P2serial.png', data, cmap='bone')