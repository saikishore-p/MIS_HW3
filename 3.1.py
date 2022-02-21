from matplotlib import image as im
import numpy as np
from matplotlib import pyplot as plt

image = im.imread('hendrix_final_20percent.png')
print("image details:\n")
print(image)
print(image.shape)
print(type(image))
print(type(image[1][1][1]))  # each channel of the image is in <class 'numpy.float32'> type
plt.figure(1)
plt.subplot(332)
plt.title("Original Image")
plt.imshow(image)

im.imsave('red.png', image[:, :, 0])
im.imsave('green.png', image[:, :, 1])
im.imsave('blue.png', image[:, :, 2])

red_image = im.imread('red.png')
green_image = im.imread('green.png')
blue_image = im.imread('blue.png')
print(red_image)
print(red_image.shape)

red_image_64 = np.array(red_image, dtype=np.float64)
green_image_64 = np.array(green_image, dtype=np.float64)
blue_image_64 = np.array(blue_image, dtype=np.float64)

print(red_image_64)
print(red_image_64.shape)
print(type(red_image_64[1][1][1]))  # Shows the type as <class 'numpy.float64'>

plt.subplot(334)
plt.title("Red Channel")
plt.imshow(red_image_64)
plt.subplot(335)
plt.title("Green Channel")
plt.imshow(green_image_64)
plt.subplot(336)
plt.title("Blue Channel")
plt.imshow(blue_image_64)

red_split = np.zeros(image.shape)
blue_split = np.zeros(image.shape)
green_split = np.zeros(image.shape)

red_split[:, :, 0] = image[:, :, 0]
plt.subplot(337)
plt.title("Red Split")
plt.imshow(red_split)
green_split[:, :, 1] = image[:, :, 1]
plt.subplot(338)
plt.title("Green Split")
plt.imshow(green_split)
blue_split[:, :, 2] = image[:, :, 2]
plt.subplot(339)
plt.title("Blue Split")
plt.imshow(blue_split)

plt.show()