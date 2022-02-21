from matplotlib import image as im
from matplotlib import pyplot as plt
import numpy.linalg as la


image = im.imread('hendrix_final_20percent.png')
plt.figure(1)
plt.subplot(221)
plt.title("hendrix_final_20percent Image")
plt.imshow(image)

im.imsave('red.png', image[:, :, 0])
im.imsave('green.png', image[:, :, 1])
im.imsave('blue.png', image[:, :, 2])

red_image = image[:, :, 0]
green_image = image[:, :, 1]
blue_image = image[:, :, 2]

Ur, Sr, Vr = la.svd(red_image)
Ug, Sg, Vg = la.svd(green_image)
Ub, Sb, Vb = la.svd(blue_image)

plt.subplot(222)
plt.loglog(Sr)
plt.xlabel('non-zero singular values for R')
plt.ylabel('number of singular values')
plt.title('SVD Red channel log-log plot')

plt.subplot(223)
plt.loglog(Sg)
plt.xlabel('non-zero singular values for G')
plt.ylabel('number of singular values')
plt.title('SVD Green channel log-log plot')

plt.subplot(224)
plt.loglog(Sb)
plt.xlabel('non-zero singular values for B')
plt.ylabel('number of singular values')
plt.title('SVD Blue channel log-log plot')

plt.show()