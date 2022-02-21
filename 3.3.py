from matplotlib import image as im
from matplotlib import pyplot as plt
import numpy.linalg as la
import numpy as np

image = im.imread('hendrix_final_20percent.png')
plt.figure(1)
plt.subplot(221)
plt.title("Original hendrix_final_20percent Image")
plt.imshow(image)

red_image = image[:, :, 0]
green_image = image[:, :, 1]
blue_image = image[:, :, 2]

Ur, Sr, Vr = la.svd(red_image)
Ug, Sg, Vg = la.svd(green_image)
Ub, Sb, Vb = la.svd(blue_image)

Er = np.zeros(len(Sr))
for n in range(len(Sr)):
    reconstruction = np.matrix(Ur[:, :n]) @ np.diag(Sr[:n]) @ np.matrix(Vr[:n, :])
    frobenius_error = la.norm(red_image - reconstruction, ord='fro')
    Er[n] = frobenius_error
plt.subplot(222)
plt.plot(Er)
plt.title('Red Channel Frobenius Norm vs Number of Singular Components')
plt.ylabel('Frobenius Norm')
plt.xlabel('No. of singular value components')

Eg = np.zeros(len(Sg))
for n in range(len(Sg)):
    reconstruction = np.matrix(Ug[:, :n]) @ np.diag(Sg[:n]) @ np.matrix(Vg[:n, :])
    frobenius_error = la.norm(green_image - reconstruction, ord='fro')
    Eg[n] = frobenius_error
plt.subplot(223)
plt.plot(Eg)
plt.title('Green Channel Frobenius Norm vs Number of Singular Components')
plt.ylabel('Frobenius Norm')
plt.xlabel('No. of singular value components')

Eb = np.zeros(len(Sb))
for n in range(len(Sb)):
    reconstruction = np.matrix(Ub[:, :n]) @ np.diag(Sb[:n]) @ np.matrix(Vb[:n, :])
    frobenius_error = la.norm(blue_image - reconstruction, ord='fro')
    Eb[n] = frobenius_error
plt.subplot(224)
plt.plot(Eb)
plt.title('Blue Channel Frobenius Norm vs Number of Singular Components')
plt.ylabel('Frobenius Norm')
plt.xlabel('No. of singular value components')

plt.show()
