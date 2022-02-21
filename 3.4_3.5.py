from matplotlib import image as im
from matplotlib import pyplot as plt
import numpy.linalg as la
import numpy as np

image = im.imread('hendrix_final_20percent.png')
plt.figure(1)
plt.subplot(121)
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

Eg = np.zeros(len(Sg))
for n in range(len(Sg)):
    reconstruction = np.matrix(Ug[:, :n]) @ np.diag(Sg[:n]) @ np.matrix(Vg[:n, :])
    frobenius_error = la.norm(green_image - reconstruction, ord='fro')
    Eg[n] = frobenius_error

Eb = np.zeros(len(Sb))
for n in range(len(Sb)):
    reconstruction = np.matrix(Ub[:, :n]) @ np.diag(Sb[:n]) @ np.matrix(Vb[:n, :])
    frobenius_error = la.norm(blue_image - reconstruction, ord='fro')
    Eb[n] = frobenius_error

n = 30

R_reconstruction = np.matrix(Ur[:, :n]) @ np.diag(Sr[:n]) @ np.matrix(Vr[:n, :])
G_reconstruction = np.matrix(Ug[:, :n]) @ np.diag(Sg[:n]) @ np.matrix(Vg[:n, :])
B_reconstruction = np.matrix(Ub[:, :n]) @ np.diag(Sb[:n]) @ np.matrix(Vb[:n, :])

SVD_30_reconstructed_image = np.concatenate([np.expand_dims(R_reconstruction, axis=2),
                                          np.expand_dims(G_reconstruction, axis=2),
                                          np.expand_dims(B_reconstruction, axis=2),
                                          ], axis=2)

plt.figure(1)
plt.imshow(SVD_30_reconstructed_image)
plt.title('Reconstructed with 30 Dimensions')


n = 130

R_reconstruction = np.matrix(Ur[:, :n]) @ np.diag(Sr[:n]) @ np.matrix(Vr[:n, :])
G_reconstruction = np.matrix(Ug[:, :n]) @ np.diag(Sg[:n]) @ np.matrix(Vg[:n, :])
B_reconstruction = np.matrix(Ub[:, :n]) @ np.diag(Sb[:n]) @ np.matrix(Vb[:n, :])

SVD_130_reconstructed_image = np.concatenate([np.expand_dims(R_reconstruction, axis=2),
                                          np.expand_dims(G_reconstruction, axis=2),
                                          np.expand_dims(B_reconstruction, axis=2),
                                          ], axis=2)

plt.figure(2)
plt.imshow(SVD_130_reconstructed_image)
plt.title('Reconstructed with 130 Dimensions')

n = 80

R_reconstruction = np.matrix(Ur[:, :n]) @ np.diag(Sr[:n]) @ np.matrix(Vr[:n, :])
G_reconstruction = np.matrix(Ug[:, :n]) @ np.diag(Sg[:n]) @ np.matrix(Vg[:n, :])
B_reconstruction = np.matrix(Ub[:, :n]) @ np.diag(Sb[:n]) @ np.matrix(Vb[:n, :])

SVD_80_reconstructed_image = np.concatenate([np.expand_dims(R_reconstruction, axis=2),
                                          np.expand_dims(G_reconstruction, axis=2),
                                          np.expand_dims(B_reconstruction, axis=2),
                                          ], axis=2)

plt.figure(3)
plt.imshow(SVD_80_reconstructed_image)
plt.title('Reconstructed with 80 Dimensions')

plt.show()
