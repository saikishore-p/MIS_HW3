from matplotlib import image as im
from matplotlib import pyplot as plt
import numpy.linalg as la
import numpy as np
import tensorly as tl

image = im.imread('hendrix_final_20percent.png')

L, M, P = image.shape
LMP = L * M * P
Xr = tl.unfold(image, 0)
Ur, Sr, Vr = la.svd(Xr)
Xg = tl.unfold(image, 1)
Ug, Sg, Vg = la.svd(Xg)
Xb = tl.unfold(image, 2)
Cb = Xb @ Xb.T
evals, evect = la.eig(Cb)
Ubb = evect

Y0 = np.einsum("ij, jkl -> ikl", Ur.T, image)
Y1 = np.einsum("ik, jkl -> jil", Ug.T, Y0)
Y2 = np.einsum("il, jkl -> jki", Ubb.T, Y1)
S = Y2

Xr = np.zeros([L, M, P])
ist = np.unravel_index(np.argsort(-np.abs(S), axis=None), S.shape)
N = 4000
Error_Xr = np.zeros([N])
for n in range(0, N):
    i = (ist[0])[n]
    j = (ist[1])[n]
    k = (ist[2])[n]
    Xr += S[i, j, k] * np.einsum("i, j, k -> ijk", Ur[:, i], Ug[:, j], Ubb[:, k])
    Error_Xr[n] = tl.norm(image - Xr, order=2, axis=None)

plt.figure(1)
plt.subplot(221)
plt.loglog(np.abs(S[ist]))
plt.title('core tensor value and components plot')
plt.xlabel('components log scale')
plt.ylabel('core tensor values log scale')

plt.subplot(222)
plt.plot(Error_Xr)
plt.title('Error plot')
plt.xlabel("No. of components")
plt.ylabel("Tensor error norm")

HOSVD_first_error = Error_Xr[0]
HOSVD_criteria_error = HOSVD_first_error - 0.6 * HOSVD_first_error
HOSVD_criteria_error_index = np.min(np.where(Error_Xr < HOSVD_criteria_error))

HOSVD_reconstructed_image = np.zeros([L, M, P])
for n in range(0, HOSVD_criteria_error_index):
    i = (ist[0])[n]
    j = (ist[1])[n]
    k = (ist[2])[n]
    HOSVD_reconstructed_image += S[i, j, k] * np.einsum("i, j, k -> ijk", Ur[:, i], Ug[:, j], Ubb[:, k])

plt.subplot(223)
plt.imshow(image)
plt.title('Original hendrix_final_20percent Image')
plt.subplot(224)
plt.imshow(HOSVD_reconstructed_image)
plt.title('HOSVD criteria reconstructed image')

plt.show()
