from matplotlib import image as im
import numpy.linalg as la
import numpy as np
import tensorly as tl

image = im.imread('hendrix_final_20percent.png')

red_image = image[:, :, 0]
green_image = image[:, :, 1]
blue_image = image[:, :, 2]

Ur, Sr, Vr = la.svd(red_image)
Ug, Sg, Vg = la.svd(green_image)
Ub, Sb, Vb = la.svd(blue_image)

n = 80

R_reconstruction = np.matrix(Ur[:, :n]) @ np.diag(Sr[:n]) @ np.matrix(Vr[:n, :])
G_reconstruction = np.matrix(Ug[:, :n]) @ np.diag(Sg[:n]) @ np.matrix(Vg[:n, :])
B_reconstruction = np.matrix(Ub[:, :n]) @ np.diag(Sb[:n]) @ np.matrix(Vb[:n, :])

SVD_80_reconstructed_image = np.concatenate([np.expand_dims(R_reconstruction, axis=2),
                                          np.expand_dims(G_reconstruction, axis=2),
                                          np.expand_dims(B_reconstruction, axis=2),
                                          ], axis=2)

SVD_error = tl.norm(image - SVD_80_reconstructed_image, order=2, axis=None)

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

criteria_HOSVD_error_index = 2200
HOSVD_reconstructed_image = np.zeros([L, M, P])
for n in range(0, criteria_HOSVD_error_index):
    i = (ist[0])[n]
    j = (ist[1])[n]
    k = (ist[2])[n]
    HOSVD_reconstructed_image += S[i, j, k] * np.einsum("i, j, k -> ijk", Ur[:, i], Ug[:, j], Ubb[:, k])

HOSVD_error = tl.norm(image - HOSVD_reconstructed_image, order=2, axis=None)
print('SVD tensor reconstruction error :::: ', SVD_error)
print('HOSVD tensor reconstruction error for error index 2200 ::::  ', HOSVD_error)

criteria_HOSVD_error_index = 4000
HOSVD_reconstructed_image = np.zeros([L, M, P])
for n in range(0, criteria_HOSVD_error_index):
    i = (ist[0])[n]
    j = (ist[1])[n]
    k = (ist[2])[n]
    HOSVD_reconstructed_image += S[i, j, k] * np.einsum("i, j, k -> ijk", Ur[:, i], Ug[:, j], Ubb[:, k])

HOSVD_error = tl.norm(image - HOSVD_reconstructed_image, order=2, axis=None)
print('SVD tensor reconstruction error :::: ', SVD_error)
print('HOSVD tensor reconstruction error for error index 4000 ::::  ', HOSVD_error)