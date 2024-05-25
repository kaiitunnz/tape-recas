import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs


def scipyCSC_to_python(A):
  m, n = A.shape
  colPtr = np.asarray(A.indptr, dtype=np.int64) + 1
  rowVal = np.asarray(A.indices, dtype=np.int64) + 1
  nzVal = A.data.astype(np.float64)
  return {"m": m, "n": n, "colPtr": colPtr, "rowVal": rowVal, "nzVal": nzVal}


def read_arxiv(filename):
  I, J = [], []
  with open(filename, "r") as f:
    for line in f:
      if line.startswith("#"):
        continue
      data = line.strip().split(",")
      I.append(int(data[0]))
      J.append(int(data[1]))
  I = np.asarray(I, dtype=np.int64) + 1
  J = np.asarray(J, dtype=np.int64) + 1
  n = max(np.max(I), np.max(J))
  A = csc_matrix((np.ones(len(I)), (I, J)), shape=(n, n))
  A = np.maximum(A, A.T)
  A = np.minimum(A, 1.0)
  return A


def main(PyA, k):
  m, n = PyA["m"], PyA["n"]
  colPtr = np.asarray(PyA["colPtr"], dtype=np.int64)
  rowVal = np.asarray(PyA["rowVal"], dtype=np.int64)
  nzVal = PyA["nzVal"].astype(np.float64)
  A = csc_matrix((nzVal, (rowVal-1, colPtr-1)), shape=(m, n))

  d = np.sum(A, axis=1)
  tau = np.sum(d) / len(d)
  N = m

  # normalized regularized laplacian
  D = np.diag(1.0 / np.sqrt(d + tau))
  Aop = np.dot(A, D) + (tau / N) * np.ones((N, N))

  NRL = np.eye(N) + np.dot(D, np.dot(Aop, D))

  (Lambda, V) = eigs(NRL, k=k, tol=1e-6, which="SM")

  # project onto the top k eigenvectors
  V = V[:, :k]

  # QR decomposition for pivoting
  Q, R, piv = np.linalg.qr(V.T, mode="reduced")

  # use SVD on pivoted subspace
  U, s, Vt = np.linalg.svd(V[piv, :], full_matrices=False)

  # reconstruct final embedding
  SCDM_V = np.dot(V, np.dot(U, Vt))

  return SCDM_V