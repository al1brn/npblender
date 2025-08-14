import numpy as np

# Définition du dtype : un entier et un tableau 1D de 3 flottants
dtype = [
    ('i', np.int32),
    ('v', np.float64, (3,))  # shape fixe (3,) pour la colonne v
]

# Création des données initiales
data = np.array([
    (0, [1.0, 2.0, 3.0]),
    (1, [4.0, 5.0, 6.0]),
    (2, [7.0, 8.0, 9.0]),
    (4, [1.0, 2.0, 3.0]),
    (5, [4.0, 5.0, 6.0]),
    (6, [7.0, 8.0, 9.0]),
], dtype=dtype)

# Conversion en recarray
rec = data.view(np.recarray)

print(rec)
print("i =", rec.i)
print("v =", rec.v)
print('-'*10)
print(rec.reshape(2, 3).i.shape)
print(rec.reshape(2, 3).v.shape)

rec = np.reshape(rec, (2, 3))
print(rec.i.shape)
print(rec.v.shape)

rec = np.reshape(rec, -1)
print(rec.i.shape)
print(rec.v.shape)

print(rec.size)

