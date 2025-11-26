import numpy as np

# matriz I (9x9)
I = np.array([
[96,80,16,80,96,48,96,64,96],
[32,32,96,16,48,48,64,16,48],
[80,64,80,96,16,32,32,16,80],
[16,16,48,96,96,96,32,48,48],
[32,32,64,64,32,64,16,80,32],
[48,80,64,16,80,96,16,32,48],
[64,48,48,80,16,64,80,48,16],
[16,16,64,32,16,16,64,48,16],
[80,96,80,96,64,96,16,48,80]
], dtype=float)

# matriz F (9x9)
F = np.array([
[0,0,0,0,0,0,0,0,0],
[0,56,65,57,41,39,44,44,0],
[0,57,80,68,69,44,39,35,0],
[0,37,60,65,64,58,53,44,0],
[0,52,55,66,65,45,33,48,0],
[0,60,65,41,57,70,45,41,0],
[0,53,44,39,44,50,47,43,0],
[0,52,67,56,44,45,49,46,0],
[0,0,0,0,0,0,0,0,0]
], dtype=float)


positions = [(3,3),(3,4),(3,5),(4,4),(5,5),(5,4),(4,5),(6,6),(6,5)]  
A_rows = []
b = []
for (r1,c1) in positions:
    r = r1-1
    c = c1-1
    row = []
    for kr in (-1,0,1):
        for kc in (-1,0,1):
            rr = r + kr
            cc = c + kc
            if 0 <= rr < 9 and 0 <= cc < 9:
                row.append(I[rr,cc])
            else:
                row.append(0.0)
    A_rows.append(row)
    b.append(F[r,c])

A = np.array(A_rows, dtype=float)
b = np.array(b, dtype=float)

k, residuals, rank, s = np.linalg.lstsq(A,b, rcond=None)
K = k.reshape((3,3))
print("Kernel K =")
print(K)
print("Residuals:", residuals, "Rank:", rank)
