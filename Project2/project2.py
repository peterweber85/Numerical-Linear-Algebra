#%% LIBRARIES
import numpy as np
from scipy import linalg as la
from scipy import sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from timeit import default_timer as timer

#%% HELPERS   
def get_G_and_D(A):
    N = A.shape[0]            
    G = np.where(A > 0, 1., .0)
    C = np.sum(G, axis = 0)
    D = np.zeros(N)
    D[C > 0] = 1/C[C > 0]
    D = sp.diags(D)
    return(sp.csc_matrix(G), sp.csc_matrix(D))
def get_L_and_c(G, vectorized = False):
    if vectorized:
        index_list = np.where(G)    
        c = np.sum(G > 0, axis = 0)
    else:    
        c = np.zeros(G.shape[1]) 
        N = G.shape[0]
        index_list = list()
        for j in range(0, N):
            bool_vec = np.where(G[:,j] > 0, 1., 0.)
            bool_vec = np.where(bool_vec)[0].tolist()
            c[j] = int(len(bool_vec))
            index_list.append(bool_vec)
            
    return(index_list,c)
def solve_linear_system(G, D, m, use_bicgstab = False, tol_ = 1e-12):
    N = G.shape[0]
    left_hand = sp.eye(N) - (1 - m)*G.dot(D)
    e = np.ones(N)
    if use_bicgstab:
        solution = sp.linalg.bicgstab(left_hand, e, tol = tol_)
        solution = solution[0]/np.sum(solution[0])
    else:
        solution = spsolve(left_hand, e)
        solution = solution/np.sum(solution)
    return(solution)
def solve_power_method(G, D, m, iter_max, tol, print_ = True):
    N = G.shape[0]            

    e = np.ones((N,1))
    z = np.ones((N,1))
    z[(np.diag(D.todense()) > 0).T] = m
    z = z/N
    x_k = np.random.random((N,1))

    iterations = 0
    convergence = False
    for i in range(iter_max):
        iterations += 1
        x_k_p1 = (1 - m)*G.dot(D).dot(x_k) +  e * np.asscalar((z.T).dot(x_k))
        x_k_p1 = x_k_p1/np.linalg.norm(x_k_p1)
        error = np.linalg.norm(x_k_p1 - x_k, ord = np.inf)
        if error < tol:
            convergence = True
            if print_:
                print("The power method needed", iterations,\
                "iterations to reach tolerance", tol)
            break
        x_k = x_k_p1

    if not convergence:
        print("The power method did not reach the desired tolerance after", \
        iter_max, "iterations!!!")
    solution = np.array(x_k/np.sum(x_k)).ravel()
    if print_:
        print("The solution of the power method is:\n", solution)
    return(solution, iterations)
def convergence_power_method(p, G, D, m, iter_max):
    N = G.shape[0]            

    e = np.ones((N,1))
    z = np.ones((N,1))
    z[(np.diag(D.todense()) > 0).T] = m
    z = z/N
    x_k = np.random.random((N,1))

    quotient = np.zeros(iter_max)
    for i in range(iter_max):
        x_k_p1 = (1 - m)*G.dot(D).dot(x_k) +  e * np.asscalar((z.T).dot(x_k))
        x_k_p1 = x_k_p1/np.linalg.norm(x_k_p1)
        quotient[i] = np.linalg.norm(x_k_p1 - p, 1)/np.linalg.norm(x_k - p, 1)
        x_k = x_k_p1
    return(quotient)
def get_x_without_storing(x, L, c, m, rows, cols, n, N, vectorized = False):
    if vectorized:
        xc = x
        x = np.zeros(n)
        x = x + (xc[c == 0]/n).sum()
        for j in range(N):
            x[rows[j]] = x[rows[j]] + xc[cols[j]]/c[cols[j]]
        x=(1-m)*x+m/n
        x=x/x.sum()
    else:    
        xc=x
        x=np.zeros(n)
        for j in range(0,n):
            if c[j]==0:
                x=x+xc[j]/n
            else:
                for i in L[j]:
                    x[i]=x[i]+xc[j]/c[j]
        x=(1-m)*x+m/n
    return(x)
def iterate_without_storing(L, c, m, iter_max, tol, vectorized = False, print_ = True):
    x0 = np.random.random(len(c))
    iterations = 0

    rows = L[0]
    cols = L[1]
    N = len(L[0])
    if vectorized:
        n = len(c)
    else:
        n = len(L)

    for i in range(iter_max):
        iterations += 1
        x1 = get_x_without_storing(x0, L, c, m, rows, cols, n, N, vectorized) 
        error = np.linalg.norm(x1-x0, ord = np.inf)
        if error < tol:
            solution = np.array(x1/np.sum(x1)).ravel()
            if print_:
                print("The power method without storing reached tolerance", tol,\
                "after", i, "iterations!!!")
                print("The solution is:\n", solution)
            return(solution, iterations)
        x0 = x1
    print("No convergence!!!")
def read_file(path, filename):
    file = open(path + filename, "r") 
    for line in file:
        if not line.startswith("%"):
            indices = line.split()
            num_rows = int(indices[0])
            num_cols = int(indices[1])
            elements = int(indices[2])
            break
    matrix = np.zeros((num_rows, num_cols))
    
    rows = list()
    cols = list()
    for line in file:
        indices = line.split()
        rows.append(int(indices[0])-1)
        cols.append(int(indices[1])-1)

    rows = np.asarray(rows)
    cols = np.asarray(cols)
    values = np.ones(elements)

    matrix_sparse = sp.csc_matrix( (values,(rows, cols)), shape = (num_rows, num_cols))
    matrix = np.asarray(matrix_sparse.todense())
    return(matrix, matrix_sparse)        

#%% PARAMETERS
vectorize = False # This vectorizes the computation of the power method without storing
plot_tolerance = False

#%% READ MATRIX
GNU, GNU_SPARSE = read_file("", "p2p-Gnutella30.mtx")

#%%
A_11 = np.array([[0, 0, 1/2., 1/2., 0], \
                 [1/3., 0, 0, 0, 0], \
                 [1/3., 1/2. , 0, 1/2., 1], \
                 [1/3., 1/2., 0, 0, 0],\
                 [0, 0, 1/2., 0, 0]])
print("The link matrix for task 1 exercise 1 is: \n")
print(A_11)



#%%
w, v = np.linalg.eig(A_11)
eigenvector_1 = v[:,0].real
eigenvector_1
sum_ev_1 = np.sum(eigenvector_1)
norm_ev_1 = eigenvector_1/sum_ev_1

print("\nThe eigenvector to eigenvalue 1 is:\n ", norm_ev_1)
print("The importance score of page 3 has therefore been boosted,\n \
so that page 3 is now the most important page in the given web")

#%%
A_14 = np.array([[0, 0, 0, 1/2], \
                 [1/3., 0, 0, 0], \
                 [1/3., 1/2. , 0, 1/2.], \
                 [1/3., 1/2., 0, 0]])
print("\nThe link matrix for task 1 exercise 4 is: \n")
print(A_14)

w, v = np.linalg.eig(A_14)
sum_evec_4 = np.sum(v[:,3].real)
evec_4 = v[:,3].real/sum_evec_4
max_eval_4 = np.max(w)

print("\nThe largest eigenvalue of this link matrix is ", max_eval_4.real)
print("With corresponding eigenvector ", evec_4)

#%% DEFINING EXAMPLE MATRICES
FIG_21 = np.array([[0., 0, 1, 1/2], 
                    [1/3, 0, 0, 0],
                    [1/3, 1/2, 0, 1/2],
                    [1/3, 1/2, 0, 0]])

FIG_22 = np.array([[0., 1, 0, 0, 0],
                    [1., 0, 0, 0, 0],
                    [0, 0, 0, 1, 1/2],
                    [0, 0, 1, 0, 1/2],
                    [0, 0, 0, 0, 0]])


#%%
print("\n\nTASK 2")
print("\nPART 1: SOLVING THE LINEAR SYSTEM")

print("\nBy solving the linear system, the solution vector of the\n\
link matrix of Fig. 2.1 is (m = 0.15):")
print(solve_linear_system(get_G_and_D(FIG_21)[0], get_G_and_D(FIG_21)[1], 0.15))

print("\nBy solving the linear system, the solution vector of the\n\
link matrix in task 1 exercise 1 is (m = 0):")
print(solve_linear_system(get_G_and_D(A_11)[0], get_G_and_D(A_11)[1], 0))

print("\nBy solving the linear system, the solution vector of the\n\
link matrix in task 1 exercise 4 is (m = 0):")
print(solve_linear_system(get_G_and_D(A_14)[0], get_G_and_D(A_14)[1], 0))

print("\nBy solving the linear system, the solution vector of the\n\
link matrix of Fig. 2.2 is (m = 0.15):")
print(solve_linear_system(get_G_and_D(FIG_22)[0], get_G_and_D(FIG_22)[1], 0.15))

#%%
print("\nPART 2: POWER METHOD")
print("\nLink matrix of Fig. 2.1 (m = 0.15).")
s=solve_power_method(get_G_and_D(FIG_21)[0], get_G_and_D(FIG_21)[1], 0.15, 1000, 1e-10)

print("\nLink matrix of task 1 exercies 1 (m = 0).")
s=solve_power_method(get_G_and_D(A_11)[0], get_G_and_D(A_11)[1], 0, 1000, 1e-10)

print("\nLink matrix of task 1 exercise 4 (m = 0).")
s=solve_power_method(get_G_and_D(A_14)[0], get_G_and_D(A_14)[1], 0, 1000, 1e-10)

print("\nLink matrix of Fig. 2.2 (m = 0.15)")
s=solve_power_method(get_G_and_D(FIG_22)[0], get_G_and_D(FIG_22)[1], 0.15, 1000, 1e-10)

#%%
print("\nPART 3: POWER METHOD WITHOUT STORING")
print("\nLink matrix of Fig. 2.1 (m = 0.15).")
s=iterate_without_storing(get_L_and_c(FIG_21, vectorize)[0], \
get_L_and_c(FIG_21, vectorize)[1], 0.15, 1000, 1e-10, vectorize)

print("\nLink Matrix of task 1 exercise 1 (m = 0).")
s=iterate_without_storing(get_L_and_c(A_11, vectorize)[0],\
 get_L_and_c(A_11, vectorize)[1], 0, 1000, 1e-10, vectorize)

print("\nLink matrix of task 1 exercise 4 (m = 0).")
s=iterate_without_storing(get_L_and_c(A_14, vectorize)[0],\
get_L_and_c(A_14, vectorize)[1], 0, 1000, 1e-10, vectorize)

print("\nLink matrix of Fig. 2.2 (m = 0.15).")
s=iterate_without_storing(get_L_and_c(FIG_22, vectorize)[0],\
get_L_and_c(FIG_22, vectorize)[1], 0.15, 1000, 1e-10, vectorize)

#%%
print("\n\nTASK 3")
print("PART 1: Solve a linear system")

#%%
print("Decomposing Gnutella Matrix into G*D...be patient")
G_, D_ = get_G_and_D(GNU)

#%%
print("Decomposing Gnutella Matrix into L and c...be patient")
L_, c_ = get_L_and_c(GNU, vectorize)

#%%
print("Solving linear system with Gnutella matrix...be patient")
start = timer()
solution_linear_system = solve_linear_system(G_, D_, 0.15)
elapsed = timer() - start
print("\nTo solve the linear system with scipy.sparse.spsolve it took", round(elapsed, 2), "s!!!")
print("The page rank eigenvector is:\n")
print(solution_linear_system)
#%%
print("The solution has", len(np.unique(solution_linear_system)), "unique elements!")

#%%
tol = 1e-12
start = timer()
solution_linear_system_bicg = solve_linear_system(G_, D_, 0.15, use_bicgstab = True, tol_ = tol)
elapsed = timer() - start
print("To solve the linear system with scipy.sparse.linalg.bicgstab it took", round(elapsed, 3), "s!!!")
print("The page rank eigenvector using tolerance", tol,"is:\n")
print(solution_linear_system_bicg)
print("The solution has", len(np.unique(solution_linear_system_bicg)), "unique elements!")

#%%
print("\n")
start = timer()
solution_power_method = solve_power_method(G_, D_, 0.15, 1000, 1e-12)
elapsed = timer() - start
print("It took", round(elapsed,2), "s to reach tolerance!")
print("The solution has", len(np.unique(solution_power_method[0])), "unique elements!")

#%%
print("\n")
start = timer()
solution_without_storing = iterate_without_storing(L_, c_, 0.15, 1000, 1e-12, vectorize)
elapsed = timer() - start
print("It took", round(elapsed, 2), "s to reach tolerance!")
print("The solution has", len(np.unique(solution_without_storing[0])), "unique elements!")

#%%


#%%
if plot_tolerance:
    tol_ = np.logspace(-2,-12,num = 21)
    iterations = np.zeros(len(tol_))
    elapsed_time = np.zeros(len(tol_))
    for idx_ in range(len(tol_)):
        start = timer()
        solution_power_method = solve_power_method(G_, D_, 0.15, 1000, tol_[idx_], False)
        iterations[idx_] = solution_power_method[1]
        elapsed_time[idx_] = timer() - start

    #%%
    f, (ax1, ax2) = plt.subplots(2, 1, sharex = True)

    ax1.loglog(tol_, elapsed_time)
    ax1.set_title("Power method")
    ax1.set_ylabel("Elapsed time (s)")

    ax2.loglog(tol_, iterations)
    ax2.set_ylabel("Iterations")
    ax2.set_xlabel("Tolerance")
    plt.show()

    #%%
    tol_ = np.logspace(-2,-12,num = 21)
    iterations = np.zeros(len(tol_))
    elapsed_time = np.zeros(len(tol_))
    for idx_ in range(len(tol_)):
        start = timer()
        solution_without_storing = iterate_without_storing(L_, c_, 0.15, 1000, tol_[idx_], vectorize, False)
        iterations[idx_] = solution_without_storing[1]
        elapsed_time[idx_] = timer() - start

    #%%
    f, (ax1, ax2) = plt.subplots(2, 1, sharex = True)

    #%%
    ax1.loglog(tol_, elapsed_time)
    ax1.set_title("Power method without storing")
    ax1.set_ylabel("Elapsed time (s)")

    ax2.loglog(tol_, iterations)
    ax2.set_ylabel("Iterations")
    ax2.set_xlabel("Tolerance")
    plt.show()


