
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import math
from timeit import default_timer as timer
from scipy import linalg
from scipy.sparse import *

# General functions
def Newton_step(lamb0,dlamb,s0,ds):
    alp=1;
    idx_lamb0=np.array(np.where(dlamb<0))
    if idx_lamb0.size>0:
        alp = min(alp,np.min(-lamb0[idx_lamb0]/dlamb[idx_lamb0]))
    idx_s0=np.array(np.where(ds<0))
    if idx_s0.size>0:
        alp = min(alp,np.min(-s0[idx_s0]/ds[idx_s0]))
    return alp

# Generates the plots
def plot_elapsed_time(elapsed_time_, c_, n_max_):
    u_ = np.linspace(1, n_max_ - 1, n_max_ - 1)

    plt.loglog(u_, elapsed_time_, label = "Algorithm")
    plt.loglog(u_, c_*u_**3, label = "Fit for large n")
    plt.ylim(1e-3, 1e2)
    plt.xlabel("Matrix size")
    plt.ylabel("Time to converge (s)")
    plt.legend()

# Executes the task specified in the input
def execute_task(n_max_, modulo_print_, task_ = ""):
    elapsed_time = np.zeros(n_max_ - 1)

    print("task number: ", task_,"\n")
    for i in range(1, n_max_):
        start = timer()

        if (i + 1) % modulo_print_ == 0 or (i + 1) == n_max_:
            print_conv = True
            print("Matrix size: ", i + 1)
        else:
            print_conv = False

        if task_ == "C3":
            error = solve_inequality_constraints_case(\
                get_new_z, get_M_kkt, n_ = i, p_ = 0, m_ = 2*i,iter_max_ = 100,\
                eps_ = 1e-16, print_iter_ = False, print_conv_ = print_conv)
        elif task_ == "C4 strategy 1":
            error = solve_inequality_constraints_case(\
                get_new_z_s1, get_matrix_s1, n_ = i, p_ = 0, m_ = 2*i,iter_max_ = 100,\
                eps_ = 1e-16, print_iter_ = False, print_conv_ = print_conv)
        elif task_ == "C4 strategy 2":
            error = solve_inequality_constraints_case(\
                get_new_z_s2, get_G_hat_s2, n_ = i, p_ = 0, m_ = 2*i,iter_max_ = 100,\
                eps_ = 1e-16, print_iter_ = False, print_conv_ = print_conv)
                
        elapsed_time[i-1] = timer() - start

        if print_conv:
            print("The algorithm needed ", np.round(elapsed_time[i-1],4), " s to converge")
            print("Error between x and g in last iteration: E = x + g = ", error,"\n")
            
    return(elapsed_time)

# M_kkt matrix
def get_M_kkt(lambda_,s_, G_, A_, C_):
    n_ = G_.shape[0]
    p_ = A_.shape[1]
    m_ = len(lambda_)
    M_kkt1_ = np.c_[G_, -A_, -C_, np.zeros((n_,m_))]
    M_kkt2_ = np.c_[-A_.T, np.zeros((p_,p_)), np.zeros((p_,m_)), np.zeros((p_,m_))]
    M_kkt3_ = np.c_[-C_.T, np.zeros((m_,p_)), np.zeros((m_,m_)), np.identity(m_)]
    M_kkt4_ = np.c_[np.zeros((m_,n_)), np.zeros((m_,p_)), np.diag(s_), np.diag(lambda_)]
    return(np.r_[M_kkt1_, M_kkt2_, M_kkt3_, M_kkt4_])

# Define F(z)
def F(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_):
    F1_ = np.dot(G_, x_) + g_ + np.dot(-A_, gamma_) + np.dot(-C_, lambda_)
    F2_ = np.dot(-A_.T, x_) + b_
    F3_ = np.dot(-C_.T, x_) + d_ + s_
    F4_ = lambda_ * s_
    return(np.r_[F1_, F2_, F3_, F4_])

# Define right-hand side vectors for unconstraint case
def r_1(x_, gamma_, lambda_, s_, G_, A_,C_,g_, b_, d_):
    return(np.dot(G_, x_) + g_ + np.dot(-A_, gamma_) + np.dot(-C_, lambda_))
def r_2(x_, gamma_, lambda_, s_, G_, A_,C_,g_, b_, d_):
    return(np.dot(-C_.T, x_) + d_ + s_)
def r_3(x_, gamma_, lambda_, s_, G_, A_,C_,g_, b_, d_):
    return(lambda_ * s_)

# Execute 6 steps C2
def get_new_z(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_, e_):
    n_ = len(x_)
    p_ = len(gamma_)
    m_ = len(lambda_)
    z_ = np.r_[x_,gamma_,lambda_,s_] 
    
    # Step 1
    F_ = -F(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_)
    M_kkt_ = get_M_kkt(lambda_,s_, G_, A_, C_)
    dz_ = np.linalg.solve(M_kkt_, F_)
    dlambda_ = dz_[(n_+p_):(n_+p_+m_)]
    ds_ = dz_[(n_+p_+m_):(n_+p_+2*m_)]

    # Step 2
    alpha_ = Newton_step(lambda_,dlambda_,s_,ds_)

    # Step 3
    mu_ = np.inner(s_.T, lambda_)/m_
    mu_tilde_ = np.inner((s_ + alpha_*ds_).T, lambda_ + alpha_*dlambda_)/m_
    sigma_ = (mu_tilde_/mu_) ** 3

    # Step 4
    s_correction_ = np.dot(np.dot(np.diag(ds_),np.diag(dlambda_)),e_) - sigma_*mu_*e_
    F_[(n_+p_+m_):(n_+p_+2*m_)] =  F_[(n_+p_+m_):(n_+p_+2*m_)] - s_correction_
    dz_ = np.linalg.solve(M_kkt_, F_)
    dx_ = dz_[0:n_]
    dgamma_ = dz_[n_:(n_+p_)]
    dlambda_ = dz_[(n_+p_):(n_+p_+m_)]
    ds_ = dz_[(n_+p_+m_):(n_+p_+2*m_)]

    # Step 5
    alpha_ = Newton_step(lambda_,dlambda_,s_,ds_)

    # Step 6
    z_ = z_ + 0.95*alpha_*dz_
    x_ = z_[0:n_]
    gamma_ = z_[n_:(n_+p_)]
    lambda_ = z_[(n_+p_):(n_+p_+m_)]
    s_ = z_[(n_+p_+m_):(n_+p_+2*m_)]
    
    return(z_, mu_, -F(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_))

# Inequality constraints case 
# The first input to this function is a function that implements the 6 steps
# The second input is a function that gets the matrix that has to be solved
def solve_inequality_constraints_case(fun_get_new_z, fun_get_matrix ,n_,p_,m_,iter_max_,eps_,print_iter_, print_conv_):
    G_ = np.identity(n_)
    A_ = np.zeros((n_,p_))
    C_ = np.c_[np.identity(n_), -np.identity(n_)]

    x_ = np.zeros(n_)
    gamma_ = np.zeros(p_)
    s_ = np.ones(m_)
    lambda_ = np.ones(m_)
    g_ = np.random.normal(size = n_) 
    b_ = np.zeros(p_)
    d_ = -10*np.ones(m_)
    e_ = np.ones(m_)
    
    for i_ in range(iter_max_):
        result_ = fun_get_new_z(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_, e_)
        z_ = result_[0]    
        mu_ = result_[1]
        righthand_ = result_[2]

        x_ = z_[:n_]
        gamma_ = z_[n_:n_+p_]
        lambda_ = z_[n_+p_:n_+p_+m_]
        s_ = z_[n_+p_+m_:n_+p_+2*m_]  

        r_l_ = np.linalg.norm(righthand_[0:n_])
        r_c_ = np.linalg.norm(righthand_[n_+p_:n_+p_+m_])

        if print_iter_:
            print("iteration: ", i_)
            print("r_L: ", r_l_)
            print("r_C: ", r_c_)
            print("mu: ", mu_, "\n")

        if math.fabs(mu_) < eps_ or r_l_ < eps_ or r_c_ < eps_:
            if print_conv_:
                M_ = fun_get_matrix(lambda_, s_, G_, A_, C_)
                condition_number_ = np.linalg.cond(M_)
                print("Convergence after ", i_, " iterations!!!")
                print("The condition number of the matrix is: ", np.round(condition_number_,4))
            break
        error_ = np.sum(x_ + g_)
    return(error_)


# Matrix for C4 strategy 1
def get_matrix_s1(lambda_, s_, G_, A_, C_):
    L_inv_ = np.linalg.inv(np.diag(lambda_))
    S_ = np.diag(s_)
    row1_ = np.c_[G_ , -C_]
    row2_ = np.c_[-C_.T, np.dot(-L_inv_, S_)]
    return(np.r_[row1_, row2_])

# Right hand side C4 strategy 1
def get_righthand_s1(x_,gamma_,lambda_,s_,G_,A_,C_,g_,b_,d_,r_3_corr_):
    L_inv_ = np.linalg.inv(np.diag(lambda_))
    
    row1_ = r_1(x_,gamma_,lambda_,s_,G_,A_,C_,g_,b_,d_)
    row2_ = r_2(x_,gamma_,lambda_,s_,G_,A_,C_,g_,b_,d_)\
            -np.dot(L_inv_,(r_3(x_,gamma_,lambda_,s_,G_,A_,C_,g_,b_,d_) + r_3_corr_))
        
    return(-np.r_[row1_, row2_])

# LDL^T factorization
def get_LDLT(A_):
    n_ = A_.shape[0]
    L_ = np.eye(n_)
    D_ = np.zeros((n_, 1))
    for i_ in range(n_):
        D_[i_] = A_[i_, i_] - np.dot(L_[i_, 0:i_] ** 2, D_[0:i_])
        for j_ in range(i_ + 1, n_):
            L_[j_, i_] = (A_[j_, i_] - np.dot(L_[j_, 0:i_] * L_[i_, 0:i_], D_[0:i_])) / D_[i_]
    D_ = np.eye(n_) * D_
    return(D_, L_)

# Forward, backward LDL^T case 
def solve_LDLT_system(Diag_, Lower_, rh_):
    dw_ = linalg.solve_triangular(Lower_, rh_, unit_diagonal=False, lower = True)
    dv_ = dw_/np.diag(Diag_)
    du_ = linalg.solve_triangular(Lower_.T, dv_, unit_diagonal=False)
    return(du_)
def get_and_solve_LDLT(M_, rh_):
    Diag_, Lower_ = get_LDLT(M_)
    return(solve_LDLT_system(Diag_, Lower_, rh_))
   
# Forward, backward Cholesky case
def solve_cholesky_system(Lower_, rh_):
    dv_ = linalg.solve_triangular(Lower_, rh_, lower = True)
    du_ = linalg.solve_triangular(Lower_.T, dv_)
    return(du_)

# delta s C4 strategy 1
def get_ds_s1(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_, dlambda_, r3_corr_):
    L_inv_ = np.linalg.inv(np.diag(lambda_))
    output_ = (-r_3(x_, gamma_, lambda_, s_, G_, A_,C_,g_, b_, d_)-r3_corr_\
               - np.dot(np.diag(s_), dlambda_)).dot(L_inv_)
    return(output_)
    
# 6 steps C4 strategy 1
def get_new_z_s1(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_, e_):
    n_ = len(x_)
    p_ = len(gamma_)
    m_ = len(lambda_)
    z_ = np.r_[x_,gamma_,lambda_,s_] 
    
    # Step 1
    M_s1_ = get_matrix_s1(lambda_, s_, G_, A_, C_)
    r3_corr_ = np.zeros(m_)
    rh_s1_ = get_righthand_s1(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_, r3_corr_)
    Diag_, Lower_ = get_LDLT(M_s1_)
    dx_dlambda_ = solve_LDLT_system(Diag_, Lower_, rh_s1_)
    dx_ = dx_dlambda_[:n_]
    dlambda_ = dx_dlambda_[(n_+p_):(n_+p_+m_)]
    ds_ = get_ds_s1(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_, dlambda_, r3_corr_)
    
    # Step 2
    alpha_ = Newton_step(lambda_,dlambda_,s_,ds_)
    
    # Step 3
    mu_ = np.inner(s_.T, lambda_)/m_
    mu_tilde_ = np.inner((s_ + alpha_*ds_).T, lambda_ + alpha_*dlambda_)/m_
    sigma_ = (mu_tilde_/mu_) ** 3
    
    # Step 4
    r3_corr_ = np.dot(np.dot(np.diag(ds_),np.diag(dlambda_)),e_) - sigma_*mu_*e_
    rh_s1_ = get_righthand_s1(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_, r3_corr_)
    dx_dlambda_ = solve_LDLT_system(Diag_, Lower_, rh_s1_)
    dx_ = dx_dlambda_[:n_]
    dlambda_ = dx_dlambda_[(n_+p_):(n_+p_+m_)]
    ds_ = get_ds_s1(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_, dlambda_,r3_corr_)
    dz_ = np.r_[dx_dlambda_, ds_]
    
    # Step 5
    alpha_ = Newton_step(lambda_,dlambda_,s_,ds_)
    
    ## Step 6
    z_ = z_ + 0.95*alpha_*dz_
    x_ = z_[0:n_]
    lambda_ = z_[(n_+p_):(n_+p_+m_)]
    s_ = z_[(n_+p_+m_):(n_+p_+2*m_)]
    
    return(z_, mu_, -F(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_))

# Matrix C4 strategy 2
def get_G_hat_s2(lambda_,s_, G_, A_, C_):
    S_inv_ = np.linalg.inv(np.diag(s_))
    L_ = np.diag(lambda_)
    G_hat_ = G_ + C_.dot(S_inv_).dot(L_).dot(C_.T)
    return(G_hat_)

# Righthand side C4 strategy 2  
def get_righthand_s2(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_, r3_corr_):
    S_inv_ = np.linalg.inv(np.diag(s_))
    L_ = np.diag(lambda_)
    r_hat_ = -C_.dot(S_inv_).dot(-r_3(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_)\
                                 -r3_corr_ + L_.dot(r_2(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_)))
    r1_ = r_1(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_)
    return(-r1_-r_hat_)

# delta lambda C4 strategy 2
def get_dlambda_s2(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_, r3_corr_,dx_):
    S_inv_ = np.linalg.inv(np.diag(s_))
    L_ = np.diag(lambda_)
    dlambda_ = S_inv_.dot(-r_3(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_)\
                                 -r3_corr_ + L_.dot(r_2(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_)))\
               -S_inv_.dot(L_).dot(C_.T).dot(dx_)
    return(dlambda_)

# delta s C4 strategy 2
def get_ds_s2(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_,dx_):
    ds_ = -r_2(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_) + (C_.T).dot(dx_)
    return(ds_)
  
# 6 steps C4 strategy 2  
def get_new_z_s2(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_, e_):
    n_ = len(x_)
    p_ = len(gamma_)
    m_ = len(lambda_)
    z_ = np.r_[x_,gamma_,lambda_,s_] 
    
    # Step 1
    G_hat_s2_ = get_G_hat_s2(lambda_,s_, G_, A_, C_)
    r3_corr_ = np.zeros(m_)
    rh_s2_ = get_righthand_s2(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_, r3_corr_)
    Lower_ = np.linalg.cholesky(G_hat_s2_)
    dx_ = solve_cholesky_system(Lower_, rh_s2_)
    dlambda_ = get_dlambda_s2(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_, r3_corr_,dx_)
    ds_ = get_ds_s2(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_,dx_)
    
    # Step 2
    alpha_ = Newton_step(lambda_,dlambda_,s_,ds_)
    
    # Step 3
    mu_ = np.inner(s_.T, lambda_)/m_
    mu_tilde_ = np.inner((s_ + alpha_*ds_).T, lambda_ + alpha_*dlambda_)/m_
    sigma_ = (mu_tilde_/mu_) ** 3
    
    # Step 4
    r3_corr_ = np.dot(np.dot(np.diag(ds_),np.diag(dlambda_)),e_) - sigma_*mu_*e_
    rh_s2_ = get_righthand_s2(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_, r3_corr_)
    dx_ = solve_cholesky_system(G_hat_s2_, rh_s2_)
    dlambda_ = get_dlambda_s2(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_, r3_corr_,dx_)
    ds_ = get_ds_s2(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_,dx_)
    dz_ = np.r_[dx_, dlambda_, ds_]
    
    # Step 5
    alpha_ = Newton_step(lambda_,dlambda_,s_,ds_)
    
    ## Step 6
    z_ = z_ + 0.95*alpha_*dz_
    x_ = z_[0:n_]
    lambda_ = z_[(n_+p_):(n_+p_+m_)]
    s_ = z_[(n_+p_+m_):(n_+p_+2*m_)]
    
    return(z_, mu_, -F(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_))

# C5 read files
def read_file(path_, filename_, matrix_):
    file_ = open(path_ + filename_, "r") 
    row_ = list()
    col_ = list()
    value_ = list()
    if matrix_:
        for line_ in file_:
            next_line_ = line_.split()
            row_.append(int(next_line_[0])-1)
            col_.append(int(next_line_[1])-1)
            value_.append(float(next_line_[2]))
    else:
        for line_ in file_:
            next_line_ = line_.split()
            row_.append(int(next_line_[0])-1)
            col_.append(0)
            value_.append(float(next_line_[1]))
    return(np.asarray(row_), np.asarray(col_), np.asarray(value_))    

# C5 read matrices
def get_matrices(path_,n_,m_,p_):
    row_,col_,value_=read_file(path_,"G.dad",True)
    G_ = csr_matrix( (value_,(row_,col_)), shape=(n_, n_)).todense()
    G_ = np.triu(G_,1) + G_.T
    row_,col_,value_=read_file(path_,"A.dad",True)
    A_= csr_matrix( (value_,(row_,col_)), shape=(n_, p_)).todense()
    row_,col_,value_=read_file(path_,"C.dad",True)
    C_= csr_matrix( (value_,(row_,col_)), shape=(n_, m_)).todense()
    return(np.asarray(G_),np.asarray(A_),np.asarray(C_))
    
# C5 read vectors
def get_vectors(path_,n_,m_,p_):
    row_,col_,value_=read_file(path_,"b.dad",False)
    b_= csr_matrix( (value_,(row_,col_)), shape=(p_, 1)).todense()
    b_ = np.asarray(b_).ravel()
    row_,col_,value_=read_file(path_,"d.dad",False)
    d_= csr_matrix( (value_,(row_,col_)), shape=(m_, 1)).todense()
    d_ = np.asarray(d_).ravel()
    try:
    	row_,col_,value_=read_file(path_,"g_.dad",False)
    except:	
    	row_,col_,value_=read_file(path_,"g.dad",False)
    g_= csr_matrix( (value_,(row_,col_)), shape=(n_, 1)).todense()
    g_ = np.asarray(g_).ravel()
    return(b_,d_,g_)

# C5 solves general case using 6 steps from above (get_new_z)
# The first input to this function is a function that implements the 6 steps
# The second input is a function that gets the matrix that has to be solved
def solve_general_case(fun_get_new_z, fun_get_matrix, path_,n_,p_,m_,iter_max_,eps_,print_iter_, print_conv_):
    G_, A_, C_ = get_matrices(path_,n_,m_,p_)
    b_, d_, g_ = get_vectors(path_,n_,m_,p_)

    x_ = np.zeros(n_)
    gamma_ = np.ones(p_)
    s_ = np.ones(m_)
    lambda_ = np.ones(m_)
    e_ = np.ones(m_)

    for i_ in range(iter_max_):
        result_ = fun_get_new_z(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_, e_)
        z_ = result_[0]    
        mu_ = result_[1]
        righthand_ = result_[2]

        x_ = z_[:n_]
        gamma_ = z_[n_:n_+p_]
        lambda_ = z_[n_+p_:n_+p_+m_]
        s_ = z_[n_+p_+m_:n_+p_+2*m_]  

        r_l_ = np.linalg.norm(righthand_[0:n_])
        r_c_ = np.linalg.norm(righthand_[n_+p_:n_+p_+m_])

        if print_iter_:
            print("iteration: ", i_)
            print("r_L: ", r_l_)
            print("r_C: ", r_c_)
            print("mu: ", mu_, "\n")

        if math.fabs(mu_) < eps_ or r_l_ < eps_ or r_c_ < eps_:
            if print_conv_:
                M_ = fun_get_matrix(lambda_, s_, G_, A_, C_)
                condition_number_ = np.linalg.cond(M_)
                print("Convergence after ", i_, " iterations!!!")
                print("The condition number of the matrix is: ", np.round(condition_number_,4))
                solution_ = 0.5*(x_.T).dot(G_).dot(x_) + (g_.T).dot(x_)
                print("The solution f(x) is: ", solution_)
            break
    return(solution_)

# For debuggin
def check_dimensions(path_, n_,m_,p_):
    G_, A_, C_ = get_matrices(path_ ,n_,m_,p_)
    b_, d_, g_ = get_vectors(path_ ,n_,m_,p_)

    print("Shape of G: ", G_.shape)
    print("Shape of A: ", A_.shape)
    print("Shape of C: ", C_.shape)

    print("Shape of g: ", g_.shape)
    print("Shape of b: ", b_.shape)
    print("Shape of d: ", d_.shape)

#check_dimensions("./optpr1/", 100, 200, 50)
#check_dimensions("./optpr2/", 1000, 2000, 500)

# define right hand vectors for general case
def r_1_c6(x_, gamma_, lambda_, s_, G_, A_,C_,g_, b_, d_):
    return(np.dot(G_, x_) + g_ + np.dot(-A_, gamma_) + np.dot(-C_, lambda_))

def r_2_c6(x_, gamma_, lambda_, s_, G_, A_,C_,g_, b_, d_):
    return(np.dot(-A_.T, x_))

def r_3_c6(x_, gamma_, lambda_, s_, G_, A_,C_,g_, b_, d_):
    return(np.dot(-C_.T, x_) + d_ + s_)

def r_4_c6(x_, gamma_, lambda_, s_, G_, A_,C_,g_, b_, d_):
    return(lambda_ * s_)
    
# Matrix for C6
def get_M_c6(lambda_,s_, G_, A_, C_):
    n_ = G_.shape[0]
    p_ = A_.shape[1]
    m_ = len(lambda_)
    L_inv_ = np.linalg.inv(np.diag(lambda_))
    S_ = np.diag(s_)
    M1_c6_ = np.c_[G_, -A_, -C_]
    M2_c6_ = np.c_[-A_.T, np.zeros((p_,p_)), np.zeros((p_,m_))]
    M3_c6_ = np.c_[-C_.T, np.zeros((m_,p_)), np.dot(-L_inv_, S_)]
    return(np.r_[M1_c6_, M2_c6_, M3_c6_])

# Right hand side C6
def get_righthand_c6(x_,gamma_,lambda_,s_,G_,A_,C_,g_,b_,d_,r4_corr_):
    L_inv_ = np.linalg.inv(np.diag(lambda_))
    
    rh1_c6_ = -r_1_c6(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_)
    rh2_c6_ = -r_2_c6(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_)
    rh3_c6_ = -r_3_c6(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_)\
              + np.dot(L_inv_, r4_corr_ + r_4_c6(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_))
    return(np.r_[rh1_c6_, rh2_c6_, rh3_c6_])
     
# delta s C6  
def get_ds_c6(x_,gamma_,lambda_,s_,G_,A_,C_,g_,b_,d_,r4_corr_,dlambda_):
    L_inv_ = np.linalg.inv(np.diag(lambda_))
    S_ = np.diag(s_)
    ds_ = L_inv_.dot(-r_4_c6(x_, gamma_, lambda_, s_, G_, A_,C_,g_, b_, d_) - r4_corr_ - S_.dot(dlambda_))
    return(ds_)

# 6 steps C6 
def get_new_z_c6(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_, e_):
    n_ = len(x_)
    p_ = len(gamma_)
    m_ = len(lambda_)
    z_ = np.r_[x_,gamma_,lambda_,s_] 
    
    # Step 1
    M_c6_ = get_M_c6(lambda_, s_, G_, A_, C_)
    r4_corr_ = np.zeros(m_)
    rh_c6_ = get_righthand_c6(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_, r4_corr_)
    
    Lower_ = np.linalg.cholesky((M_c6_.T).dot(M_c6_))   
    dx_dgamma_dlambda_ = solve_cholesky_system(Lower_, (M_c6_.T).dot(rh_c6_))
    dlambda_ = dx_dgamma_dlambda_[(n_+p_):(n_+p_+m_)]
    ds_ = get_ds_c6(x_,gamma_,lambda_,s_,G_,A_,C_,g_,b_,d_,r4_corr_,dlambda_)

    # Step 2
    alpha_ = Newton_step(lambda_,dlambda_,s_,ds_)
    
    # Step 3
    mu_ = np.inner(s_.T, lambda_)/m_
    mu_tilde_ = np.inner((s_ + alpha_*ds_).T, lambda_ + alpha_*dlambda_)/m_
    sigma_ = (mu_tilde_/mu_) ** 3

    # Step 4
    r4_corr_ = np.dot(np.dot(np.diag(ds_),np.diag(dlambda_)),e_) - sigma_*mu_*e_
    rh_c6_ = get_righthand_c6(x_, gamma_, lambda_, s_, G_, A_, C_, g_, b_, d_, r4_corr_)
    
    dx_dgamma_dlambda_= solve_cholesky_system(Lower_, (M_c6_.T).dot(rh_c6_))
    dlambda_ = dx_dgamma_dlambda_[(n_+p_):(n_+p_+m_)]
    ds_ = get_ds_c6(x_,gamma_,lambda_,s_,G_,A_,C_,g_,b_,d_,r4_corr_,dlambda_)
    dz_ = np.r_[dx_dgamma_dlambda_, ds_]
    
    # Step 5
    alpha_ = Newton_step(lambda_,dlambda_,s_,ds_)
    
    # Step 6
    z_ = z_ + 0.95*alpha_*dz_
    x_ = z_[0:n_]
    gamma_ = z_[n_:(n_+p_)]
    lambda_ = z_[(n_+p_):(n_+p_+m_)]
    s_ = z_[(n_+p_+m_):(n_+p_+2*m_)]
    
    return(z_, mu_, -F(x_,gamma_,lambda_,s_, G_, A_, C_, g_, b_, d_))


if __name__ == "__main__":

    # C2
    print("\ntask number: C2\n")
    error = solve_inequality_constraints_case(
            get_new_z, get_M_kkt, n_ = 100, p_ = 0, m_ = 200, 
            iter_max_ = 100, eps_ = 1e-16, print_iter_ = True,
            print_conv_ = True)
    
    print("Error between x and g in last iteration (C2): E = x + g = ", error)
    print("\n")
    
    # C3
    n_max = 100 ### This is the maximum size of the matrix. Use here at least 400 for nice plot
    elapsed_time_c3 = execute_task(n_max_ = n_max, modulo_print_ = 20,
                                   task_ = "C3")
    plot_elapsed_time(elapsed_time_c3, 0.000000095, n_max)
    
    # C4 strategy 1
    print("\n")
    n_max = 40 ### Use here at least 120, better 150 for a nice plot
    elapsed_time_c4_s1 = execute_task(n_max_ = n_max, modulo_print_ = 10,
                                      task_ = "C4 strategy 1")
    plot_elapsed_time(elapsed_time_c4_s1, 0.000004, n_max) 
    
    # C4 strategy 2
    print("\n")
    n_max = 80 ### Use here at least 400 for a nice plot
    elapsed_time_c4_s2 = execute_task(n_max_ = n_max, modulo_print_ = 20,
                                      task_ = "C4 strategy 2")
    plot_elapsed_time(elapsed_time_c4_s2, 0.00000015, n_max)
    
    # C5
    start = timer()
    print("\n\ntask number: C5 optpr1")
    print("first data set\n")
    f1 = solve_general_case(get_new_z, get_M_c6,
                            path_ = "./optpr1/", n_ = 100, p_ = 50, m_ = 200,
                            iter_max_ = 100, eps_ = 1e-16, print_iter_ = True, 
                            print_conv_ = True)
    end = timer()
    print("The algorithm needed ", np.round(end-start, 4),"s to converge")
    
    start = timer()
    print("\ntask number: C5 optpr2")
    print("second data set\n")
    f2 = solve_general_case(get_new_z, get_M_c6,
                            path_ = "./optpr2/", n_ = 1000, p_ = 500, m_ = 2000,
                            iter_max_ = 100, eps_ = 1e-16, print_iter_ = True,
                            print_conv_ = True)
    end = timer()
    print("The algorithm needed ", np.round(end-start, 4),"s to converge")

    # C6 unsuccessful
    #f3 = lib.solve_general_case(lib.get_new_z_c6, lib.get_M_c6,
    #                        path_ = "./optpr1/", n_ = 100, p_ = 50, m_ = 200,
    #                        iter_max_ = 100, eps_ = 1e-16, print_iter_ = True, 
    #                        print_conv_ = True)