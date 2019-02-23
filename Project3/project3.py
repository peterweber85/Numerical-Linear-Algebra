#%% LIBRARIES
import numpy as np
import pandas  as pd
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt 

print("Libraries loaded")

# Parameters
PRODUCE_PLOTS = False # Dont set this to True when executing the whole .py file, it gets stuck in the plots
IMAGE_FOLDER = ""
PROPORTIONS = [0.001, 0.01, 0.1, 1]

#%% Load images
image_bm = ndimage.imread("BlackMirrorS4.jpg")
image_hm = ndimage.imread("HolsteeManifesto.jpg")

print("Images loaded")

#%% Helpers
def get_svd_decomposition(image_):
    output = []
    for color_idx in range(image_.shape[2]):
        svd_ = dict()
        svd_['U'], svd_['s'], svd_['VT'] = np.linalg.svd(image_[:,:,color_idx], full_matrices=False)
        output.append(svd_)
    return(output)
def get_frobenius_norm(svd_, image_):
    norm_ = 0.
    color_channels = image_.shape[2]
    for color_idx in range(image_.shape[2]):
        s_ = svd_[color_idx]['s']
        norm_ = norm_ + np.sqrt(np.sum(s_ ** 2))
    return(norm_/color_channels)
def compress_image(svd_, proportion_, total_norm_):
    num_svalues = len(svd_[0]['s'])
    num_comps = int(np.ceil(proportion_*num_svalues))

    reduced_image = []
    frobenius_norm = 0.
    for color_channel in svd_:
        U, s, VT = color_channel['U'], color_channel['s'], color_channel['VT']
        partial_image = U[:, :num_comps].dot(np.diag(s[:num_comps])).dot(VT[:num_comps, :])
        partial_image = partial_image.astype('uint8')
        reduced_image.append(partial_image)
        frobenius_norm = frobenius_norm + np.sqrt(np.sum(s[:num_comps] ** 2))
  
    all_colors = np.dstack(reduced_image)
    frobenius_norm = frobenius_norm/all_colors.shape[2]
    percentage = frobenius_norm/total_norm_*100
    return(all_colors, frobenius_norm, '{0:.3f}'.format(percentage), num_comps)
def save_compressed_images(image_, proportions, image_folder, image_name):
    svd_ = get_svd_decomposition(image_)
    total_norm = get_frobenius_norm(svd_, image_)
    m, n = image_.shape[:-1]
    
    compressed_images = []
    frobenius_norms = []
    percentages = []
    numbers_comps = []
    
    for proportion in proportions:
        compressed_image, norm_, percentage, num_comps = compress_image(svd_, proportion,total_norm)
        compressed_images.append(compressed_image)
        frobenius_norms.append(norm_)
        percentages.append(percentage)
        numbers_comps.append(num_comps)
        if compressed_image.shape[2] < 3:
            misc.imsave(image_folder + image_name + str(num_comps) + ".jpg", compressed_image[:,:,0])
        else:
            misc.imsave(image_folder + image_name + str(num_comps) + ".jpg", compressed_image)
    print("\n------------------------------------------------------------")
    print("\n-------------------- ", image_name, "--------------------")
    print("\nNumber of components:", numbers_comps)
    print("\nRelative number of components:", proportions)
    print("\nAveraged Frobenius Norms:", np.round(frobenius_norms))
    print("\nRelative Frobenius Norms in (%):", percentages)
def get_image_information(image_, num_, image_name):
    svd_ = get_svd_decomposition(image_)
    total_norm = get_frobenius_norm(svd_, image_)
    m, n = image_.shape[:-1]

    compressed_images = []
    frobenius_norms = []
    percentages = []
    numbers_comps = []
    proportions = []
    
    if(type(num_) != list):
        seq = np.logspace(-5, 0, num = num_)
    else:
        seq = num_

    for proportion in seq:
        compressed_image, norm_, percentage, num_comps = compress_image(svd_, proportion,total_norm)
        proportions.append(proportion)
        frobenius_norms.append(norm_)
        percentages.append(percentage)
        numbers_comps.append(num_comps)

    return(numbers_comps, frobenius_norms, percentages, proportions)
def plot_frobenius_vs_components(components, percentages, title, filename):
    fig = plt.figure(figsize=(5,4))
    plt.plot(components, percentages)
    plt.xlabel("Rank of approximation")
    plt.ylabel("Frobenius norm captured (%)")
    plt.xscale('log')
    plt.title(title)
    plt.savefig(IMAGE_FOLDER + filename)
    plt.show()
    plt.close()
def get_cov_and_corr_matrix(filename):
    X = np.loadtxt(filename)
    print("\nShape of toy example data set:", X.shape)
    print("\nTranspose data set to have features in rows and samples in columns")
    X = X.T
    X_center = X-X.mean(axis=1)[:,np.newaxis]
    print("\nThe mean zero dataset:\n", X_center)
    Y_cov = X_center.T/np.sqrt(X.shape[1]-1)
    print("\nAnd its covariance matrix:\n", Y_cov)

    X_norm = X_center/np.sqrt(X_center.var(axis=1))[:,np.newaxis]
    print("\nThe dataset with unit standard deviation:\n", X_norm)
    Y_corr = X_norm.T/np.sqrt(X.shape[1]-1)
    print("\nAnd its correlation matrix_\n", Y_corr)
    return(X_center, Y_cov, X_norm, Y_corr)
def plot_svalues_and_cumul_var(svalues, cumul_var, title1, title2, filename):
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.plot(np.linspace(1,len(svalues),len(svalues)), \
        svalues, 'ko', np.linspace(1,len(svalues),len(svalues)), \
        svalues, 'k--')
    plt.ylim([0, 1.1*np.max(svalues)])
    plt.xlabel("Principal component")
    plt.ylabel("Standard deviation")
    plt.title(title1)
    plt.subplot(122)
    plt.plot(np.linspace(1,len(svalues),len(svalues)), \
        cumul_var, 'ko', np.linspace(1,len(svalues),len(svalues)), \
        cumul_var, 'k--')
    plt.ylim([0, 1.1*100])
    plt.xlabel("Principal component")
    plt.ylabel("Cumulative variance (%)")
    plt.title(title2)
    plt.tight_layout()
    plt.savefig(IMAGE_FOLDER + filename)
    plt.show()
    plt.close()
def get_new_basis_and_variance_explained(X, Y, print_new_basis = False):
    U, s, VT = np.linalg.svd(Y, full_matrices=False)
    X_new_ = VT.dot(X)
    if(print_new_basis):
        print("\nThe new basis of the dataset is:\n", X_new_)
    print("\nThe dimension of the new dataset is:", X_new_.shape)
    print("\nThe standard deviation of the principal components (PC1, PC2,...) is:\n", s)
    variance_explained = 100. * s ** 2/np.sum(s ** 2)
    cumulative_variance = [np.sum(variance_explained[:i]) for i in range(1, len(variance_explained)+1)]
    print("\nThe cumulative variance for each of the principal components is:\n",cumulative_variance)
    return(X_new_, s, variance_explained, cumulative_variance)
def get_cov_matrix_genes(filename):
    genes = pd.read_csv(filename)
    genes = genes.drop('gene', 1)
    values = genes.values
    column_names = genes.columns
    X_center = values-values.mean(axis=1)[:,np.newaxis]
    Y_cov = X_center.T/np.sqrt(values.shape[1]-1)
    return(genes, column_names, X_center, Y_cov)
def plot_first_two_prcomp(X, var):
    plt.figure(figsize=(8, 5))
    plt.plot(X[0,:], X[1,:], "+")
    plt.xlabel("PC1: {:.2f} % of variance ".format(var[0]))
    plt.ylabel("PC2: {:.2f} % of variance".format(var[1]))
    plt.title("First two principal components of genes data")
    plt.savefig(IMAGE_FOLDER + "first_two_prcomps.pdf")
    plt.tight_layout()
    plt.show()
    plt.close()
def generate_and_save_output_file(X_new_, samples_, var_, filename):
    columns = ['col']
    for i in range(1,21):
        column = 'PC{}'.format(i)
        columns.append(column)
    columns.append('Variance')
    X_new = pd.DataFrame(np.c_[samples_, X_new_.T, var])
    X_new.columns = columns
    X_new.to_csv(filename, index = False)
    print("\nThe requested output file is saved as", filename)
    return(X_new)

print("Functions loaded")

#%%
save_compressed_images(image_bm, PROPORTIONS, IMAGE_FOLDER, "BlackMirror")
save_compressed_images(image_hm, PROPORTIONS, IMAGE_FOLDER, "HolsteeManifesto")

print("Images saved")

#%%
number_components, norms, percentages, proportions  = \
get_image_information(image_bm, 50, "BlackMirror.jpg")
if(PRODUCE_PLOTS):
    plot_frobenius_vs_components(number_components, percentages, "BlackMirror", "Frobenius_bm.pdf")

#%%
number_components, norms, percentages, proportions  = \
get_image_information(image_hm, 50, "HolsteeManifesto.jpg")
if(PRODUCE_PLOTS):
    plot_frobenius_vs_components(number_components, percentages, "HolsteeManifesto", "Frobenius_hm.pdf")

#%% PCA example
print("\n------------- Principal Componentes Analysis -------------\n")
print("\n-------------------------- Task 1 --------------------------\n")
X_cov, Y_cov, X_corr, Y_corr = get_cov_and_corr_matrix('example.dat')

#%% Covariance
print("\n-------------------------- Covariance --------------------------\n")
X_new, s, var, cumul_var = get_new_basis_and_variance_explained(X_cov, Y_cov, True)
if(PRODUCE_PLOTS):
    plot_svalues_and_cumul_var(s, cumul_var,"Singular values", "Variance explained", "Example_covariance.pdf")

#%% Correlation
print("\n-------------------------- Correlation --------------------------\n")
X_new, s, var, cumul_var = get_new_basis_and_variance_explained(X_corr, Y_corr, True)
if(PRODUCE_PLOTS):
    plot_svalues_and_cumul_var(s, cumul_var,"Singular values", "Variance explained", "Example_correlation.pdf")

#%% PCA genes
print("\n-------------------------- Task 2 --------------------------\n")
fname = "RCsGoff.csv"
genes, samples, X_genes, Y_genes = get_cov_matrix_genes(fname)
X_new, s, var, cumul_var = get_new_basis_and_variance_explained(X_genes, Y_genes)
if(PRODUCE_PLOTS):
    plot_svalues_and_cumul_var(s, cumul_var, "Singular values", "Variance explained", "Genes_covariance.pdf")
    plot_first_two_prcomp(X_new, var)

#%%
print(generate_and_save_output_file(X_new, samples, var, "RCsGoff_output.csv"))

