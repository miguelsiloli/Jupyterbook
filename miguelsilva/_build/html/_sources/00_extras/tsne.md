# t-SNE as a Nonlinear Visualization Technique

We studied (kernel) PCA as an example for a method that reduces the dimensionality of a dataset and makes features apparent by which data points can be efficiently distinguished. Often, it is desirable to more clearly cluster similar data points and visualize this clustering in a low (two- or three-) dimensional space. We focus our attention on a relatively recent algorithm (from 2008) that has proven very performant. It goes by the name t-distributed stochastic neighborhood embedding (t-SNE).

The basic idea is to think of the data (images, for instance) as objects xi
in a very high-dimensional space and characterize their relation by the Euclidean distance ||xi−xj|| between them. These pairwise distances are mapped to a probability distribution pij. The same is done for the distances ||yi−yj|| of the images of the data points yi in the target low-dimensional space. Their probability distribution is denoted qij. The mapping is optimized by changing the locations yi

so as to minimize the distance between the two probability distributions. Let us substantiate these words with formulas.

The probability distribution in the space of data points is given as the symmetrized version (joint probability distribution)

$p_{ij}=\frac{p_{i|j}+p_{j|i}}{2}$

of the conditional probabilities

$p_{j|i}=\frac{\mathrm{exp}\left(-||\mathbf{x}_i-\mathbf{x}_j||^2/2\sigma_i^2\right)}
{\sum_{k\neq i}\mathrm{exp}\left(-||\mathbf{x}_i-\mathbf{x}_k||^2/2\sigma_i^2\right)}$

where the choice of variances $σ_i$ will be explained momentarily. Distances are thus turned into a Gaussian distribution. Note that pj|i≠pi|j while pji=pij.

The probability distribution in the target space is chosen to be a Student t-distribution

$q_{ij}=\frac{
(1+||\mathbf{y}_i-\mathbf{y}_j||^2)^{-1}
}{
\sum_{k\neq l}
(1+||\mathbf{y}_k-\mathbf{y}_l||^2)^{-1}
}$

![Application of both methods on 5000 samples from the MNIST handwritten digit dataset.](assets/pca_tSNE.png)

### Perplexity

Let us now discuss the choice of $σ_i$. Intuitively, in dense regions of the dataset, a smaller value of $σ_i$ is usually more appropriate than in sparser regions, in order to resolve the distances better. Any particular value of $σ_i$ induces a probability distribution Pi over all the other data points. This distribution has an entropy (here we use the Shannon entropy, in general it is a measure for the “uncertainty” represented by the distribution)

$H(P_i)=-\sum_j p_{j|i}\, \mathrm{log}_2 \,p_{j|i}.$

The value of $H(Pi)$ increases as $σ_i$ increases, i.e., the more uncertainty is added to the distances. The algorithm searches for the $σ_i$ that result in a $P_i$ with fixed perplexity.

$\mathrm{Perp}(P_i)=2^{H(P_i)}.$

The target value of the perplexity is chosen a priory and is the main parameter that controls the outcome of the t-SNE algorithm. It can be interpreted as a smooth measure for the effective number of neighbors. Typical values for the perplexity are between 5 and 50.

### Optimization function

Finally, we have to introduce a measure for the similarity between the two probability distributions $pij$
and $qij$. This defines a so-called loss function. Here, we choose the **Kullback-Leibler divergence**.

$L(\{\mathbf{y}_i\})=\sum_i\sum_jp_{ij}\mathrm{log}\frac{p_{ij}}{q_{ij}}$

The minimization of $L({yi})$ with respect to the positions yi can be achieved with a variety of methods. In the simplest case it can be gradient descent, which we will discuss in more detail in a later chapter. As the name suggests, it follows the direction of largest gradient of the cost function to find the minimum. To this end it is useful that these gradients can be calculated in a simple form

$\frac{\partial L}{\partial \mathbf{y}_i}
=4\sum_j (p_{ij}-q_{ij})(\mathbf{y}_i-\mathbf{y}_j)(1+||\mathbf{y}_i-\mathbf{y}_j||^2)^{-1}.$

## Final Remarks

While t-SNE is a very powerful clustering technique, it has its limitations. 
- (i) The target dimension should be 2 or 3, for much larger dimensions ansatz for $q_{ij}$ is not suitable. 
- (ii) If the dataset is intrinsically high-dimensional (so that also the PCA pre-processing fails), t-SNE may not be a suitable technique. 
- (iii) Due to the stochastic nature of the optimization, results are not reproducible. The result may end up looking very different when the algorithm is initialized with some slightly different initial values for $y_i$.

# Principal Compoent Analysis as a linear transformation technique

Very often, we are presented with a dataset containing many types of information, called features of the data. Such a dataset is also described as being high-dimensional. Techniques that extract information from such a dataset are broadly summarised as high-dimensional inference.

PCA is a systematic way to find out which feature or combination of features varies the most across the data samples. We can think of PCA as approximating the data with a high-dimensional ellipsoid, where the principal axes of this ellipsoid correspond to the principal components. A feature, which is almost constant across the samples, in other words has a very short principal axis, might not be very useful. PCA then has two main applications: 
- (1) It helps to visualise the data in a low dimensional space and 
- (2) it can reduce the dimensionality of the input data to an amount that a more complex algorithm can handle.

### PCA algorithm

The procedure to perform PCA can then be described as follows:

```{admonition} Principle Component Analysis

1.  Center the data by subtracting from each column the mean of that
    column,

    ```{math}
	{x}_i \mapsto {x}_{i} - \frac{1}{m} \sum_{i=1}^{m} {x}_{i}.
          %  x_{ij} \longrightarrow x_{ij} - \frac{1}{m} \sum_{i=1}^{m} x_{ij}.
	```
    This ensures that the mean of each data feature is zero.

2.  Form the $n$ by $n$ (unnormalised) covariance matrix
    ```{math}
	:label: eqn:PCA-Covariance-Matrix
	C = {X}^{T}{X} = \sum_{i=1}^{m} {x}_{i}{x}_{i}^{T}.
    ```

3.  Diagonalize the matrix to the form
    $C = {X}^{T}{X} = W\Lambda W^{T}$, where the columns of $W$ are the
    normalised eigenvectors, or principal components, and $\Lambda$ is a
    diagonal matrix containing the eigenvalues. It will be helpful to
    arrange the eigenvalues from largest to smallest.

4.  Pick the $l$ largest eigenvalues $\lambda_1, \dots \lambda_l$,
    $l\leq n$ and their corresponding eigenvectors
    ${v}_1 \dots {v}_l$. Construct the $n$ by $l$ matrix
    $\widetilde{W} = [{v}_1 \dots {v}_l]$.

5.  Dimensional reduction: Transform the data matrix as

    ```{math}
	:label: eqn:PCA-Dimensional-Reduction
            \widetilde{X} = X\widetilde{W}.
    ``` 
The transformed data
    matrix $\widetilde{X}$ now has dimensions $m$ by $l$.
```

PCA algorithm amounts simply to a rotation of the original data. However, it still produces 2 new features which are orthogonal linear combinations of the original features.

### Kernel PCA

The basic idea of this method is to apply to the data x∈Rn a chosen non-linear vector-valued transformation function $Φ(x)$ with

$\mathbf{\Phi}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{N},$

which is a map from the original n-dimensional space (corresponding to the n original data features) to a N-dimensional feature space. Kernel PCA then simply involves performing the standard PCA on the transformed data Φ(x). Here, we will assume that the transformed data is centered, i.e.,

$\sum_i \Phi(\mathbf{x}_i) = 0$

In practice, when N is large, it is not efficient or even possible to explicitly perform the transformation Φ. Instead we can make use of a method known as the kernel trick. Recall that in standard PCA, the primary aim is to find the eigenvectors and eigenvalues of the covariance matrix C . In the case of kernel PCA, this matrix becomes

$C = \sum_{i=1}^{m} \mathbf{\Phi}(\mathbf{x}_{i})\mathbf{\Phi}(\mathbf{x}_{i})^T,$

with the eigenvalue equation

$\sum_{i=1}^{m} \mathbf{\Phi}(\mathbf{x}_{i})\mathbf{\Phi}(\mathbf{x}_{i})^T \mathbf{v}_{j} = \lambda_{j}\mathbf{v}_{j}.$

