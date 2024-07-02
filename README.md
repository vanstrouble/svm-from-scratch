# Support Vector Machine (SVM)

The Support Vector Machine (SVM) is a supervised learning algorithm used primarily for binary classification tasks. SVM aims to find the optimal hyperplane that separates data points of different classes with the maximum margin. This hyperplane ensures that the separation between the classes is as clear and distinct as possible.

![Hyperplane SVM](https://la.mathworks.com/discovery/support-vector-machine/_jcr_content/mainParsys/image.adapt.full.medium.jpg/1718266582410.jpg)

## Algorithm Steps
The basic operation of the SVM involves the following steps:

1. **Initialization:** Input the training data, which consists of labeled instances.
2. **Kernel Selection (Optional):** Choose a kernel function if the data is not linearly separable. Common kernels include linear, polynomial, and radial basis function (RBF).
3. **Hyperplane Construction:** Compute the optimal hyperplane that maximizes the margin between the two classes.
4. **Classification:** Use the hyperplane to classify new data points by determining which side of the hyperplane they fall on.

## Advantages
- **Effective for High-Dimensional Spaces:** SVM is particularly effective in high-dimensional spaces.
- **Memory Efficient:** SVM uses a subset of training points (support vectors) in the decision function, making it memory efficient.
- **Versatile:** Different kernel functions can be specified for the decision function, making SVM versatile and adaptable to various data distributions.

## References
- Cortes, C., & Vapnik, V. (1995). "Support-Vector Networks." Machine Learning, 20(3), 273-297.
- Scholkopf, B., & Smola, A. J. (2001). "Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond."
