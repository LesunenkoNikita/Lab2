import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_eigen(square_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(square_matrix)
    print("Eigenvalues:")
    for i in range(len(eigenvalues)):
        print(f"λ{i + 1} = {eigenvalues[i]}")

    print("Eigenvectors")
    for i in range(len(eigenvectors)):
        print(f"v{i + 1} = {eigenvectors[:, i]}")

    for i in range(len(eigenvalues)):
        if np.allclose(np.dot(square_matrix, eigenvectors[i]), eigenvalues[i] * eigenvectors[i]):
            print(f"Рівність A*v{i + 1} = λ{i + 1}*v{i + 1} істинна")
        else:
            print(f"Рівність A*v{i + 1} = λ{i + 1}*v{i + 1} хибна")

image_raw = imread("C:\\Users\\User\\Downloads\\image.jpg")
print(image_raw.shape)
image_sum = image_raw.sum(axis=2)
print(image_sum.shape)
image_bw = image_sum / image_sum.max()
print(image_bw.max())
plt.imshow(image_bw, cmap='gray')
plt.show()

pca = PCA()
pca.fit(image_bw)
cumulative_var = np.cumsum(pca.explained_variance_ratio_)
amount_of_components_95p = np.argmax(cumulative_var >= 0.95) + 1
print(f"{amount_of_components_95p} потрібно для покриття 95% variance")

plt.plot(cumulative_var)
plt.xlabel('Кількість компонентів')
plt.ylabel('Кумулятивна дисперсія')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.axvline(x=amount_of_components_95p, color='k', linestyle='--')
plt.show()

pca2 = PCA(amount_of_components_95p)
image_bw_rework = pca.inverse_transform(pca.fit_transform(image_bw))

plt.imshow(image_bw_rework, cmap='gray')
plt.show()

components_amount = 20
pca = PCA()
while components_amount != 200:
    image_bw_rework = pca.inverse_transform(pca.fit_transform(image_bw))

    plt.imshow(image_bw_rework, cmap='gray')
    plt.title(f'Реконструкція з {components_amount} компонентами')
    plt.show()