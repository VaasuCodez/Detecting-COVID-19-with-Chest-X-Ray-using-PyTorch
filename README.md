# Detecting COVID-19 with Chest X-Ray using PyTorch

Objective of this project is to create an image classification model that can predict Chest X-Ray scans that belong to one of the three classes: Normal, Viral Pneumonia, COVID-19 with a reasonably high accuracy using PyTorch.

![image](https://user-images.githubusercontent.com/56118766/144033377-255abf1c-d0bc-471d-9ff7-acc6d537c03f.png)

# Table of contents

- [Dataset](#dataset)
- [Development](#development)
- [References](#references)
- [Contribute](#contribute)
- [License](#license)
- [Footer](#footer)

# Dataset

Dataset used: [COVID-19 Radiography Dataset](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) <br>
This dataset has more than 3500 Chest X-Ray scans which are categorized in four classes - Normal, Viral Pneumonia, COVID and Lung Opacity (not used here).

# Development

THis project uses the ResNet-18 model from PyTorch trained on the COVID-19 Radiography dataset. To take advantage of tranfer learning and use the pre-trained weights from training on the ImageNet dataset, the dataset needed to be normalized in the same way as the ImageNet dataset.

## Data Visualization

```python
images, labels = next(iter(dl_train))
show_images(images, labels, labels)
```

![image](https://user-images.githubusercontent.com/56118766/144042819-de75ebfc-ee94-4773-aee4-79f76c6b92c9.png)


```python
images, labels = next(iter(dl_test))
show_images(images, labels, labels)
```

![image](https://user-images.githubusercontent.com/56118766/144042864-4f6cabb4-f813-47d9-95ff-031a0ac3176d.png)


## Training

![image](https://user-images.githubusercontent.com/56118766/144036761-28412057-9313-4874-b208-013fb577f7cb.png)

## Final Result

```python
show_preds()
```
![image](https://user-images.githubusercontent.com/56118766/144042511-061a7f2d-535f-437f-bacd-8d56475baebe.png)


# References

1.  M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676. [Paper link](https://ieeexplore.ieee.org/document/9144185)

2.  Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. [Paper link](https://doi.org/10.1016/j.compbiomed.2021.104319)

3.  Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In: Advances in Neural Information Processing Systems 32 [Internet]. Curran Associates, Inc.; 2019. p. 8024–35. Available from: [Paper link](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf)

# Contribute

Contributions are welcomed. For major changes, kindly open an issue first to discuss the same before creating a pull request.

# License

[MIT](https://opensource.org/licenses/MIT)

# Footer

Please note that this dataset, and the model trained in the project, can not be used to diagnose COVID-19 or Viral Pneumonia. This data is only for educational purpose.

Leave a star in the GitHub repository if you found this project helpful!

[(Back to top)](#table-of-contents)
