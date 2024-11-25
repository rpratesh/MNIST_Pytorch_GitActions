# MNIST Digit Classification with PyTorch

[![Model Tests](https://github.com/rpratesh/MNIST_Pytorch_GitActions/actions/workflows/model_test.yml/badge.svg?branch=master)](https://github.com/rpratesh/MNIST_Pytorch_GitActions/actions/workflows/model_test.yml)

This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset using PyTorch. The implementation includes automated testing using GitHub Actions to ensure model performance and parameter constraints.

## Project Structure

```
├── model.py           # CNN model architecture
├── train_test.py     # Training and testing functions
├── .github/
│   └── workflows/
│       └── model_test.yml  # GitHub Actions
└── README.md
```
## Model Architecture

The CNN architecture consists of:
- 3 convolutional layers with increasing filter sizes (8, 16, 32)
- Max pooling layers
- Dropout for regularization
- A fully connected layer for final classification
- Total parameters = 13,898

## Model Performance

The model achieves the following targets (on MNIST dataset):
- Training accuracy: > 90% after 1 epoch
- Test accuracy: > 95%
- Parameter count: < 25,000 parameters

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- pytest

Install dependencies:
```
bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pytest numpy
```

### Automated Testing

The project includes automated tests that run on GitHub Actions. The tests verify:
1. Model parameter count (< 25,000 parameters)
2. Training accuracy (> 90% after 1 epoch)
3. Test accuracy (> 95%)

To run tests locally:
```
bash
pytest test_model.py -v
```

## GitHub Actions Workflow

The project uses GitHub Actions to automatically run tests on push and pull requests to main/master branches. The workflow:
1. Sets up Python environment
2. Installs dependencies
3. Creates and runs test file
4. Verifies model performance and constraints

## License

This project is open source and available under the MIT License.

## Acknowledgments

- MNIST Dataset: [LeCun et al.](http://yann.lecun.com/exdb/mnist/)
- PyTorch Documentation and Tutorials



