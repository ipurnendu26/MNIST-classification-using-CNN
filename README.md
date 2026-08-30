# MNIST Digit Classification with a CNN

A Jupyter notebook that builds and evaluates a convolutional neural network for handwritten-digit classification on MNIST.

## Project file

- `MNIST Deep learning.ipynb`: preprocessing, CNN architecture, training, and evaluation

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## Skills demonstrated

- Image tensor normalization and shape handling
- Convolution, pooling, and dense classification layers
- Training/validation curves
- Multiclass prediction and error analysis

## Limitations

MNIST is a small, highly standardized benchmark. Good notebook performance does not imply robustness to handwriting styles, image quality, adversarial inputs, or other domains. A stronger experiment would fix seeds, separate validation from test data, inspect misclassifications, save versioned artifacts, and report per-class metrics.

## License

[MIT](LICENSE)
