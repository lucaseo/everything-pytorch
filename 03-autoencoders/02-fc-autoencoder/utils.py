from torchvision import datasets, transforms

def load_mnist(is_train=True, flatten=True):

    datasets = datasets.MNIST(
        "../data",
        train = is_train,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor()
        ]),
    )

    x = datasets.data.float() / 255.
    y = datasets.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y