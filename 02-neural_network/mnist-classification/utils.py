from torchvision import datasets, transforms

def load_mnist(is_train=True, flatten=True):

    dataset = datasets.MNIST(
        root='../data',
        train = is_train,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    x = dataset.data.float() / 255
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y