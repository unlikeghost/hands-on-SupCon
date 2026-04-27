import torch
import numpy as np
import torchvision.transforms as transforms


class ContrastiveLearningDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        transform: transforms.Compose
    ):
        self.dataset = dataset
        if hasattr(dataset, 'targets'):
            self.labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            self.labels = np.array(dataset.labels)
        elif isinstance(dataset, torch.utils.data.Subset):
            self.labels = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            raise ValueError("Dataset must have targets, labels or be a Subset of a dataset with targets")

        self.class_indices = [
            np.where(self.labels == i)[0].tolist()
            for i in range(len(np.unique(self.labels)))
        ]

        self.transform = transform

    def __getitem__(self, index):
        img1, label1 = self.dataset[index]

        is_same = torch.randint(0, 2, (1,)).item()

        if is_same:
            indices = self.class_indices[label1]
            idx2 = indices[torch.randint(0, len(indices), (1,)).item()]
            target = torch.tensor([0.0], dtype=torch.float32)
        else:
            label2 = torch.randint(0, len(self.class_indices), (1,)).item()
            while label2 == label1:
                label2 = torch.randint(0, len(self.class_indices), (1,)).item()

            indices = self.class_indices[label2]
            idx2 = indices[torch.randint(0, len(indices), (1,)).item()]
            target = torch.tensor([1.0], dtype=torch.float32)

        img2, _ = self.dataset[idx2]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target

    def __len__(self):
        return len(self.dataset)


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        transform: transforms.Compose
    ):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img, label