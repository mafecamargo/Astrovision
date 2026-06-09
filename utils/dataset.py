from pathlib import Path

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class DisasterDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def build_dataset(root_dir):

    root_dir = Path(root_dir)

    class_names = sorted(
        [d.name for d in root_dir.iterdir() if d.is_dir()]
    )

    class_to_idx = {
        name: idx
        for idx, name in enumerate(class_names)
    }

    samples = []

    for class_name in class_names:

        class_dir = root_dir / class_name

        for img_path in class_dir.rglob("*"):

            if img_path.suffix.lower() in IMG_EXTENSIONS:
                samples.append(
                    (str(img_path), class_to_idx[class_name])
                )

    return samples, class_names


def get_dataloaders(
    root_dir,
    batch_size=32,
    img_size=224,
    val_size=0.15,
    test_size=0.15,
    num_workers=4,
):

    samples, class_names = build_dataset(root_dir)

    labels = [label for _, label in samples]

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(val_size + test_size),
        stratify=labels,
        random_state=42
    )

    temp_labels = [label for _, label in temp_samples]

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=test_size / (val_size + test_size),
        stratify=temp_labels,
        random_state=42
    )

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = DisasterDataset(
        train_samples,
        train_transform
    )

    val_dataset = DisasterDataset(
        val_samples,
        eval_transform
    )

    test_dataset = DisasterDataset(
        test_samples,
        eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        class_names
    )
