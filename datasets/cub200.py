import os

from PIL import Image
from torch.utils.data import Dataset


class CUB200Dataset(Dataset):
    """CUB-200-2011 classification dataset using the official train/test split.

    Expected directory layout:
        root/
          CUB_200_2011/
            images/
            images.txt
            image_class_labels.txt
            train_test_split.txt
            classes.txt

    The official metadata files use a shared image id. This loader joins those
    files by image id and returns standard PyTorch samples: (image, label).
    """

    base_folder = "CUB_200_2011"

    def __init__(self, root="./data", train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data_dir = os.path.join(root, self.base_folder)
        self.image_dir = os.path.join(self.data_dir, "images")

        self._check_exists()
        self.classes = self._load_classes()
        self.samples = self._load_samples()

    def _check_exists(self):
        required_paths = [
            self.image_dir,
            os.path.join(self.data_dir, "images.txt"),
            os.path.join(self.data_dir, "image_class_labels.txt"),
            os.path.join(self.data_dir, "train_test_split.txt"),
            os.path.join(self.data_dir, "classes.txt"),
        ]
        missing_paths = [path for path in required_paths if not os.path.exists(path)]
        if missing_paths:
            missing = "\n".join(missing_paths)
            raise FileNotFoundError(
                "CUB-200-2011 files were not found. Expected the dataset under "
                f"{self.data_dir}.\nMissing paths:\n{missing}"
            )

    def _load_classes(self):
        # classes.txt maps class ids to names, e.g.
        #   1 001.Black_footed_Albatross
        # It is not needed for training, but keeping it mirrors torchvision
        # datasets and makes sanity checks/debugging easier.
        classes_path = os.path.join(self.data_dir, "classes.txt")
        classes = []
        with open(classes_path, "r") as file:
            for line in file:
                _, class_name = line.strip().split(maxsplit=1)
                classes.append(class_name)
        return classes

    def _read_mapping(self, filename):
        # CUB metadata files share the same "image_id value" text format:
        #   images.txt:             image_id relative_image_path
        #   image_class_labels.txt: image_id class_id
        #   train_test_split.txt:   image_id split_flag
        # Returning a dict lets _load_samples join them by image id.
        mapping = {}
        path = os.path.join(self.data_dir, filename)
        with open(path, "r") as file:
            for line in file:
                image_id, value = line.strip().split(maxsplit=1)
                mapping[int(image_id)] = value
        return mapping

    def _load_samples(self):
        image_paths = self._read_mapping("images.txt")
        labels = self._read_mapping("image_class_labels.txt")
        split_flags = self._read_mapping("train_test_split.txt")
        # Official split flags: 1 = train, 0 = test.
        target_split = "1" if self.train else "0"

        samples = []
        for image_id, relative_path in sorted(image_paths.items()):
            if split_flags[image_id] != target_split:
                continue

            image_path = os.path.join(self.image_dir, relative_path)
            # Official CUB labels are 1-indexed (1..200), while PyTorch
            # classification losses expect class indices in the range 0..199.
            label = int(labels[image_id]) - 1
            samples.append((image_path, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
