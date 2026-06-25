# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler

# import lmdb
import torchvision.transforms as transforms
import six
import sys
from PIL import Image
import numpy as np
import os
import sys
import pickle
import numpy as np
from params import *
import glob, cv2
import torchvision.transforms as transforms
from collections import defaultdict
from io import BytesIO
from pathlib import Path

def get_transform(grayscale=False, convert=True):

    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


class TextDataset:

    def __init__(self, base_path=DATASET_PATHS, num_examples=20, target_transform=None):

        self.NUM_EXAMPLES = num_examples

        # base_path = DATASET_PATHS
        file_to_store = open(base_path, "rb")
        self.IMG_DATA = pickle.load(file_to_store)["train"]
        self.IMG_DATA = dict(list(self.IMG_DATA.items()))  # [:NUM_WRITERS])
        if "None" in self.IMG_DATA.keys():
            del self.IMG_DATA["None"]
        self.author_id = list(self.IMG_DATA.keys())
        self.data = []
        for idx, (author_id, images) in enumerate(self.IMG_DATA.items()):
            for img_data in images:
                self.data.append(
                    {
                        "author_idx": idx,
                        "author_id": author_id,
                        "img": img_data["img"],
                        "label": img_data["label"],
                    }
                )
        self.transform = get_transform(grayscale=True)

        self.target_transform = target_transform

        self.collate_fn = TextCollator()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        NUM_SAMPLES = self.NUM_EXAMPLES
        item_data = self.data[index]
        author_id = item_data["author_id"]
        img = item_data["img"]
        label = item_data["label"]
        author_idx = item_data["author_idx"]

        self.IMG_DATA_AUTHOR = self.IMG_DATA[author_id]
        random_idxs = np.random.choice(
            len(self.IMG_DATA_AUTHOR), NUM_SAMPLES, replace=True
        )

        rand_id_real = np.random.choice(len(self.IMG_DATA_AUTHOR))
        real_img = self.transform(Image.fromarray(np.array(img.convert("L"))))
        real_labels = label.encode()
        imgs = [
            np.array(self.IMG_DATA_AUTHOR[idx]["img"].convert("L"))
            for idx in random_idxs
        ]
        labels = [self.IMG_DATA_AUTHOR[idx]["label"].encode() for idx in random_idxs]

        max_width = 192  # [img.shape[1] for img in imgs]

        imgs_pad = []
        imgs_wids = []

        for img in imgs:

            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros((img_height, max_width), dtype="float32")
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg

            imgs_pad.append(self.transform((Image.fromarray(img))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0)
        

        item = {
            "simg": imgs_pad,
            "swids": imgs_wids,
            "img": real_img,
            "label": real_labels,
            "img_path": "img_path",
            "idx": "indexes",
            "wcl": author_idx,
         
        }

        return item


def _require_hf_datasets():
    try:
        from datasets import Image as HFImage
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Kuzushiji COCO column training requires `datasets`. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return load_dataset, HFImage


def _bookwise_split(book_ids, split, val_ratio, test_ratio, seed):
    books = sorted(set(book_ids))
    rng = random.Random(seed)
    rng.shuffle(books)

    n_books = len(books)
    n_test = int(round(n_books * test_ratio))
    n_val = int(round(n_books * val_ratio))
    if n_books >= 3:
        n_test = max(1, n_test)
        n_val = max(1, n_val)
    if n_val + n_test >= n_books:
        n_val = max(1, min(n_val, n_books - 2))
        n_test = max(1, min(n_test, n_books - n_val - 1))

    test_books = set(books[:n_test])
    val_books = set(books[n_test : n_test + n_val])
    train_books = set(books[n_test + n_val :])

    if split == "train":
        return train_books
    if split in {"val", "valid", "validation"}:
        return val_books
    if split == "test":
        return test_books
    raise ValueError(f"Unknown split: {split}. Use train, val, or test.")


def _char_id_to_index(char_id):
    if isinstance(char_id, int):
        return char_id
    digits = "".join(ch for ch in str(char_id) if ch.isdigit())
    if not digits:
        return None
    return int(digits) - 1


def _safe_get_sequence(mapping, key):
    value = mapping.get(key, [])
    return value if value is not None else []


def _is_supported_label(label):
    return all(char in ALPHABET for char in label)


def _load_pil_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return Image.open(BytesIO(image["bytes"])).convert("RGB")
        if image.get("path") is not None:
            return Image.open(image["path"]).convert("RGB")
    return Image.open(image).convert("RGB")


def _safe_cache_component(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


class KuzushijiColumnDataset(Dataset):
    """Column-level dataset built from Kotomiya07/kuzushiji-dataset-coco."""

    def __init__(
        self,
        split="train",
        dataset_id=None,
        hf_split=None,
        cache_dir=None,
        num_examples=20,
        max_pages=None,
        target_transform=None,
    ):
        self.NUM_EXAMPLES = num_examples
        self.split = split
        self.dataset_id = dataset_id or HF_DATASET_ID
        self.hf_split = hf_split or HF_DATASET_SPLIT
        self.cache_dir = cache_dir or HF_CACHE_DIR
        self.max_width = KUZUSHIJI_MAX_COLUMN_WIDTH
        self.max_label_length = KUZUSHIJI_MAX_LABEL_LENGTH
        self.min_label_length = KUZUSHIJI_MIN_LABEL_LENGTH
        self.max_pages = KUZUSHIJI_MAX_PAGES if max_pages is None else max_pages
        self.data = []
        self.by_author = defaultdict(list)

        if self._load_column_cache():
            self.transform = get_transform(grayscale=True)
            self.target_transform = target_transform
            self.collate_fn = TextCollator()
            return

        load_dataset, hf_image = _require_hf_datasets()
        hf_dataset = load_dataset(
            self.dataset_id,
            split=self.hf_split,
            cache_dir=self.cache_dir,
        )
        hf_dataset = hf_dataset.cast_column("image", hf_image(decode=False))

        all_book_ids = hf_dataset["book_id"]
        selected_books = _bookwise_split(
            all_book_ids,
            split=split,
            val_ratio=KUZUSHIJI_VAL_RATIO,
            test_ratio=KUZUSHIJI_TEST_RATIO,
            seed=KUZUSHIJI_SPLIT_SEED,
        )
        all_books = sorted(set(all_book_ids))
        self.author_to_idx = {book_id: idx for idx, book_id in enumerate(all_books)}
        self.author_id = all_books

        selected_page_count = 0
        max_pages = self.max_pages
        for row in hf_dataset:
            book_id = row["book_id"]
            if book_id not in selected_books:
                continue
            selected_page_count += 1
            for sample in self._row_to_columns(row):
                self.by_author[book_id].append(len(self.data))
                self.data.append(sample)
            if max_pages > 0 and selected_page_count >= max_pages:
                break

        if not self.data:
            raise RuntimeError(
                f"No Kuzushiji column samples were built for split={split!r}."
            )

        if split == "train":
            self._write_lexicon()
        self._save_column_cache()

        self.transform = get_transform(grayscale=True)
        self.target_transform = target_transform
        self.collate_fn = TextCollator()

    def _column_cache_path(self):
        cache_name = "_".join(
            [
                "kuzushiji_columns",
                _safe_cache_component(self.dataset_id),
                _safe_cache_component(self.hf_split),
                _safe_cache_component(self.split),
                f"seed{KUZUSHIJI_SPLIT_SEED}",
                f"val{KUZUSHIJI_VAL_RATIO}",
                f"test{KUZUSHIJI_TEST_RATIO}",
                f"w{self.max_width}",
                f"min{self.min_label_length}",
                f"max{self.max_label_length}",
                f"pages{self.max_pages}",
            ]
        )
        return Path(self.cache_dir) / f"{cache_name}.pkl"

    def _load_column_cache(self):
        cache_path = self._column_cache_path()
        if not cache_path.is_file():
            return False
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        self.data = payload["data"]
        self.author_to_idx = payload["author_to_idx"]
        self.author_id = payload["author_id"]
        self.by_author = defaultdict(list, payload["by_author"])
        if self.split == "train":
            self._write_lexicon()
        return True

    def _save_column_cache(self):
        cache_path = self._column_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "data": self.data,
            "author_to_idx": self.author_to_idx,
            "author_id": self.author_id,
            "by_author": dict(self.by_author),
        }
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)

    def _row_to_columns(self, row):
        image = _load_pil_image(row["image"])

        objects = row.get("objects") or {}
        columns = row.get("columns") or {}
        chars = _safe_get_sequence(objects, "char")
        column_bboxes = _safe_get_sequence(columns, "bbox")
        column_ids = _safe_get_sequence(columns, "column_id")
        column_char_ids = _safe_get_sequence(columns, "char_ids")

        samples = []
        for column_idx, bbox in enumerate(column_bboxes):
            char_ids = column_char_ids[column_idx] if column_idx < len(column_char_ids) else []
            label_chars = []
            for char_id in char_ids:
                char_idx = _char_id_to_index(char_id)
                if char_idx is None or char_idx < 0 or char_idx >= len(chars):
                    continue
                label_chars.append(chars[char_idx])
            label = "".join(label_chars)
            if not (self.min_label_length <= len(label) <= self.max_label_length):
                continue
            if not _is_supported_label(label):
                continue

            x, y, w, h = [int(round(v)) for v in bbox]
            if w <= 0 or h <= 0:
                continue
            crop = image.crop((x, y, x + w, y + h))
            crop = self._normalize_column_image(crop)
            column_id = (
                column_ids[column_idx]
                if column_idx < len(column_ids)
                else f"COL{column_idx + 1:04d}"
            )
            samples.append(
                {
                    "author_idx": self.author_to_idx[row["book_id"]],
                    "author_id": row["book_id"],
                    "img": crop,
                    "label": label,
                    "img_path": f'{row["image_id"]}:{column_id}',
                    "idx": f'{row["image_id"]}:{column_id}',
                }
            )
        return samples

    def _normalize_column_image(self, image):
        image = image.convert("L").transpose(Image.Transpose.ROTATE_90)
        scale = IMG_HEIGHT / float(image.height)
        width = max(1, int(round(image.width * scale)))
        width = min(width, self.max_width)
        return image.resize((width, IMG_HEIGHT), Image.Resampling.BICUBIC)

    def _write_lexicon(self):
        os.makedirs(os.path.dirname(WORDS_PATH), exist_ok=True)
        labels = sorted({item["label"] for item in self.data if item["label"]})
        with open(WORDS_PATH, "w", encoding="utf-8") as f:
            for label in labels:
                f.write(label + "\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item_data = self.data[index]
        author_id = item_data["author_id"]
        author_indices = self.by_author[author_id]
        random_idxs = np.random.choice(
            author_indices, self.NUM_EXAMPLES, replace=True
        )

        real_img = self.transform(item_data["img"])
        imgs = [
            np.array(self.data[idx]["img"].convert("L"))
            for idx in random_idxs
        ]

        imgs_pad = []
        imgs_wids = []
        for img in imgs:
            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            out_img = np.zeros((img_height, self.max_width), dtype="float32")
            clipped_width = min(img_width, self.max_width)
            out_img[:, :clipped_width] = img[:, :clipped_width]
            img = 255 - out_img
            imgs_pad.append(self.transform(Image.fromarray(img.astype(np.uint8))))
            imgs_wids.append(clipped_width)

        item = {
            "simg": torch.cat(imgs_pad, 0),
            "swids": imgs_wids,
            "img": real_img,
            "label": item_data["label"].encode("utf-8"),
            "img_path": item_data["img_path"],
            "idx": item_data["idx"],
            "wcl": item_data["author_idx"],
        }
        return item


class TextDatasetval:

    def __init__(self, base_path=DATASET_PATHS, num_examples=20, target_transform=None):

        self.NUM_EXAMPLES = num_examples
        # base_path = DATASET_PATHS
        file_to_store = open(base_path, "rb")
        self.IMG_DATA = pickle.load(file_to_store)["test"]
        self.IMG_DATA = dict(list(self.IMG_DATA.items()))  # [NUM_WRITERS:])
        if "None" in self.IMG_DATA.keys():
            del self.IMG_DATA["None"]
        self.author_id = list(self.IMG_DATA.keys())
        self.data = []
        for idx, (author_id, images) in enumerate(self.IMG_DATA.items()):
            for img_data in images:
                self.data.append(
                    {
                        "author_idx": idx,
                        "author_id": author_id,
                        "img": img_data["img"],
                        "label": img_data["label"],
                    }
                )
        self.transform = get_transform(grayscale=True)
        self.target_transform = target_transform

        self.collate_fn = TextCollator()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        NUM_SAMPLES = self.NUM_EXAMPLES
        item_data = self.data[index]
        author_id = item_data["author_id"]
        img = item_data["img"]
        label = item_data["label"]
        author_idx = item_data["author_idx"]

        self.IMG_DATA_AUTHOR = self.IMG_DATA[author_id]
        random_idxs = np.random.choice(
            len(self.IMG_DATA_AUTHOR), NUM_SAMPLES, replace=True
        )

        rand_id_real = np.random.choice(len(self.IMG_DATA_AUTHOR))
        real_img = self.transform(Image.fromarray(np.array(img.convert("L"))))
        real_labels = label.encode()
        imgs = [
            np.array(self.IMG_DATA_AUTHOR[idx]["img"].convert("L"))
            for idx in random_idxs
        ]
        labels = [self.IMG_DATA_AUTHOR[idx]["label"].encode() for idx in random_idxs]

        max_width = 192  # [img.shape[1] for img in imgs]

        imgs_pad = []
        imgs_wids = []

        for img in imgs:

            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros((img_height, max_width), dtype="float32")
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg

            imgs_pad.append(self.transform((Image.fromarray(img))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0)
        
        item = {
            "simg": imgs_pad,
            "swids": imgs_wids,
            "img": real_img,
            "label": real_labels,
            "img_path": "img_path",
            "idx": "indexes",
            "wcl": author_idx,
       
        }

        return item


class TextCollator(object):
    def __init__(self):
        self.resolution = resolution

    def __call__(self, batch):

        img_path = [item["img_path"] for item in batch]
        width = [item["img"].shape[2] for item in batch]
        indexes = [item["idx"] for item in batch]
        simgs = torch.stack([item["simg"] for item in batch], 0)
     
        wcls = torch.Tensor([item["wcl"] for item in batch])
        swids = torch.Tensor([item["swids"] for item in batch])
        imgs = torch.ones(
            [
                len(batch),
                batch[0]["img"].shape[0],
                batch[0]["img"].shape[1],
                max(width),
            ],
            dtype=torch.float32,
        )
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0 : item["img"].shape[2]] = item["img"]
            except:
                print(imgs.shape)
        item = {
            "img": imgs,
            "img_path": img_path,
            "idx": indexes,
            "simg": simgs,
            "swids": swids,
            "wcl": wcls,
       
        }
        if "label" in batch[0].keys():
            labels = [item["label"] for item in batch]
            item["label"] = labels
        if "z" in batch[0].keys():
            z = torch.stack([item["z"] for item in batch])
            item["z"] = z
        return item
