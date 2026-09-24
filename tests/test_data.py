import numpy as np
import pytest
import torch

from leafnet.data import LeafDataset, build_transforms, repeated_splits, scan_image_folder


def test_scan_sorts_classes_and_skips_non_images(leaf_folder):
    index = scan_image_folder(leaf_folder)
    assert index.class_names == ("acer", "betula", "quercus")
    assert len(index.paths) == 24
    assert all(p.suffix == ".png" for p in index.paths)
    assert index.class_counts() == {"acer": 8, "betula": 8, "quercus": 8}
    # Labels follow the sorted class order, independent of filesystem order.
    assert index.labels[0] == 0 and index.labels[-1] == 2


def test_scan_rejects_empty_class(leaf_folder):
    (leaf_folder / "empty").mkdir()
    with pytest.raises(ValueError, match="no images"):
        scan_image_folder(leaf_folder)


def test_scan_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_image_folder(tmp_path / "missing")


def test_stratified_splits_keep_class_ratio_and_are_disjoint():
    labels = [c for c in range(10) for _ in range(12)]
    splits = repeated_splits(labels, repeats=5, test_size=0.3, seed=0)
    assert len(splits) == 5
    y = np.asarray(labels)
    for train_idx, test_idx in splits:
        assert not set(train_idx) & set(test_idx)
        assert len(train_idx) + len(test_idx) == len(labels)
        per_class_test = np.bincount(y[test_idx], minlength=10)
        assert per_class_test.min() >= 3 and per_class_test.max() <= 4


def test_splits_are_reproducible_and_differ_between_repeats():
    labels = [c for c in range(4) for _ in range(10)]
    first = repeated_splits(labels, repeats=3, seed=7)
    second = repeated_splits(labels, repeats=3, seed=7)
    for (a_train, a_test), (b_train, b_test) in zip(first, second, strict=True):
        assert np.array_equal(a_train, b_train) and np.array_equal(a_test, b_test)
    assert not np.array_equal(first[0][1], first[1][1])


def test_split_i_matches_seed_plus_i():
    labels = [c for c in range(4) for _ in range(10)]
    campaign = repeated_splits(labels, repeats=3, seed=5)
    alone = repeated_splits(labels, repeats=1, seed=7)
    assert np.array_equal(campaign[2][1], alone[0][1])


@pytest.mark.parametrize("kwargs", [{"repeats": 0}, {"repeats": 1, "test_size": 1.0}])
def test_split_argument_validation(kwargs):
    with pytest.raises(ValueError):
        repeated_splits([0, 1, 0, 1], **kwargs)


@pytest.mark.parametrize("augment", [False, True])
def test_dataset_returns_normalised_tensor(leaf_folder, augment):
    index = scan_image_folder(leaf_folder)
    dataset = LeafDataset(index.paths, index.labels, build_transforms(32, augment=augment))
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 32)
    assert label == 0
