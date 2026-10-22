"""Checks for scientific failure modes and trainable model interfaces."""
import unittest
import numpy as np
import torch
from benchmark import NAMES, make_model, make_splits, normalize_from_training, rating_class, spectrogram


class BenchmarkChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_rating_boundaries_and_invalid_values(self):
        np.testing.assert_array_equal(rating_class([1, 3, 4, 6, 7, 9]), [0, 0, 1, 1, 2, 2])
        for invalid in [[0], [10], [2.5]]:
            with self.assertRaises(ValueError):
                rating_class(invalid)

    def test_subject_disjoint_splits_and_exact_test_coverage(self):
        subjects = np.repeat([s for s in range(1, 49) if s not in [5, 24, 42]], 74)
        splits = make_splits(subjects, 5, 42)
        tested = []
        for split in splits:
            train, val, test = (set(split[k]) for k in ["train_subjects", "validation_subjects", "test_subjects"])
            self.assertFalse(train & val or train & test or val & test)
            self.assertEqual(len(test), 9)
            self.assertEqual(train | val | test, set(np.unique(subjects)))
            tested.extend(test)
        self.assertEqual(sorted(tested), sorted(set(subjects)))
        self.assertEqual(splits, make_splits(subjects, 5, 42))

    def test_test_values_cannot_change_training_normalization(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(6, 14, 512)).astype(np.float32)
        original, mean, std = normalize_from_training(x, np.array([0, 1, 2, 3]))
        changed = x.copy()
        changed[4:] += 100000
        revised, revised_mean, revised_std = normalize_from_training(changed, np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(mean, revised_mean)
        np.testing.assert_array_equal(std, revised_std)
        np.testing.assert_array_equal(original[:4], revised[:4])

    def test_all_model_families_have_finite_gradients(self):
        torch.manual_seed(42)
        x = torch.randn(4, 14, 512)
        spectral = spectrogram(x)
        self.assertEqual(tuple(spectral.shape), (4, 14, 32, 32))
        for name in NAMES:
            with self.subTest(model=name):
                model = make_model(name)
                output = model(spectral if name.startswith("STFT") else x)
                self.assertEqual(tuple(output.shape), (4, 3))
                loss = torch.nn.functional.cross_entropy(output, torch.tensor([0, 1, 2, 1]))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                gradients = [p.grad for p in model.parameters() if p.requires_grad]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))


if __name__ == "__main__":
    unittest.main()
