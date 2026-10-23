"""Checks for the v3 runner: same evaluation as v1, alignment is per subject and label-free,
normalisation and inner folds never touch test subjects, 2-class heads, demean behaviour."""
import unittest

import numpy as np
import torch

import benchmark as v1
import benchmark_v3 as v3


class BenchmarkV3Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        rng = np.random.default_rng(0)
        cls.cont = (rng.normal(size=(6, 14, 19200)) * 10 + 4000).astype(np.float32)
        cls.subjects = np.array([1, 1, 2, 2, 3, 3])

    def test_evaluation_windows_match_v1(self):
        rec, start = v3.window_grid(np.arange(6), 512)
        self.assertEqual(len(rec), 6 * 37)
        self.assertEqual(start.max(), 18432)
        x = v3.Windows(self.cont, torch.device("cpu"), demean=False).cut(rec, start, normalize=False).numpy()
        for i in [0, 5, 100, len(rec) - 1]:
            np.testing.assert_array_equal(x[i], self.cont[rec[i], :, start[i]:start[i] + 512])

    def test_alignment_depends_only_on_the_subjects_own_data(self):
        for step in ["ea", "sag", "sa"]:
            a = v3.prepare_variant(f"raw-{step}", {"raw": self.cont}, self.subjects)
            changed = self.cont.copy()
            changed[2:] = changed[2:] * 7 + 3      # other subjects' recordings change
            b = v3.prepare_variant(f"raw-{step}", {"raw": changed}, self.subjects)
            np.testing.assert_allclose(a[:2], b[:2], rtol=1e-4, atol=1e-4, err_msg=step)

    def test_alignment_is_label_free_and_symmetric_in_recordings(self):
        # swapping a subject's two recordings (e.g. rest/task order) just swaps the aligned outputs
        a = v3.prepare_variant("raw-ea-sag", {"raw": self.cont}, self.subjects)
        swapped = self.cont[[1, 0, 2, 3, 4, 5]]
        b = v3.prepare_variant("raw-ea-sag", {"raw": swapped}, self.subjects)
        np.testing.assert_allclose(a[[1, 0]], b[:2], rtol=1e-4, atol=1e-4)

    def test_ea_whitens_each_subject(self):
        a = v3.prepare_variant("raw-ea", {"raw": self.cont}, self.subjects)
        for s in [1, 2, 3]:
            ids = np.flatnonzero(self.subjects == s)
            segs = np.stack([a[r, :, t:t + 512] for r in ids for t in range(0, 19200 - 511, 512)]).astype(np.float64)
            segs -= segs.mean(-1, keepdims=True)
            C = np.einsum("nct,ndt->cd", segs, segs) / (len(segs) * 512)
            np.testing.assert_allclose(C, np.eye(14), atol=1e-3)

    def test_variant_names(self):
        for ok in ["raw", "filtered", "filtered-ea", "filtered-ea-sag", "raw-sa-ea"]:
            self.assertTrue(v3.valid_variant(ok), ok)
        for bad in ["clean", "filtered-xx", "filtered-ea-ea"]:
            self.assertFalse(v3.valid_variant(bad), bad)

    def test_inner_groups_cover_development_subjects_once(self):
        dev = list(range(1, 37))
        groups = v3.inner_groups(dev, 4, 43)
        flat = sorted(s for g in groups for s in g)
        self.assertEqual(flat, dev)
        self.assertEqual(len(groups), 4)

    def test_folds_identical_to_v1(self):
        subjects = np.repeat(np.arange(1, 46), 74)
        self.assertEqual(v3.make_splits(subjects, 5, 42), v1.make_splits(subjects, 5, 42))

    def test_two_class_heads(self):
        old = v3.CLASSES
        try:
            v3.CLASSES = ["rest", "task"]
            x = torch.randn(3, 14, 512)
            for name in v3.DEEP:
                model = v3.make_model(name).eval()
                with torch.no_grad():
                    self.assertEqual(tuple(model(x).shape), (3, 2), name)
        finally:
            v3.CLASSES = old

    def test_three_class_models_unchanged(self):
        for name in v3.DEEP:
            a = sum(p.numel() for p in v3.make_model(name).parameters())
            b = sum(p.numel() for p in v1.make_model(name).parameters())
            self.assertEqual(a, b, name)


if __name__ == "__main__":
    unittest.main()
