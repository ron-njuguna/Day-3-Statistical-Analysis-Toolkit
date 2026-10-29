"""Unit tests for the toolkit. Run with:  python -m unittest -v   (or pytest).

All inputs are tiny and hand-checkable; the real dataset is not used.
"""
import csv
import tempfile
import unittest
from pathlib import Path

from stats_toolkit import (
    StatisticsAnalyzer,
    iqr_outliers,
    load_column,
    mean,
    median,
    mode,
    quartiles,
    std_dev,
    variance,
)


class TestCentralTendency(unittest.TestCase):
    def test_mean_of_known_values(self):
        self.assertEqual(mean([1, 2, 3, 4]), 2.5)

    def test_median_even_and_odd_lengths(self):
        self.assertEqual(median([3, 1, 2]), 2)         
        self.assertEqual(median([4, 1, 3, 2]), 2.5)    

    def test_mode_with_ties(self):
        self.assertEqual(mode([1, 1, 2, 2, 3]), [1, 2])   
        self.assertEqual(mode([1, 2, 2]), [2])           
        self.assertEqual(mode([1, 2, 3]), [])             

    def test_empty_input_raises(self):
        for func in (mean, median, mode, variance, std_dev, quartiles, iqr_outliers):
            with self.subTest(func=func.__name__):
                with self.assertRaises(ValueError):
                    func([])


class TestSpread(unittest.TestCase):
    DATA = [2, 4, 4, 4, 5, 5, 7, 9] 

    def test_sample_vs_population_variance(self):
        self.assertAlmostEqual(variance(self.DATA, sample=False), 4.0)      
        self.assertAlmostEqual(variance(self.DATA, sample=True), 32 / 7)    
        self.assertAlmostEqual(std_dev(self.DATA, sample=False), 2.0)

    def test_std_dev_single_value(self):
        self.assertEqual(std_dev([5], sample=False), 0.0)
        with self.assertRaises(ValueError):   
            std_dev([5], sample=True)


class TestQuartiles(unittest.TestCase):
    def test_quartiles_linear_method(self):
        self.assertEqual(quartiles([1, 2, 3, 4, 5]), (2, 3, 4))
        self.assertEqual(quartiles([1, 2, 3, 4]), (1.75, 2.5, 3.25))


class TestOutliers(unittest.TestCase):
    def test_no_outliers_in_uniform_data(self):
        result = iqr_outliers([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(result["outliers"], [])
        self.assertEqual((result["q1"], result["q3"], result["iqr"]), (3, 7, 4))
        self.assertEqual((result["lower_fence"], result["upper_fence"]), (-3, 13))

    def test_detects_obvious_outlier(self):
        result = iqr_outliers([1, 2, 3, 4, 5, 6, 7, 8, 100])
        self.assertEqual(result["outliers"], [100])

    def test_without_outliers_returns_new_analyzer(self):
        original = StatisticsAnalyzer([1, 2, 3, 4, 5, 6, 7, 8, 100])
        cleaned = original.without_outliers()
        self.assertEqual(original.count(), 9)   
        self.assertEqual(cleaned.count(), 8)
        self.assertEqual(cleaned.mean(), 4.5)
        self.assertIsNot(cleaned, original)


class TestLoading(unittest.TestCase):
    def _write_csv(self, amounts):
        folder = tempfile.mkdtemp()
        path = Path(folder) / "sample.csv"
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "amount"])
            for i, amount in enumerate(amounts, start=1):
                writer.writerow([i, amount])
        return path

    def test_invalid_and_missing_values_are_rejected(self):
        path = self._write_csv(["10", "", "-", "n/a", "12,40O", "-5", " 20 ", "30.5"])
        valid, rejected = load_column(path, "amount")
        self.assertEqual(valid, [10.0, 20.0, 30.5])
        # row numbers are file lines (header = line 1)
        self.assertEqual(rejected, [(3, ""), (4, "-"), (5, "n/a"), (6, "12,40O"), (7, "-5")])

    def test_unknown_column_raises(self):
        path = self._write_csv(["1"])
        with self.assertRaises(KeyError):
            load_column(path, "nope")


if __name__ == "__main__":
    unittest.main()
