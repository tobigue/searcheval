import unittest

import searcheval.metrics as sm


class MetricsTests(unittest.TestCase):

    def test_mean(self):
        vector = [2, 3, 7]
        mean = sm.mean(vector)
        self.assertEqual(mean, 4)

    def test_precision(self):
        relevance_vector = [1, 0, 0, 1, 0]
        precision = sm.precision(relevance_vector)
        self.assertEqual(precision, 0.4)

    def test_precision_at_rank(self):
        relevance_vector = [1, 0, 0, 1, 0]
        rank = 2
        precision_at_rank = sm.precision_at_rank(relevance_vector, rank)
        self.assertEqual(precision_at_rank, 0.5)

    def test_precision_vector(self):
        relevance_vector = [1, 0]
        precision_vector = sm.precision_vector(relevance_vector)
        self.assertEqual(list(precision_vector), [1.0, 0.5])

    def test_avg_prec(self):
        relevance_vector = [1, 0]
        avg_prec = sm.avg_prec(relevance_vector)
        self.assertEqual(avg_prec, 0.5)

    def test_r_prec(self):
        relevance_vector = [1, 0, 0, 1, 0]
        recall_base = 2
        r_prec = sm.r_prec(relevance_vector, recall_base)
        self.assertEqual(r_prec, 0.5)

        # check that r_prec handles recall base larger than number of samples
        r_prec = sm.r_prec([1, 0], 5)
        self.assertEqual(r_prec, 0.2)

    def test_recall(self):
        relevance_vector = [1, 0, 0, 1, 0]
        recall_base = 4
        recall = sm.recall(relevance_vector, recall_base)
        self.assertEqual(recall, 0.5)

    def test_recall_at_rank(self):
        relevance_vector = [1, 0, 0, 1, 0]
        recall_base = 4
        rank = 2
        precision_at_rank = sm.recall_at_rank(relevance_vector, recall_base,
                                              rank)
        self.assertEqual(precision_at_rank, 0.25)

    def test_recall_vector(self):
        relevance_vector = [1, 0, 0, 1, 0]
        recall_base = 4
        recall_vector = sm.recall_vector(relevance_vector, recall_base)
        self.assertEqual(list(recall_vector), [0.25, 0.25, 0.25, 0.5, 0.5])

    def test_nDCG(self):
        # binary relevance
        gain_vector = [1, 1, 0, 0, 0]  # perfect query
        ideal_gain_vector = [1, 1, 0, 0, 0]
        nDCG = sm.nDCG(gain_vector, ideal_gain_vector)
        self.assertEqual(nDCG, 1.0)

        # graded relevance
        gain_vector = [3, 2, 1, 1, 0]  # perfect query
        ideal_gain_vector = [3, 2, 1, 1, 0]
        nDCG = sm.nDCG(gain_vector, ideal_gain_vector)
        self.assertEqual(nDCG, 1.0)

    def test_nDCG_at_rank(self):
        gain_vector = [1, 0, 1, 0, 0]
        ideal_gain_vector = [1, 1, 0, 0, 0]
        rank = 1
        nDCG_at_rank = sm.nDCG_at_rank(gain_vector, ideal_gain_vector, rank)
        self.assertEqual(nDCG_at_rank, 1.0)

        # not perfect query
        gain_vector = [1, 0, 1, 0, 0]
        ideal_gain_vector = [1, 1, 0, 0, 0]
        rank = 2
        nDCG_at_rank = sm.nDCG_at_rank(gain_vector, ideal_gain_vector, rank)
        self.assertTrue(nDCG_at_rank < 1.0)

    def test_nDCG_vector(self):
        gain_vector = [1, 1, 0, 0, 0]
        ideal_gain_vector = [1, 1, 0, 0, 0]
        nDCG_vector = sm.nDCG_vector(gain_vector, ideal_gain_vector)
        self.assertEqual(nDCG_vector, [1.0, 1.0, 1.0, 1.0, 1.0])


if __name__ == '__main__':
    unittest.main()
