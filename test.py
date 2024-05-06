import unittest
import pandas as pd
from main import sum_sect_returns, multiply_sect_returns


class TestSumSectReturns(unittest.TestCase):
    def test_sum_sect_returns(self):
        # create a sample df for sect_returns
        sect_returns_data = {
            'A': [0, 0.1, 0.2, 0.3],
            'B': [0.2, 0.2, 0.3, 0.4],
            'C': [0.2, 0.3, 0.4, 0.5]
        }
        sect_returns_df = pd.DataFrame(sect_returns_data)
        print(sect_returns_df)

        # call the sum_sect_returns function
        summed_returns = list(sum_sect_returns(sect_returns_df))

        # define expected results - sums except first col
        expected_summed_returns = [0.4, 0.5, 0.7, 0.9]

        # test multiplier on summed_returns
        multiplied_df = pd.DataFrame(expected_summed_returns)
        multiply_sect_returns(10, multiplied_df)
        multiplied_returns = list(multiplied_df[0])

        # define expected results - sums except first col
        expected_multiplied_returns = [4.0, 2.0, 1.4, 1.26]

        # test for equality
        self.assertEqual(summed_returns, expected_summed_returns)
        self.assertEqual(multiplied_returns, expected_multiplied_returns)

if __name__ == '__main__':
    unittest.main()