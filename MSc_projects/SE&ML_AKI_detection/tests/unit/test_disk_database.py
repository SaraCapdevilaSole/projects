import acces_main_directory

import unittest
import os
import pandas as pd
from disk_database import DiskDatabase



class TestDiskDatabase(unittest.TestCase):
    def setUp(self):
        # Create test_patients.csv
        with open('tests/testdata/test_patients.csv', 'w') as file:
            file.write('MRN,AGE,SEX\n')

        # Create test_results.csv
        with open('tests/testdata/test_results.csv', 'w') as file:
            file.write('MRN,DATE,RESULT\n')

        # Create test_history.csv
        with open('tests/testdata/test_history.csv', 'w') as file:
            file.write('MRN,DATE,RESULT\n')

        self.db = DiskDatabase('tests/testdata/test_patients.csv',
                               'tests/testdata/test_results.csv', 
                               'tests/testdata/test_history.csv')

    def tearDown(self):
        os.remove('tests/testdata/test_patients.csv')
        os.remove('tests/testdata/test_results.csv')
        os.remove('tests/testdata/test_history.csv')


    def test_admit(self):
        msg = {'MRN': '123', 'AGE': '30', 'SEX': 'M'}
        self.db.admit(msg)
        self.assertTrue('123' in self.db.patients_file['MRN'].values)

    def test_result(self):
        msg = {'MRN': '123', 'DATE': '2022-01-01', 'RESULT': 'Positive'}
        self.db.result(msg)
        self.assertTrue('123' in self.db.results_file['MRN'].values)

    def test_extract_model_info(self):
        msg_patient = {'MRN': '123', 'AGE': '30', 'SEX': 'M'}
        msg_result = {'MRN': '123', 'DATE': '2022-01-01', 'RESULT': 'Positive'}
        self.db.admit(msg_patient)
        self.db.result(msg_result)
        info = self.db.extract_model_info('123')
        self.assertEqual(info, {
            'AGE': '30',
            'SEX': 'M',
            'LATEST_DATE': '2022-01-01',
            'LATEST': 'Positive'
        })


if __name__ == '__main__':
    unittest.main()
