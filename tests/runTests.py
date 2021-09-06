""" One can run all tests from this script if there is not an automated test procedure configured. """

import unittest

import tests.test_featureImportance
import tests.test_saliencyMap
import tests.test_wordImportance
import tests.test_decisionRule


if __name__ == '__main__':
    modules = [tests.test_featureImportance, tests.test_saliencyMap, tests.test_wordImportance, tests.test_decisionRule]
    moduleNames = ['featureImportance', 'saliencyMap', 'wordImportance', 'decisionRule']

    tests = [unittest.TestLoader().loadTestsFromModule(module) for module in modules]
    results = [unittest.TextTestRunner().run(testSuite) for testSuite in tests]

    nTestsFailed = 0
    for res in results:
        if len(res.failures) > 0:
            nTestsFailed += 1

    if nTestsFailed == 0:
        print('All tests passed.')
    else:
        print(f'{nTestsFailed} tests failed:')
        for i, res in enumerate(results):
            if len(res.failures) > 0:
                print(f'{repr(moduleNames[i])} -> {len(res.failures)} failures.')
