from pathlib import Path

_newsgroup_explanation_urls = {'electronics': 'https://github.com/SinaMohseni/ML-Interpretability-Evaluation-Benchmark'
                                              '/tree/master/Text/20news_group/human_attention/sci.electronics',
                               'med': 'https://github.com/SinaMohseni/ML-Interpretability-Evaluation-Benchmark/tree'
                                      '/master/Text/20news_group/human_attention/sci.med'}

_newsgroup_data_urls = {'electronics': 'https://github.com/SinaMohseni/ML-Interpretability-Evaluation-Benchmark'
                                       '/tree/master/Text/20news_group/org_documents/20news-bydate/20news-bydate-test/'
                                       'sci.electronics',
                        'med': 'https://github.com/SinaMohseni/ML-Interpretability-Evaluation-Benchmark/tree/master/'
                               'Text/20news_group/org_documents/20news-bydate/20news-bydate-test/sci.med'}

# todo
_newsgroupIndexes = {}
_newsgroupNEntries = None
_newsgroupLabels = {}

_newgroupRoot = Path(__file__).parent.parent.absolute() / '/text/newgroup/'
