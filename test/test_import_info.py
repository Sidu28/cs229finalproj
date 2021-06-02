import sys

sys.path.append('../')
from features.import_info import ImportInfoExtractor


def test_import_info():
    extractor = ImportInfoExtractor('data/goodware_example')
    assert (type(list(extractor.extract().values())[0]) == str)
