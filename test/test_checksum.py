import sys

sys.path.append('../')
from features.checksum import ChecksumExtractor


def test_checksum():
    extractor = ChecksumExtractor('data/goodware_example')
    assert ((list(extractor.extract().values())[0]) > 0 )