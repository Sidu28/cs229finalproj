import sys

sys.path.append('../')
from features.section_info import SectionInfoExtractor


def test_section_info():
    extractor = SectionInfoExtractor('data/goodware_example')
    output = list(extractor.extract().values())
    assert(type(output[i]) == float for i in range(len(output)))