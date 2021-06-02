from .extractor import FeatureExtractor


class DataDirectoryInfoExtractor(FeatureExtractor):

    def __init__(self, file, pefile_parsed=None, lief_parsed=None):
        super().__init__(file, pefile_parsed, lief_parsed)

    def extract(self, **kwargs):
        features = {}

        self.pefile_parse()
        pe = self.pefile_parsed

        for dir_entry in pe.OPTIONAL_HEADER.DATA_DIRECTORY:
            name = dir_entry.name
            features[name + '_size'] = dir_entry.Size
            features[name + '_rva'] = dir_entry.VirtualAddress

        return features