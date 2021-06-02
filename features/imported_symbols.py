from .extractor import FeatureExtractor


class ImportedSymbolsExtractor(FeatureExtractor):

    def __init__(self, file, pefile_parsed=None, lief_parsed=None):
        super().__init__(file, pefile_parsed, lief_parsed)

    def extract(self, **kwargs):
        features = {}

        self.pefile_parse()

        pe = self.pefile_parsed

        pe.parse_data_directories()

        import_list = []
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                DLL = entry.dll
                for imp in entry.imports:
                    features[imp.name] = 1

        return features