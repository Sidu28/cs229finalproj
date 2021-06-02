import sys

sys.path.append('../')
from features.imported_symbols import ImportedSymbolsExtractor
from features.exported_symbols import ExportedSymbolsExtractor


def test_imported_exported_symbols():
    extractor_import = ImportedSymbolsExtractor('data/goodware_example')
    extractor_export = ExportedSymbolsExtractor('data/goodware_example')
    if extractor_import.extract() and extractor_export.extract():
        assert (type(extractor_import.extract()) == int and type(extractor_export.extract()) == int)

