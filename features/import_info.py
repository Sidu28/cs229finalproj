from .extractor import FeatureExtractor

class ImportInfoExtractor(FeatureExtractor):
  
  def __init__(self, file, pefile_parsed=None, lief_parsed=None):
    super().__init__(file, pefile_parsed, lief_parsed)
  
  def extract(self, **kwargs):
    features = {}

    self.pefile_parse()

    features['imphash'] = self.pefile_parsed.get_imphash()

    return features