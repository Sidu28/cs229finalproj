from .extractor import FeatureExtractor

class SectionInfoExtractor(FeatureExtractor):
  
  def __init__(self, file, pefile_parsed=None, lief_parsed=None):
    super().__init__(file, pefile_parsed, lief_parsed)
  
  def extract(self, **kwargs):
    features = {}

    self.pefile_parse()

    features['num_sections'] = len(self.pefile_parsed.sections)

    for section in self.pefile_parsed.sections:
      normalized_name = section.Name.decode('ascii', 'ignore').split('\x00')[0][1:]
      section_entropy = section.get_entropy()
      features['section_entropy_' + normalized_name] = section_entropy

    return features