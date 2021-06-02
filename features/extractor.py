from abc import ABC, abstractmethod
import pefile
# import lief

"""
Abstract class required for feature extractors to override.
"""
class FeatureExtractor(ABC):

  """
  Every feature extractor operates on a PE file. Optionally, they can
  be provided with pre-parsed data from `pefile` or `lief`.
  """
  def __init__(self, file, pefile_parsed=None, lief_parsed=None):
    self.file = file
    self.pefile_parsed = pefile_parsed
    self.lief_parsed = lief_parsed
    super().__init__()
  
  """
  Setter method to allow for re-use of an extractor class across multiple
  files. We don't allow a user to set pefile_parsed, as this might result in
  internal inconsistencies.
  """
  def set_file(self, file):
    self.file = file

  """
  Helper method to generate pefile_parsed
  """
  def pefile_parse(self):
    if not self.pefile_parsed: self.pefile_parsed = pefile.PE(self.file)
  
  """
  Helper method to generate lief_parsed
  """
  def lief_parse(self):
    if not self.lief_parsed: self.lief_parsed = lief.parse(self.file)

  """
  Every feature extractor must expose a method extract, which takes optional 
  keyword arguments and returns a dictionary of features.
  """
  @abstractmethod
  def extract(self, **kwargs): pass