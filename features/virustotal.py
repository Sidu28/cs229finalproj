from .extractor import FeatureExtractor
import os
import hashlib
from virustotal_python import Virustotal

VIRUSTOTAL_API_KEY = ""

class VirusTotalExtractor(FeatureExtractor):
  
  def __init__(self, file, pefile_parsed=None, lief_parsed=None):
    super().__init__(file, pefile_parsed, lief_parsed)
    self.endpoint = Virustotal(API_KEY=VIRUSTOTAL_API_KEY)

  def extract(self, **kwargs):
    features = {}
    
    # First hash the file so we don't re-request analysis on previously analyzed files
    md5_hash = hashlib.md5(open(self.file,'rb').read()).hexdigest()
    
    # Should there be a delay before analysis_response?
    analysis_response = self.endpoint.request("file/report", {"resource": md5_hash}).json()

    # Send over the file if we don't have a response
    if analysis_response["response_code"] == 0:
      encoding = {"file": (os.path.basename(self.file), open(os.path.abspath(self.file), "rb"))}
      queue_response = self.endpoint.request("file/scan", files=encoding, method="POST").json()
      analysis_response = self.endpoint.request("file/report", {"resource": queue_response["resource"]}).json()

    # Features for each scanner
    for scan in analysis_response['scans']:
      features['virustotal_' + scan] = analysis_response['scans'][scan]['detected']
    
    # Total # of positives
    features['virustotal_total_positives'] = analysis_response['positives']

    for key, value in features.items():
        features[key] = int(value)
    return features
