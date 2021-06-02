import sys
sys.path.append('../')
from features.asm import ASMExtractor

def test_disassemble():
  extractor = ASMExtractor('data/goodware_example')
  disassembly,_,_ = extractor.disassemble()
  
  # Make sure that the last disassembly line prints the right pointer
  assert('0x644a0052013' in disassembly.split('\n')[-2])