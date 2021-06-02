from .extractor import FeatureExtractor
from capstone import *
from collections import defaultdict

class ASMExtractor(FeatureExtractor):
  
  def __init__(self, file, n=5, pefile_parsed=None, lief_parsed=None):
    super().__init__(file, pefile_parsed, lief_parsed)
    self.n = n

  def disassemble(self):
    self.pefile_parse()

    disassembly = ""
    opcodes = []
    mnemonics_freq = defaultdict(int)
    for section in self.pefile_parsed.sections:
      disassembly += str(section.Name)
      md = Cs(CS_ARCH_X86, CS_MODE_32)
      for j in md.disasm(section.get_data(), self.pefile_parsed.OPTIONAL_HEADER.ImageBase + section.VirtualAddress):
        disassembly += "0x%x:\t%s\t%s\n" % (j.address, j.mnemonic, j.op_str)
        opcodes.append(j.mnemonic)
        mnemonics_freq[j.mnemonic + '_count'] += 1
    return disassembly, opcodes, mnemonics_freq

  def generate_ngrams(self, opcodes):
    output = []
    for count, i in enumerate(range(len(opcodes) - self.n + 1)):
      output.append(opcodes[i:i + self.n])
      if count == 20:
        break
    return output
  
  def extract(self, **kwargs):
    num_features = {}
    alph_features = {
      'opcode_ngrams': None,
      'opcode_seq' : None
    }

    
    disassembly, opcodes, mnemonics_freq = self.disassemble()
    
    # Separated into numeric features and alphabetical features
    num_features.update(mnemonics_freq)
    alph_features['opcode_ngrams'] = self.generate_ngrams(opcodes)

    alph_features['opcode_seq'] = opcodes

    return alph_features, num_features