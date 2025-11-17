"""
VLM-FO1 selfcheck CLI.

Run with: python -m vlm_fo1

This performs comprehensive diagnostics of the VLM-FO1 installation,
checking CUDA availability, driver versions, ABI compatibility, and
extension loading.
"""

import sys
from vlm_fo1._backend import selfcheck

if __name__ == '__main__':
    sys.exit(selfcheck())
