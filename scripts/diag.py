import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import traceback
from src.inference.pipeline import ECLIPSEInferencePipeline

def run_test():
    try:
        pipe = ECLIPSEInferencePipeline(sector=1)
        res = pipe.run(261136679)
        print("RESULT CLASS:", res.get("predicted_class"))
        print("ERROR:", res.get("error"))
        print("SNR TLS:", res.get("snr_tls"))
        print("KEYS:", list(res.keys()))
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
