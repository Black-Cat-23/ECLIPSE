import lightkurve as lk
import numpy as np
from pathlib import Path

def precache_target(tic_id, sector):
    # Fixed to use data/raw which is where TESSFetcher looks by default
    out_dir = Path(f'data/raw/sector_{sector:02d}')
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fixed filename convention: TIC{id}_S{sector:02d}.npz
    out_path = out_dir / f'TIC{tic_id}_S{sector:02d}.npz'

    print(f"Searching for TIC {tic_id} sector {sector}...")
    sr = lk.search_lightcurve(f'TIC {tic_id}', sector=sector, cadence='2min', mission='TESS', author='SPOC')
    if len(sr) == 0:
        print(f"No data found for TIC {tic_id} sector {sector}")
        return

    print("Downloading...")
    lc = sr.download().select_flux('pdcsap_flux').remove_nans()
    q = (lc.quality == 0)

    np.savez_compressed(
        out_path,
        time=lc.time.value[q].astype('float32'),
        flux=lc.flux.value[q].astype('float32'),
        flux_err=lc.flux_err.value[q].astype('float32')
    )
    print(f'Saved to {out_path}. Shape:', lc.flux.value[q].shape)

if __name__ == "__main__":
    # HD 21749
    precache_target(261136679, 1)
    # WASP-18
    precache_target(100100827, 2)
