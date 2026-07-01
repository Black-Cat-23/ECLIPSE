import lightkurve as lk

for tic in ["261136246", "279741379", "100100827", "261136679"]:
    for sector in [1, 2]:
        search = lk.search_lightcurve(f"TIC {tic}", sector=sector, cadence="2min", mission="TESS", author="SPOC")
        print(f"TIC {tic} Sector {sector}: {len(search)} results")
