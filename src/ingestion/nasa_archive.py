"""
NASA Exoplanet Archive Cross-Matching via IPAC TAP (Table Access Protocol) API.
Queries confirmed exoplanets (PS table) and TESS Project Candidates (TOI table).
"""
import requests
from typing import Dict, Optional
from loguru import logger

NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

def check_nasa_exoplanet_archive(tic_id: int) -> Dict[str, Optional[str]]:
    """
    Check if a TIC ID is in NASA Exoplanet Archive confirmed planets or TOI catalog.
    Returns:
        is_confirmed (bool), planet_name (str), discovery_facility (str), disposition (str)
    """
    result = {
        "is_confirmed": False,
        "planet_name": None,
        "discovery_facility": None,
        "disposition": "Unknown",
        "archive_url": f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic_id}"
    }

    try:
        # 1. Check TOI (TESS Objects of Interest)
        query = f"SELECT toi, tfopwg_disp, pl_name FROM toi WHERE tid={tic_id}"
        resp = requests.get(NASA_TAP_URL, params={"query": query, "format": "json"}, timeout=4.0)
        if resp.status_code == 200:
            rows = resp.json()
            if rows and len(rows) > 0:
                row = rows[0]
                result["disposition"] = row.get("tfopwg_disp") or "TOI Candidate"
                if row.get("pl_name"):
                    result["is_confirmed"] = True
                    result["planet_name"] = row.get("pl_name")
                    result["discovery_facility"] = "TESS / NASA"
                    return result
                result["planet_name"] = f"TOI-{row.get('toi')}"
                result["discovery_facility"] = "TESS"
                if result["disposition"] in ("CP", "KP"):  # Confirmed Planet / Known Planet
                    result["is_confirmed"] = True
                return result

        # 2. Check Confirmed Planets (ps table)
        query_ps = f"SELECT pl_name, disc_facility FROM ps WHERE tic_id='TIC {tic_id}'"
        resp_ps = requests.get(NASA_TAP_URL, params={"query": query_ps, "format": "json"}, timeout=4.0)
        if resp_ps.status_code == 200:
            rows_ps = resp_ps.json()
            if rows_ps and len(rows_ps) > 0:
                result["is_confirmed"] = True
                result["planet_name"] = rows_ps[0].get("pl_name")
                result["discovery_facility"] = rows_ps[0].get("disc_facility") or "NASA Exoplanet Archive"
                result["disposition"] = "Confirmed Planet"

    except Exception as e:
        logger.debug(f"NASA TAP check for TIC {tic_id} timed out or failed: {e}")

    return result
