import importlib
from io import BytesIO

import numpy as np
import requests


def _parse_fits_bytes(data: bytes, tic_id: int, sector: int):
    """Very minimal FITS binary table reader for TESS _lc.fits files."""
    try:
        astropy = importlib.import_module("astropy.io.fits")
        with astropy.open(BytesIO(data)) as hdul:
            t = hdul[1].data["TIME"].astype(float)
            f = hdul[1].data["PDCSAP_FLUX"].astype(float)
            e = hdul[1].data["PDCSAP_FLUX_ERR"].astype(float)
            ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
            return {
                "time": t[ok],
                "flux": f[ok],
                "flux_err": e[ok],
                "meta": {"tic": tic_id, "sector": sector},
            }
    except Exception:
        return None


def fetch_tess_lc(tic_id: int, sector: int) -> dict | None:
    """Download a TESS 2-min cadence light curve from MAST for a given TIC and sector.

    Returns dict with keys: time, flux, flux_err, meta.
    Falls back to None if the network/unparse fails.
    """
    url = (
        f"https://mast.stsci.edu/api/v0.1/Download/file?uri="
        f"mast:TESS/product/tess2018206045859-s0001-{tic_id:016d}-0120-s_lc.fits"
    )

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return _parse_fits_bytes(resp.content, tic_id, sector)
    except Exception:
        pass

    return None

