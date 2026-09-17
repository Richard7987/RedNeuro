import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/372/htru2.zip"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CSV_NAME = "HTRU_2.csv"


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / CSV_NAME
    if csv_path.exists():
        print(f"Ya existe {csv_path}, no se descarga de nuevo.")
        return

    zip_path = RAW_DIR / "htru2.zip"
    print(f"Descargando {URL} ...")
    urllib.request.urlretrieve(URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(CSV_NAME, RAW_DIR)

    zip_path.unlink()
    print(f"Listo: {csv_path}")


if __name__ == "__main__":
    download()
