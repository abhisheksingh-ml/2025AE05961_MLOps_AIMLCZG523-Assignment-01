"""Download the UCI Heart Disease (Cleveland) dataset into data/raw/.

Run:  python data/download_data.py
Source page: https://archive.ics.uci.edu/ml/datasets/heart+Disease
The file has 303 rows: 13 clinical features + 1 target column called "num". Missing values are written as "?".
"""
import urllib.request
from pathlib import Path

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")
SAVE_AS = Path(__file__).resolve().parent / "raw" / "processed.cleveland.data"


def download():
    SAVE_AS.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_AS.exists():
        print("Already downloaded:", SAVE_AS)
        return
    print("Downloading", URL)
    urllib.request.urlretrieve(URL, SAVE_AS)
    print("Saved to", SAVE_AS)


if __name__ == "__main__":
    download()
