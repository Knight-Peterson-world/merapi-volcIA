"""
ingestion/ — Téléchargement, indexation et synchronisation des données Merapi.

Modules :
    downloader.py  - Scraping + téléchargement (Kalor uniquement, 30 imgs/mois)
    indexer.py     - IndexManager : source unique de vérité (index.csv)
    quality.py     - QualityClassifier : usable / dark / cloudy / corrupted
    processor.py   - ImageProcessor : preprocessing garanti sans quality_flag='unknown'
"""
from src.ingestion.downloader import KalorDownloader
from src.ingestion.indexer import IndexManager
from src.ingestion.quality import QualityClassifier
from src.ingestion.processor import ImageProcessor

__all__ = [
    "KalorDownloader",
    "IndexManager",
    "QualityClassifier",
    "ImageProcessor",
]
