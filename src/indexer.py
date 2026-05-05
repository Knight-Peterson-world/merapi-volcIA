"""
indexer.py — Gestion de l'index centralisé des images Merapi.

Responsabilités :
- Créer et maintenir data/index/index.csv
- Fusionner les nouvelles entrées sans créer de doublons (clé = URL)
- Enrichir les métadonnées à partir des fichiers locaux
- Fournir des fonctions de lecture filtrée de l'index

Usage :
    from src.indexer import MerapiIndexer
    from src.utils import load_config, setup_logger

    config = load_config()
    setup_logger(config)

    indexer = MerapiIndexer(config)
    indexer.upsert(records)        # ajoute / met à jour depuis liste de dicts
    df = indexer.load()            # charge tout l'index
    df_nov = indexer.load_month(2014, 11)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
try:
    from loguru import logger
except ModuleNotFoundError:
    import logging as _logging
    import sys as _sys
    class _FallbackLogger:
        def __init__(self):
            self._l = _logging.getLogger("indexer")
            self._l.setLevel(_logging.INFO)
            if not self._l.handlers:
                _h = _logging.StreamHandler(_sys.stderr)
                _h.setFormatter(_logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
                self._l.addHandler(_h)
        def info(self, m, *a, **k): self._l.info(m)
        def warning(self, m, *a, **k): self._l.warning(m)
        def debug(self, m, *a, **k): self._l.debug(m)
        def error(self, m, *a, **k): self._l.error(m)
        def exception(self, m, *a, **k): self._l.exception(m)
    logger = _FallbackLogger()

from src.utils import get_index_path, ensure_dir, load_config, setup_logger, PROJECT_ROOT


# ============================================================
# Extensions d'images reconnues
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}


# ============================================================
# Schéma de colonnes de l'index
# ============================================================

INDEX_COLUMNS = [
    "url",              # URL source (clé primaire)
    "filename",         # nom du fichier image
    "local_path",       # chemin local absolu (str)
    "extension",        # .jpg, .jpeg, etc.
    "year",             # int
    "month",            # int (1-12)
    "day",              # int | None
    "hour",             # int | None
    "minute",           # int | None
    "second",           # int | None
    "downloaded",       # bool
    "file_size_bytes",  # int | None
    "quality_flag",     # str | None : 'usable', 'cloudy', 'dark', 'corrupted', 'unknown'
    "is_night",         # bool | None (déterminé lors du prétraitement)
    "anomaly_score",    # float | None (rempli lors de la détection — Phase 4/5)
    "notes",            # str | None (annotations manuelles éventuelles)
]

# Types cibles pour la lecture depuis CSV
INDEX_DTYPES = {
    "url": str,
    "filename": str,
    "local_path": str,
    "extension": str,
    "year": "Int64",
    "month": "Int64",
    "day": "Int64",
    "hour": "Int64",
    "minute": "Int64",
    "second": "Int64",
    "downloaded": bool,
    "file_size_bytes": "Int64",
    "quality_flag": str,
    "is_night": "boolean",
    "anomaly_score": float,
    "notes": str,
}


# ============================================================
# Indexer
# ============================================================

class MerapiIndexer:
    """
    Gestionnaire de l'index centralisé du dataset Merapi.

    L'index est un CSV dont la clé primaire est l'URL source.
    Toutes les opérations sont idempotentes : appeler upsert()
    plusieurs fois avec les mêmes URLs met à jour sans dupliquer.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.index_path = get_index_path(config)
        ensure_dir(self.index_path.parent)

    # ----------------------------------------------------------
    # Chargement
    # ----------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Charge l'index complet depuis le CSV.

        Returns:
            DataFrame avec toutes les colonnes de INDEX_COLUMNS.
            DataFrame vide (schéma correct) si l'index n'existe pas encore.
        """
        if not self.index_path.exists():
            logger.info("Index inexistant — retour d'un DataFrame vide.")
            return self._empty_dataframe()

        df = pd.read_csv(
            self.index_path,
            dtype=str,  # tout en str d'abord pour gérer les valeurs nulles
            na_values=["", "None", "nan", "NaN", "NA"],
            keep_default_na=True,
        )

        # Ajouter les colonnes manquantes (compatibilité lors des mises à jour de schéma)
        for col in INDEX_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA

        df = df[INDEX_COLUMNS]

        # Conversion de types
        df = self._cast_types(df)
        logger.debug(f"Index chargé : {len(df)} entrées depuis {self.index_path}")
        return df

    def load_month(self, year: int, month: int) -> pd.DataFrame:
        """Retourne les entrées de l'index pour un mois donné."""
        df = self.load()
        mask = (df["year"] == year) & (df["month"] == month)
        return df[mask].reset_index(drop=True)

    def load_downloaded(self) -> pd.DataFrame:
        """Retourne uniquement les images téléchargées avec succès."""
        df = self.load()
        return df[df["downloaded"] == True].reset_index(drop=True)

    def load_by_quality(self, flag: str) -> pd.DataFrame:
        """
        Filtre par quality_flag.

        Args:
            flag: 'usable', 'cloudy', 'dark', 'corrupted', 'unknown'

        Returns:
            DataFrame filtré.
        """
        df = self.load()
        return df[df["quality_flag"] == flag].reset_index(drop=True)

    # ----------------------------------------------------------
    # Écriture / mise à jour
    # ----------------------------------------------------------

    def upsert(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Insère ou met à jour des records dans l'index.

        Logique :
        - Si l'URL existe déjà → mise à jour des colonnes non-nulles du nouveau record.
        - Si l'URL est nouvelle → insertion.

        Args:
            records: liste de dicts (sortie du scraper ou enrichissement).

        Returns:
            DataFrame complet après mise à jour.
        """
        if not records:
            logger.warning("upsert() appelé avec une liste vide.")
            return self.load()

        new_df = self._records_to_dataframe(records)
        existing_df = self.load()

        if existing_df.empty:
            merged = new_df
        else:
            merged = self._merge(existing_df, new_df)

        self._save(merged)
        logger.info(
            f"Index mis à jour : {len(merged)} entrées totales "
            f"(+{len(new_df)} records traités)."
        )
        return merged

    def update_fields(
        self,
        urls: list[str],
        fields: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Met à jour des champs spécifiques pour une liste d'URLs.

        Utile pour enrichir l'index après prétraitement ou scoring.

        Args:
            urls: liste d'URLs à mettre à jour.
            fields: dict {nom_colonne: valeur} à appliquer.

        Returns:
            DataFrame mis à jour.
        """
        df = self.load()

        if df.empty:
            logger.warning("update_fields() : index vide, rien à mettre à jour.")
            return df

        mask = df["url"].isin(urls)
        found = mask.sum()

        if found == 0:
            logger.warning(f"Aucune URL trouvée parmi les {len(urls)} fournies.")
            return df

        for col, value in fields.items():
            if col not in df.columns:
                logger.warning(f"Colonne '{col}' inconnue dans l'index, ignorée.")
                continue
            df.loc[mask, col] = value

        self._save(df)
        logger.info(f"Champs {list(fields.keys())} mis à jour pour {found} entrées.")
        return df

    def sync_file_status(self) -> pd.DataFrame:
        """
        Resynchronise le champ 'downloaded' et 'file_size_bytes'
        en vérifiant l'existence réelle des fichiers locaux.

        Utile si des fichiers ont été supprimés/ajoutés manuellement.

        Returns:
            DataFrame mis à jour.
        """
        df = self.load()

        if df.empty:
            return df

        updated = 0
        for idx, row in df.iterrows():
            local_path = Path(str(row["local_path"]))
            # Si le chemin est relatif, le résoudre par rapport à PROJECT_ROOT
            if not local_path.is_absolute():
                local_path = PROJECT_ROOT / local_path
            exists = local_path.exists()
            new_downloaded = exists
            new_size = local_path.stat().st_size if exists else pd.NA

            if df.at[idx, "downloaded"] != new_downloaded:
                df.at[idx, "downloaded"] = new_downloaded
                df.at[idx, "file_size_bytes"] = new_size
                updated += 1

        if updated > 0:
            self._save(df)
            logger.info(f"sync_file_status : {updated} entrées resynchronisées.")
        else:
            logger.info("sync_file_status : tout est déjà synchronisé.")

        return df

    # ----------------------------------------------------------
    # Construction de l'index depuis les fichiers sur disque
    # ----------------------------------------------------------

    def build_from_disk(self) -> pd.DataFrame:
        """
        Construit (ou enrichit) l'index à partir des images réellement
        présentes dans data/raw/.

        Parcourt récursivement data/raw/{year}/{month}/*.{jpg,png,...},
        crée un record pour chaque fichier trouvé et fusionne avec
        l'index existant (sans dupliquer).

        Returns:
            DataFrame complet après mise à jour.
        """
        from src.utils import parse_filename_datetime, get_raw_image_dir

        raw_base = PROJECT_ROOT / self.config["paths"]["data_raw"]

        if not raw_base.exists():
            logger.warning(f"Dossier raw introuvable : {raw_base}")
            return self.load()

        records = []
        for year_dir in sorted(raw_base.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue

            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                try:
                    month = int(month_dir.name)
                except ValueError:
                    continue

                for img_path in sorted(month_dir.iterdir()):
                    if not img_path.is_file():
                        continue
                    if img_path.suffix not in IMAGE_EXTENSIONS:
                        continue

                    filename = img_path.name

                    # ── Filtre caméra Kalor ──────────────────────────────
                    if not filename.lower().startswith("kalor"):
                        continue
                    dt_info = parse_filename_datetime(filename) or {}

                    record = {
                        "url": f"file://{img_path}",
                        "filename": filename,
                        "local_path": str(img_path),
                        "extension": img_path.suffix.lower(),
                        "year": dt_info.get("year", year),
                        "month": dt_info.get("month", month),
                        "day": dt_info.get("day"),
                        "hour": dt_info.get("hour"),
                        "minute": dt_info.get("minute"),
                        "second": dt_info.get("second"),
                        "downloaded": True,
                        "file_size_bytes": img_path.stat().st_size,
                        "quality_flag": None,
                        "is_night": None,
                        "anomaly_score": None,
                        "notes": None,
                    }
                    records.append(record)

        if not records:
            logger.warning("Aucune image trouvée dans data/raw/.")
            return self.load()

        logger.info(f"build_from_disk : {len(records)} images trouvées sur disque.")

        # Fusionner avec l'index existant (par filename, car les URL file:// diffèrent des URL web)
        existing_df = self.load()
        new_df = self._records_to_dataframe(records)

        if existing_df.empty:
            merged = new_df
        else:
            # Fusionner par filename plutôt que par URL
            # (les fichiers scannés ont une URL file://, les scrapés une URL http://)
            existing_fnames = set(existing_df["filename"].dropna())
            to_insert = new_df[~new_df["filename"].isin(existing_fnames)]

            # Pour les fichiers déjà dans l'index, mettre à jour downloaded + file_size
            merged = existing_df.copy()
            for _, row in new_df[new_df["filename"].isin(existing_fnames)].iterrows():
                mask = merged["filename"] == row["filename"]
                merged.loc[mask, "downloaded"] = True
                merged.loc[mask, "file_size_bytes"] = row["file_size_bytes"]
                merged.loc[mask, "local_path"] = row["local_path"]

            merged = pd.concat([merged, to_insert], ignore_index=True)

        self._save(merged)
        n_dl = int((merged["downloaded"] == True).sum())
        logger.info(
            f"Index reconstruit : {len(merged)} entrées, {n_dl} téléchargées."
        )
        return merged

    # ----------------------------------------------------------
    # Statistiques rapides
    # ----------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """
        Retourne un résumé statistique de l'index.

        Returns:
            dict avec les métriques clés.
        """
        df = self.load()

        if df.empty:
            return {"status": "index vide", "total": 0}

        total = len(df)
        downloaded = int((df["downloaded"] == True).sum())
        missing = total - downloaded

        quality_counts = (
            df["quality_flag"].value_counts(dropna=False).to_dict()
        )

        year_counts = df["year"].value_counts().sort_index().to_dict()
        month_counts = (
            df.groupby(["year", "month"]).size()
            .reset_index(name="count")
            .to_dict(orient="records")
        )

        return {
            "total_indexed": total,
            "downloaded": downloaded,
            "not_downloaded": missing,
            "quality_distribution": quality_counts,
            "images_per_year": year_counts,
            "images_per_month": month_counts,
            "index_path": str(self.index_path),
        }

    def print_summary(self) -> None:
        """Affiche le résumé de l'index dans les logs."""
        s = self.summary()
        logger.info("=" * 50)
        logger.info(f"Index Merapi — résumé")
        logger.info(f"  Total indexé   : {s.get('total_indexed', 0)}")
        logger.info(f"  Téléchargé     : {s.get('downloaded', 0)}")
        logger.info(f"  Non téléchargé : {s.get('not_downloaded', 0)}")
        logger.info(f"  Qualité        : {s.get('quality_distribution', {})}")
        logger.info(f"  Par année      : {s.get('images_per_year', {})}")
        logger.info("=" * 50)

    # ----------------------------------------------------------
    # Méthodes internes
    # ----------------------------------------------------------

    def _empty_dataframe(self) -> pd.DataFrame:
        """Retourne un DataFrame vide avec le bon schéma."""
        return pd.DataFrame(columns=INDEX_COLUMNS)

    def _records_to_dataframe(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Convertit une liste de dicts en DataFrame normalisé."""
        df = pd.DataFrame(records)

        # Ajouter les colonnes manquantes
        for col in INDEX_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA

        df = df[INDEX_COLUMNS]
        df = self._cast_types(df)

        # Dédoublonnage interne (par URL)
        before = len(df)
        df = df.drop_duplicates(subset=["url"], keep="last")
        after = len(df)
        if before != after:
            logger.warning(f"Doublons internes supprimés : {before - after}")

        return df

    def _merge(self, existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        """
        Fusionne existing et new en utilisant 'url' comme clé.

        Pour les URLs existantes, les champs non-nuls du nouveau record
        écrasent les anciens. Les champs nuls ne modifient pas l'existant.
        """
        # URLs déjà présentes
        existing_urls = set(existing["url"].dropna())
        to_insert = new[~new["url"].isin(existing_urls)]
        to_update = new[new["url"].isin(existing_urls)]

        # Mise à jour des records existants
        updated = existing.copy()
        if not to_update.empty:
            # Mettre à jour ligne par ligne (préserve les champs non-nuls existants)
            updated = updated.set_index("url")
            for _, row in to_update.iterrows():
                url = row["url"]
                if url in updated.index:
                    for col in INDEX_COLUMNS:
                        if col == "url":
                            continue
                        new_val = row[col]
                        if pd.notna(new_val):
                            updated.at[url, col] = new_val
            updated = updated.reset_index()

        # Concaténation avec les nouveaux records
        merged = pd.concat([updated, to_insert], ignore_index=True)
        return merged

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique les types cibles aux colonnes de l'index."""
        for col, dtype in INDEX_DTYPES.items():
            if col not in df.columns:
                continue
            try:
                if dtype == bool:
                    df[col] = df[col].map(
                        lambda x: True if str(x).lower() == "true"
                        else (False if str(x).lower() == "false" else pd.NA)
                    )
                elif dtype == float:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif dtype in ("Int64", "boolean"):
                    df[col] = pd.array(
                        pd.to_numeric(df[col], errors="coerce"),
                        dtype=dtype
                    )
                else:
                    df[col] = df[col].astype(dtype, errors="ignore")
            except Exception:
                pass  # conserver la colonne telle quelle si conversion impossible

        return df

    def _save(self, df: pd.DataFrame) -> None:
        """Sauvegarde l'index sur disque de manière atomique."""
        # Tri chronologique avant sauvegarde
        sort_cols = [c for c in ["year", "month", "day", "hour", "minute"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

        # Écriture atomique (tmp puis rename)
        tmp_path = self.index_path.with_suffix(".tmp.csv")
        df.to_csv(tmp_path, index=False, encoding="utf-8")
        tmp_path.replace(self.index_path)

        logger.debug(f"Index sauvegardé : {self.index_path} ({len(df)} lignes)")


# ============================================================
# Point d'entrée en ligne de commande
# ============================================================

if __name__ == "__main__":
    cfg = load_config()
    setup_logger(cfg)

    indexer = MerapiIndexer(cfg)
    indexer.print_summary()
