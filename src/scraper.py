"""
scraper.py — Pipeline de scraping des images Merapi.

Responsabilités :
- Parcourir les pages mensuelles du site de télésurveillance
- Extraire les liens vers les images (URLs absolues)
- Retourner une liste structurée de métadonnées par image
- Télécharger les images de manière progressive et contrôlée

Usage :
    from src.scraper import MerapiScraper
    from src.utils import load_config, setup_logger

    config = load_config()
    setup_logger(config)

    scraper = MerapiScraper(config)
    records = scraper.scrape_month(year=2014, month=11)
    scraper.download_images(records, max_images=50)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
try:
    from loguru import logger
except ModuleNotFoundError:
    import logging as _logging
    import sys as _sys
    class _FallbackLogger:
        def __init__(self):
            self._l = _logging.getLogger("scraper")
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

from src.utils import (
    get_monthly_page_url,
    get_raw_image_dir,
    parse_filename_datetime,
    setup_logger,
    load_config,
)


# ============================================================
# Constantes
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}

# Noms d'images de décoration/navigation à exclure
DECORATION_PATTERNS = {"barre_vert", "barre_hor", "merapi_title", "fleche", "arrow", "spacer"}

# ── Filtre caméra : seules les images Kalor sont utilisées dans ce projet ──
KALOR_PREFIX = "kalor"  # insensible à la casse


# ============================================================
# Scraper principal
# ============================================================

class MerapiScraper:
    """
    Scraper modulaire pour les archives d'images du volcan Merapi.

    Conçu pour être utilisé de manière progressive (mois par mois)
    et robuste aux erreurs réseau et aux variations du HTML source.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.src_cfg = config["source"]
        self.dl_cfg = config["download"]
        self.scr_cfg = config["scraping"]

        self.session = self._build_session()

    # ----------------------------------------------------------
    # Session HTTP
    # ----------------------------------------------------------

    def _build_session(self) -> requests.Session:
        """Crée une session requests avec headers appropriés."""
        session = requests.Session()
        session.headers.update({
            "User-Agent": self.scr_cfg.get(
                "user_agent",
                "MerapiAnomalyResearch/0.1 (research project)"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        return session

    # ----------------------------------------------------------
    # Requête HTTP robuste
    # ----------------------------------------------------------

    def _get_with_retry(self, url: str) -> requests.Response | None:
        """
        Effectue une requête GET avec retry exponentiel.

        Args:
            url: URL à récupérer.

        Returns:
            Response si succès, None si toutes les tentatives échouent.
        """
        max_retries = self.scr_cfg.get("max_retries", 3)
        timeout = self.scr_cfg.get("request_timeout_s", 20)
        backoff = self.scr_cfg.get("retry_backoff_s", 5.0)

        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                return response

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else "?"
                if status == 404:
                    logger.warning(f"Page introuvable (404) : {url}")
                    return None  # pas la peine de retenter
                logger.warning(f"HTTP {status} — tentative {attempt}/{max_retries} : {url}")

            except requests.exceptions.ConnectionError:
                logger.warning(f"Erreur connexion — tentative {attempt}/{max_retries} : {url}")

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout — tentative {attempt}/{max_retries} : {url}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Erreur inattendue : {e}")
                return None

            if attempt < max_retries:
                wait = backoff * attempt
                logger.debug(f"Attente {wait:.1f}s avant nouvelle tentative...")
                time.sleep(wait)

        logger.error(f"Échec définitif après {max_retries} tentatives : {url}")
        return None

    # ----------------------------------------------------------
    # Scraping d'une page mensuelle
    # ----------------------------------------------------------

    def scrape_month(self, year: int, month: int) -> list[dict[str, Any]]:
        """
        Parcourt la page mensuelle et extrait tous les liens d'images.

        Args:
            year: année (ex. 2014).
            month: mois 1–12.

        Returns:
            Liste de dicts, un par image trouvée, avec les métadonnées
            disponibles à ce stade (URL, nom de fichier, date partielle...).
            Retourne une liste vide si la page est inaccessible ou sans images.
        """
        page_url = get_monthly_page_url(self.config, year, month)
        logger.info(f"Scraping page mensuelle : {page_url}")

        # Délai poli entre requêtes
        time.sleep(self.scr_cfg.get("request_delay_s", 1.5))

        response = self._get_with_retry(page_url)
        if response is None:
            logger.warning(f"Page inaccessible pour {year}/{month:02d}")
            return []

        records = self._parse_image_links(response.text, page_url, year, month)
        logger.info(f"→ {len(records)} image(s) trouvée(s) pour {year}/{month:02d}")
        return records

    def _parse_image_links(
        self,
        html: str,
        page_url: str,
        year: int,
        month: int,
    ) -> list[dict[str, Any]]:
        """
        Parse le HTML d'une page mensuelle et extrait les métadonnées des images.

        Le site présente les images via des balises <img> et/ou des liens <a>
        pointant vers des fichiers .jpg. Les deux cas sont gérés.

        Args:
            html: contenu HTML brut de la page.
            page_url: URL de la page (pour résoudre les URLs relatives).
            year: année de la page (contexte).
            month: mois de la page (contexte).

        Returns:
            Liste de dicts avec les métadonnées extraites.
        """
        if not html or not html.strip():
            logger.warning("HTML vide reçu")
            return []

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        image_urls: set[str] = set()
        thumbnail_urls: list[str] = []  # thumbnails à upgrader en full-res

        # --- Cas 1 : balises <img src="..."> ---
        # Ère 1 (2014-2019) : thumbnails dans /kalor/icones/ ou /thumbnails/
        # Ère 3 (2021+)     : thumbnails dans /thumbnails/ech_kalor_xxx.jpg
        for tag in soup.find_all("img"):
            src = tag.get("src", "")
            if not src or src.startswith("file://"):
                continue
            if not self._is_image_url(src):
                continue
            abs_url = urljoin(page_url, src)
            path_lower = urlparse(abs_url).path.lower()
            if "/icones/" in path_lower or "/thumbnails/" in path_lower:
                thumbnail_urls.append(abs_url)
            else:
                image_urls.add(abs_url)

        # --- Cas 2 : liens <a href="..."> ---
        for tag in soup.find_all("a"):
            href = tag.get("href", "")
            if not href:
                continue

            # Ère 2 (2019-2020) : liens file:// Windows locaux
            if href.startswith("file://") and "/html_images/" in href and href.lower().endswith(".html"):
                img_url = self._file_href_to_image_url(href, year, month)
                if img_url:
                    image_urls.add(img_url)
                continue

            # Lien direct vers une image
            if self._is_image_url(href):
                image_urls.add(urljoin(page_url, href))
                continue

            # Ère 2-3 : lien vers une page de détail html_images/xxx.html
            if self._is_detail_page_link(href):
                img_url = self._detail_to_image_url(href, page_url)
                if img_url:
                    image_urls.add(img_url)

        # Upgrader les thumbnails en images full-res (probe une fois par mois)
        if thumbnail_urls:
            upgraded = self._upgrade_thumbnails(thumbnail_urls, year, month)
            image_urls.update(upgraded)

        # Filtrer les images de décoration / navigation
        image_urls = {u for u in image_urls if not self._is_decoration(u)}

        # ── Filtre caméra Kalor ──────────────────────────────────────────
        # Accepte : kalor_xxx.jpg  ET  ech_kalor_xxx.jpg (format 2018-2021+)
        before = len(image_urls)
        image_urls = {
            u for u in image_urls
            if Path(urlparse(u).path).stem.lower().startswith(KALOR_PREFIX)
            or Path(urlparse(u).path).stem.lower().startswith(f"ech_{KALOR_PREFIX}")
        }
        n_excluded = before - len(image_urls)
        if n_excluded:
            logger.debug(f"Filtre Kalor : {n_excluded} image(s) non-Kalor exclue(s)")

        if not image_urls:
            logger.warning(
                f"Aucune image trouvée sur {page_url}. "
                "Le format HTML a peut-être changé — inspecter manuellement."
            )
            return []

        records = []
        for url in sorted(image_urls):
            record = self._build_record(url, year, month)
            if record is not None:
                records.append(record)

        # Tri chronologique si la date est parsable (None → 0 pour le tri)
        records.sort(key=lambda r: (r.get("day") or 0, r.get("hour") or 0, r.get("minute") or 0))
        return records

    def _is_image_url(self, url: str) -> bool:
        """Vérifie si une URL pointe vers une image reconnue."""
        if not url:
            return False
        # Exclure les images de navigation/logo du site
        low = url.lower()
        if "title" in low or "logo" in low or "barre" in low:
            return False
        ext = Path(urlparse(url).path).suffix
        return ext in IMAGE_EXTENSIONS

    def _is_detail_page_link(self, href: str) -> bool:
        """Vérifie si un lien pointe vers une page de détail image."""
        if not href or href.startswith("file://"):
            return False
        low = href.lower()
        # Ères 2-3 : .../html_images/kalor_xxx.html
        # Ère ancienne : .../html/kalor_xxx.html
        has_html_path = "/html_images/" in low or "/html/" in low
        return (
            has_html_path
            and href.lower().endswith(".html")
            and any(c in low for c in ["canon", "cam", "kalor", "suki"])
        )

    def _detail_to_image_url(self, href: str, page_url: str) -> str | None:
        """
        Convertit un lien de page de détail en URL directe de l'image.

        Patterns :
          .../html_images/kalor_xxx.html → .../html_images/ech_kalor_xxx.jpg  (2020-2021+)
          .../html/Name.html             → .../images_publiques/Name.jpg      (ancien)
        """
        abs_url = urljoin(page_url, href)
        parsed = urlparse(abs_url)
        stem = Path(parsed.path).stem  # ex. "kalor_Canon_2021-05-31_17.00.13"
        dir_part = abs_url.rsplit("/", 1)[0]

        if "/html_images/" in abs_url:
            # Nouveau format : préfixe ech_
            return f"{dir_part}/ech_{stem}.jpg"
        elif "/html/" in abs_url:
            # Ancien format : images_publiques
            img_url = abs_url.replace("/html/", "/images_publiques/")
            return img_url.rsplit(".", 1)[0] + ".jpg"
        return None

    def _probe_detail_page(self, href: str, page_url: str) -> bool:
        """
        Teste si une page de détail est accessible (probe unique par mois).

        Retourne True si la page existe et contient un <img> exploitable,
        False sinon (404, timeout, etc.).
        """
        abs_url = urljoin(page_url, href)
        resp = self._get_with_retry(abs_url)
        if resp is None:
            return False
        try:
            soup = BeautifulSoup(resp.text, "html.parser")
            for img_tag in soup.find_all("img"):
                src = img_tag.get("src", "")
                if src and self._is_image_url(src):
                    return True
        except Exception:
            pass
        return False

    def _head_with_retry(self, url: str) -> "requests.Response | None":
        """Requête HEAD légère pour sonder l'existence d'une URL."""
        timeout = self.scr_cfg.get("request_timeout_s", 20)
        try:
            return self.session.head(url, timeout=timeout, allow_redirects=True)
        except requests.exceptions.RequestException:
            return None

    def _file_href_to_image_url(self, href: str, year: int, month: int) -> str | None:
        """
        Convertit un lien file:// Windows en URL serveur réelle.

        Pattern (ère 2019-2020) :
          file:///C|/Users/kelfoun/.../2020_04/html_images/kalor_xxx.html
          → {base_url}/2020_04/html_images/ech_kalor_xxx.jpg
        """
        import re
        fname = href.split("/")[-1]  # "kalor_Canon_2020-03-31_17.15.15.html"
        stem = Path(fname).stem      # "kalor_Canon_2020-03-31_17.15.15"

        # Ne garder que les caméras connues
        if not any(stem.lower().startswith(cam) for cam in ["kalor", "suki", "canon", "cam"]):
            return None

        # Extraire le dossier year_month depuis le chemin Windows
        m = re.search(r'(\d{4}_\d{2})[/\\]html_images', href)
        year_month = m.group(1) if m else f"{year}_{month:02d}"

        base_url = self.src_cfg["base_url"]
        return f"{base_url}/{year_month}/html_images/ech_{stem}.jpg"

    def _upgrade_thumbnails(
        self,
        thumbnail_urls: list[str],
        year: int,
        month: int,
    ) -> set[str]:
        """
        Convertit des URLs de thumbnails en images full-res.

        Deux patterns possibles selon l'ère :
          • /kalor/images_publiques/stem.jpg     (ère 2014-2017)
          • /html_images/ech_stem.jpg            (ère 2018-2020)
          • /html_images/ech_stem.jpg            (ère 2021+ via /thumbnails/)

        Une seule requête HEAD par mois est faite pour détecter le bon pattern.
        """
        if not thumbnail_urls:
            return set()

        year_month = f"{year}_{month:02d}"
        base_url = self.src_cfg["base_url"]
        sample = thumbnail_urls[0]
        parsed_sample = urlparse(sample)
        sample_stem = Path(parsed_sample.path).stem  # ex. "kalor_xxx" ou "ech_kalor_xxx"

        def _clean_stem(stem: str) -> str:
            """Retire le préfixe ech_ si présent."""
            return stem[4:] if stem.startswith("ech_") else stem

        if "/thumbnails/" in parsed_sample.path:
            # Ère 2021+ : thumbnails/ech_kalor_xxx.jpg → html_images/ech_kalor_xxx.jpg
            def to_fullres_thumb(u: str) -> str:
                s = _clean_stem(Path(urlparse(u).path).stem)
                return f"{base_url}/{year_month}/html_images/ech_{s}.jpg"
            return {to_fullres_thumb(u) for u in thumbnail_urls}

        elif "/icones/" in parsed_sample.path:
            # Ère 2014-2019 : essayer images_publiques d'abord, puis html_images/ech_
            stem_clean = _clean_stem(sample_stem)
            candidate_publi = f"{base_url}/{year_month}/kalor/images_publiques/{stem_clean}.jpg"
            candidate_ech   = f"{base_url}/{year_month}/html_images/ech_{stem_clean}.jpg"

            resp = self._head_with_retry(candidate_publi)
            if resp is not None and resp.status_code == 200:
                logger.debug(f"{year}/{month:02d} : pattern images_publiques détecté")
                def to_publi(u: str) -> str:
                    return u.replace("/icones/", "/images_publiques/")
                return {to_publi(u) for u in thumbnail_urls}

            resp2 = self._head_with_retry(candidate_ech)
            if resp2 is not None and resp2.status_code == 200:
                logger.debug(f"{year}/{month:02d} : pattern html_images/ech_ détecté")
                def to_ech(u: str) -> str:
                    s = _clean_stem(Path(urlparse(u).path).stem)
                    return f"{base_url}/{year_month}/html_images/ech_{s}.jpg"
                return {to_ech(u) for u in thumbnail_urls}

            logger.warning(
                f"{year}/{month:02d} : aucun pattern full-res accessible "
                f"(testé {candidate_publi} et {candidate_ech}) — thumbnails ignorés"
            )
            return set()

        return set()

    def _is_decoration(self, url: str) -> bool:
        """Vérifie si l'URL est une image de décoration/navigation."""
        path = urlparse(url).path.lower()
        filename = Path(path).stem
        return any(pat in filename for pat in DECORATION_PATTERNS)

    def _build_record(
        self,
        url: str,
        year: int,
        month: int,
    ) -> dict[str, Any] | None:
        """
        Construit un dictionnaire de métadonnées pour une image.

        Args:
            url: URL absolue de l'image.
            year: année de contexte (page mensuelle).
            month: mois de contexte.

        Returns:
            dict de métadonnées ou None si l'URL est invalide.
        """
        if not url:
            return None

        parsed = urlparse(url)
        filename = Path(parsed.path).name
        extension = Path(parsed.path).suffix.lower()

        if not filename:
            return None

        # Construire le chemin local attendu
        local_dir = get_raw_image_dir(self.config, year, month)
        local_path = local_dir / filename

        # Tenter de parser date/heure depuis le nom de fichier
        dt_info = parse_filename_datetime(filename) or {}

        record = {
            "url": url,
            "filename": filename,
            "local_path": str(local_path),
            "extension": extension,
            "year": dt_info.get("year", year),
            "month": dt_info.get("month", month),
            "day": dt_info.get("day"),
            "hour": dt_info.get("hour"),
            "minute": dt_info.get("minute"),
            "second": dt_info.get("second"),
            "downloaded": False,
            "file_size_bytes": None,
            "quality_flag": None,  # rempli lors du prétraitement (Phase 3)
            "is_night": None,       # rempli lors du prétraitement (Phase 3)
        }
        return record

    # ----------------------------------------------------------
    # Téléchargement progressif
    # ----------------------------------------------------------

    def download_images(
        self,
        records: list[dict[str, Any]],
        max_images: int | None = None,
        overwrite: bool | None = None,
    ) -> list[dict[str, Any]]:
        """
        Télécharge les images référencées dans la liste de records.

        Conçu pour une approche progressive : max_images permet de limiter
        la session de téléchargement sans modifier le code.

        Args:
            records: liste de dicts (sortie de scrape_month).
            max_images: nombre maximal d'images à télécharger dans cette session.
                        None = toutes.
            overwrite: si True, re-télécharge même si le fichier existe.
                       Prioritaire sur le paramètre de config.

        Returns:
            Liste des records mis à jour (champ 'downloaded' et 'file_size_bytes').
        """
        from tqdm import tqdm

        overwrite_cfg = self.dl_cfg.get("overwrite_existing", False)
        if overwrite is None:
            overwrite = overwrite_cfg

        min_size = self.dl_cfg.get("min_file_size_bytes", 1000)
        delay = self.dl_cfg.get("image_delay_s", 0.5)

        # Filtrer : à télécharger
        to_download = [
            r for r in records
            if overwrite or not Path(r["local_path"]).exists()
        ]

        if max_images is not None:
            to_download = to_download[:max_images]

        if not to_download:
            logger.info("Aucune image à télécharger (toutes déjà présentes).")
            return records

        logger.info(f"Téléchargement de {len(to_download)} image(s)...")

        success_count = 0
        fail_count = 0

        for record in tqdm(to_download, desc="Téléchargement images", unit="img"):
            url = record["url"]
            local_path = Path(record["local_path"])

            # Créer le dossier si besoin
            local_path.parent.mkdir(parents=True, exist_ok=True)

            time.sleep(delay)
            response = self._get_with_retry(url)

            if response is None:
                logger.warning(f"Échec téléchargement : {url}")
                fail_count += 1
                continue

            # Vérification taille minimale
            content = response.content
            if len(content) < min_size:
                logger.warning(
                    f"Fichier trop petit ({len(content)} octets), ignoré : {local_path.name}"
                )
                fail_count += 1
                continue

            # Écriture
            try:
                with open(local_path, "wb") as f:
                    f.write(content)

                # Mise à jour du record
                record["downloaded"] = True
                record["file_size_bytes"] = local_path.stat().st_size
                success_count += 1
                logger.debug(f"✓ {local_path.name} ({record['file_size_bytes']} octets)")

            except OSError as e:
                logger.error(f"Erreur écriture {local_path} : {e}")
                fail_count += 1

        logger.info(
            f"Téléchargement terminé — "
            f"{success_count} succès, {fail_count} échecs "
            f"sur {len(to_download)} tentatives."
        )
        return records

    # ----------------------------------------------------------
    # Scraping multi-mois
    # ----------------------------------------------------------

    def scrape_months(
        self,
        periods: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        """
        Scrape plusieurs mois en séquence.

        Args:
            periods: liste de tuples (year, month).
                     Exemple : [(2014, 11), (2014, 12), (2015, 1)]

        Returns:
            Liste agrégée de tous les records trouvés.
        """
        all_records: list[dict[str, Any]] = []

        for year, month in periods:
            records = self.scrape_month(year, month)
            all_records.extend(records)
            logger.info(
                f"Total cumulé après {year}/{month:02d} : {len(all_records)} images"
            )

        return all_records


# ============================================================
# Point d'entrée en ligne de commande (test rapide)
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scraper Merapi — un mois ou plusieurs années")
    parser.add_argument("--year", type=int, default=2014,
                        help="Année à scraper (mode mois unique)")
    parser.add_argument("--month", type=int, default=11,
                        help="Mois à scraper (mode mois unique, 1-12)")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Liste d'années à scraper en entier (ex: --years 2017 2018 2019). "
                             "Surpasse --year/--month et scrape tous les mois de chaque année.")
    parser.add_argument("--max-download", type=int, default=10,
                        help="Nombre maximal d'images à télécharger par session (0 = aucun, "
                             "-1 = tout télécharger)")
    args = parser.parse_args()

    cfg = load_config()
    setup_logger(cfg)

    scraper = MerapiScraper(cfg)

    # ── Mode multi-années ──────────────────────────────────────────────────
    if args.years:
        periods = [
            (year, month)
            for year in sorted(args.years)
            for month in range(1, 13)
        ]
        logger.info(
            "Mode multi-années : %d périodes à scraper (%s)",
            len(periods),
            ", ".join(str(y) for y in sorted(args.years)),
        )

        all_records: list[dict] = []
        year_totals: dict[int, int] = {}

        for year, month in periods:
            month_records = scraper.scrape_month(year, month)
            all_records.extend(month_records)
            year_totals[year] = year_totals.get(year, 0) + len(month_records)
            logger.info(
                "  %d/%02d — %d image(s) trouvée(s) | cumul année %d : %d",
                year, month, len(month_records), year, year_totals[year],
            )

        print(f"\n{'='*60}")
        print(f"Résumé multi-années :")
        for year in sorted(args.years):
            print(f"  {year} : {year_totals.get(year, 0)} image(s)")
        print(f"  TOTAL : {len(all_records)} image(s)")
        print(f"{'='*60}")

        max_dl = args.max_download
        if max_dl != 0 and all_records:
            dl_limit = None if max_dl < 0 else max_dl
            scraper.download_images(all_records, max_images=dl_limit)

    # ── Mode mois unique (comportement historique) ─────────────────────────
    else:
        records = scraper.scrape_month(args.year, args.month)

        print(f"\n{len(records)} image(s) trouvée(s) pour {args.year}/{args.month:02d}")
        if records:
            print("Exemple de record :")
            for k, v in records[0].items():
                print(f"  {k}: {v}")

        if args.max_download > 0 and records:
            scraper.download_images(records, max_images=args.max_download)
