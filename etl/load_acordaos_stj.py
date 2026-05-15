"""
ETL: Espelhos de Acórdãos STJ — Portal de Dados Abertos do STJ.

Fonte: dadosabertos.web.stj.jus.br (Creative Commons Attribution)
Datasets: Corte Especial + 3 Seções + 6 Turmas (10 órgãos julgadores)
Formato: JSONs mensais incrementais (mai/2022 – presente)
Destino: tabela jurisprudencia (tribunal='STJ') + FTS5

Execução:
    python -m etl.load_acordaos_stj                  # 2022+ (padrão)
    python -m etl.load_acordaos_stj --desde 2023     # somente 2023+
    python -m etl.load_acordaos_stj --force          # apaga STJ existentes e recarrega
    python -m etl.load_acordaos_stj --dry-run        # lista arquivos disponíveis, não baixa
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sqlite3
import zipfile
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

CKAN_BASE = "https://dadosabertos.web.stj.jus.br/api/3/action"
CACHE_DIR = Path("data/stj/acordaos")
DEFAULT_DB = "data/db/iajuris.db"
DOWNLOAD_TIMEOUT = 120  # segundos por arquivo

DATASETS = [
    "espelhos-de-acordaos-corte-especial",
    "espelhos-de-acordaos-primeira-secao",
    "espelhos-de-acordaos-segunda-secao",
    "espelhos-de-acordaos-terceira-secao",
    "espelhos-de-acordaos-primeira-turma",
    "espelhos-de-acordaos-segunda-turma",
    "espelhos-de-acordaos-terceira-turma",
    "espelhos-de-acordaos-quarta-turma",
    "espelhos-de-acordaos-quinta-turma",
    "espelhos-de-acordaos-sexta-turma",
]

# ---------------------------------------------------------------------------
# Helpers de dados
# ---------------------------------------------------------------------------


def _val(v: object) -> str | None:
    """Converte 'None' string / None / '' para None; demais → str stripped."""
    if v is None:
        return None
    s = str(v).strip()
    return None if s in ("None", "") else s


def _convert_date(s: str | None) -> str | None:
    """Converte YYYYMMDD → DD/MM/YYYY. Retorna None para entradas inválidas."""
    if not s:
        return None
    s = s.strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[6:]}/{s[4:6]}/{s[:4]}"
    return s


def _build_ementa(row: dict) -> str | None:
    """
    Monta o texto completo da ementa combinando:
      Rel. X. ÓRGÃO.
      {ementa principal}
      TESE JURÍDICA: {teseJuridica}         (se presente)
      INFORMAÇÕES COMPLEMENTARES: {ICE}     (se presente)
    """
    parts: list[str] = []

    relator = _val(row.get("ministroRelator"))
    orgao = _val(row.get("nomeOrgaoJulgador"))
    header_items = []
    if relator:
        header_items.append(f"Rel. {relator}")
    if orgao:
        header_items.append(orgao)
    if header_items:
        parts.append(". ".join(header_items) + ".")

    ementa = _val(row.get("ementa"))
    if ementa:
        parts.append(ementa)

    tese = _val(row.get("teseJuridica"))
    if tese:
        parts.append(f"TESE JURÍDICA: {tese}")

    ice = _val(row.get("informacoesComplementares"))
    if ice:
        parts.append(f"INFORMAÇÕES COMPLEMENTARES: {ice}")

    return "\n\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path, client: httpx.Client) -> bool:
    """
    Baixa um arquivo em streaming para dest.
    Retorna True se baixou, False se já existia no cache.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("  ↓ %s", dest.name)

    with client.stream("GET", url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=65536):
                f.write(chunk)

    size_kb = dest.stat().st_size / 1024
    logger.info("    ✓ %.1f KB", size_kb)
    return True


# ---------------------------------------------------------------------------
# CKAN API
# ---------------------------------------------------------------------------


def _fetch_resources(slug: str, client: httpx.Client) -> list[dict]:
    """Retorna a lista de recursos do dataset via API CKAN."""
    url = f"{CKAN_BASE}/package_show?id={slug}"
    resp = client.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"CKAN API retornou erro para {slug}: {data}")
    return data["result"]["resources"]


# ---------------------------------------------------------------------------
# Processamento de registros
# ---------------------------------------------------------------------------


def _parse_records(raw: list | dict) -> list[dict]:
    """Normaliza a resposta JSON — pode ser lista ou dict com lista."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("data", "results", "acordaos", "items"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
    return []


def _insert_batch(
    records: list[dict],
    conn: sqlite3.Connection,
    desde_yyyymmdd: str,
    existing: set[str],
) -> int:
    """Insere registros filtrados por data. Retorna quantidade inserida."""
    cur = conn.cursor()
    inserted = 0

    for row in records:
        data_decisao = _val(row.get("dataDecisao")) or ""

        # Filtra por data mínima (comparação lexicográfica funciona para YYYYMMDD)
        if data_decisao and data_decisao < desde_yyyymmdd:
            continue

        # Monta número do processo
        sigla = _val(row.get("siglaClasse")) or ""
        num = _val(row.get("numeroProcesso")) or ""
        numero_processo = f"{sigla} {num}".strip()
        if not numero_processo:
            continue

        # Deduplica
        if numero_processo in existing:
            continue

        ementa = _build_ementa(row)
        if not ementa:
            continue

        decisao_text = _val(row.get("decisao"))
        if decisao_text and len(decisao_text) > 3000:
            decisao_text = decisao_text[:3000]

        repercussao = 1 if _val(row.get("tema")) else 0

        try:
            cur.execute(
                """
                INSERT INTO jurisprudencia
                    (tribunal, numero_processo, ementa, decisao,
                     data_julgamento, orgao_julgador, repercussao_geral)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "STJ",
                    numero_processo,
                    ementa,
                    decisao_text,
                    _convert_date(data_decisao),
                    _val(row.get("nomeOrgaoJulgador")),
                    repercussao,
                ),
            )
            existing.add(numero_processo)
            inserted += 1
        except sqlite3.Error as exc:
            logger.debug("Erro ao inserir %s: %s", numero_processo, exc)

    return inserted


def _load_json_bytes(
    data: bytes,
    conn: sqlite3.Connection,
    desde_yyyymmdd: str,
    existing: set[str],
) -> int:
    """Parseia bytes JSON e insere registros. Retorna quantidade inserida."""
    try:
        raw = json.loads(data)
    except json.JSONDecodeError as exc:
        logger.warning("JSON inválido: %s", exc)
        return 0
    records = _parse_records(raw)
    return _insert_batch(records, conn, desde_yyyymmdd, existing)


def _load_file(
    path: Path,
    conn: sqlite3.Connection,
    desde_yyyymmdd: str,
    existing: set[str],
) -> int:
    """Carrega arquivo JSON ou ZIP. Retorna quantidade inserida."""
    if path.suffix.lower() == ".zip":
        total = 0
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".json"):
                    total += _load_json_bytes(
                        zf.read(name), conn, desde_yyyymmdd, existing
                    )
        return total
    else:
        return _load_json_bytes(path.read_bytes(), conn, desde_yyyymmdd, existing)


# ---------------------------------------------------------------------------
# Entrypoint principal
# ---------------------------------------------------------------------------


def load(
    desde: str = "2022",
    force: bool = False,
    dry_run: bool = False,
    db_path: str = DEFAULT_DB,
    include_zip: bool = False,
) -> int:
    """
    Baixa e carrega acórdãos STJ no banco.

    Args:
        desde:      Ano mínimo de julgamento (ex: "2022", "2023").
        force:      Remove registros STJ existentes antes de carregar.
        dry_run:    Apenas lista arquivos disponíveis, não baixa nem insere.
        db_path:    Caminho para o banco SQLite.
        include_zip: Também baixa o arquivo ZIP com histórico completo.

    Returns:
        Total de registros inseridos.
    """
    desde_yyyymmdd = f"{desde}0101"

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA cache_size=-65536;")

    if force:
        stj_count = conn.execute(
            "SELECT COUNT(*) FROM jurisprudencia WHERE tribunal='STJ'"
        ).fetchone()[0]
        if stj_count > 0:
            conn.execute("DELETE FROM jurisprudencia WHERE tribunal='STJ'")
            conn.commit()
            logger.info("Removidos %d registros STJ existentes.", stj_count)

    # Conjunto para deduplicação em memória
    existing: set[str] = {
        row[0]
        for row in conn.execute(
            "SELECT numero_processo FROM jurisprudencia WHERE tribunal='STJ'"
        )
    }
    logger.info("STJ já no banco: %d registros.", len(existing))

    total_inserted = 0

    with httpx.Client() as client:
        for slug in DATASETS:
            orgao = slug.replace("espelhos-de-acordaos-", "").replace("-", " ").title()
            logger.info("\n[%s]", orgao)

            try:
                resources = _fetch_resources(slug, client)
            except Exception as exc:
                logger.error("  Erro ao buscar recursos de %s: %s", slug, exc)
                continue

            # Filtra recursos por formato
            zips = [r for r in resources if r.get("format", "").upper() == "ZIP"]
            jsons = sorted(
                [r for r in resources if r.get("format", "").upper() == "JSON"],
                key=lambda r: r.get("name", ""),
            )

            if dry_run:
                logger.info(
                    "  %d ZIP + %d JSON disponíveis (total %d recursos)",
                    len(zips),
                    len(jsons),
                    len(resources),
                )
                if jsons:
                    logger.info("  Primeiro: %s | Último: %s",
                                jsons[0]["name"], jsons[-1]["name"])
                continue

            to_download: list[tuple[str, Path]] = []

            if include_zip:
                for r in zips:
                    fname = r.get("name") or Path(r["url"]).name
                    dest = CACHE_DIR / slug / fname
                    to_download.append((r["url"], dest))

            for r in jsons:
                fname = r.get("name") or Path(r["url"]).name
                dest = CACHE_DIR / slug / fname
                to_download.append((r["url"], dest))

            dataset_inserted = 0
            for url, dest in to_download:
                try:
                    _download(url, dest, client)
                except httpx.HTTPError as exc:
                    logger.warning("  Erro ao baixar %s: %s", dest.name, exc)
                    continue

                n = _load_file(dest, conn, desde_yyyymmdd, existing)
                if n:
                    logger.info("  %s → %d registros", dest.name, n)
                    dataset_inserted += n

            if dataset_inserted:
                conn.commit()
                logger.info("  [%s] subtotal: %d inseridos", orgao, dataset_inserted)
                total_inserted += dataset_inserted

    if not dry_run and total_inserted > 0:
        logger.info("\nReconstruindo índice FTS5...")
        conn.execute("INSERT INTO jurisprudencia_fts(jurisprudencia_fts) VALUES('rebuild')")
        conn.commit()
        logger.info("FTS5 reconstruído com sucesso.")

    conn.close()
    logger.info("\nTotal STJ inseridos: %d", total_inserted)
    return total_inserted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Carrega acórdãos STJ do Portal de Dados Abertos no SQLite."
    )
    p.add_argument(
        "--desde",
        default="2022",
        metavar="AAAA",
        help="Ano mínimo de julgamento (padrão: 2022)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Remove registros STJ existentes antes de carregar",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas lista arquivos disponíveis, não baixa nem insere",
    )
    p.add_argument(
        "--include-zip",
        action="store_true",
        help="Também baixa o ZIP com histórico completo (pré mai/2022)",
    )
    p.add_argument(
        "--db",
        default=DEFAULT_DB,
        metavar="CAMINHO",
        help=f"Caminho do banco SQLite (padrão: {DEFAULT_DB})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    load(
        desde=args.desde,
        force=args.force,
        dry_run=args.dry_run,
        db_path=args.db,
        include_zip=args.include_zip,
    )
