"""
ETL: Download em lote de acórdãos do STJ via SCON usando Playwright.

IMPORTANTE: Execute este script na sua MÁQUINA LOCAL (não no servidor).
O IP residencial passa o desafio Cloudflare automaticamente.

Instalação (uma vez):
    pip install playwright pandas
    python -m playwright install chromium

Uso:
    # Baixar acórdãos de 2023
    python etl/baixar_stj.py --ano 2023

    # Baixar intervalo específico
    python etl/baixar_stj.py --inicio 01/01/2022 --fim 31/12/2022

    # Baixar múltiplos anos
    python etl/baixar_stj.py --ano 2022
    python etl/baixar_stj.py --ano 2023
    python etl/baixar_stj.py --ano 2024

Saída: data/raw/stj_acordaos_AAAA.csv  (mesmas colunas dos CSVs do STF)

O script abre uma janela Chrome visível. Se aparecer "Verificação automática
em andamento", aguarde — o Cloudflare resolve sozinho em alguns segundos.
Após isso, o download é totalmente automático.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SCON_URL     = "https://scon.stj.jus.br/SCON/"
PESQUISA_URL = "https://scon.stj.jus.br/SCON/pesquisar.jsp"
TOC_URL      = "https://scon.stj.jus.br/SCON/jurisprudencia/toc.jsp"

# Colunas de saída — idênticas aos CSVs do STF para reaproveitamento do ETL
COLUNAS = ["Titulo", "Relator", "Data de publicação", "Data de julgamento",
           "Órgão julgador", "Ementa"]

# Seletores CSS derivados dos XPaths do pacote R {stj} (jjesusfilho/stj)
SEL_RESULTADO  = "div.paragrafoBRS"
SEL_DOC_TITULO = "div.docTitulo"
SEL_DOC_TEXTO  = "div.docTexto"

# Aguardar até 15 s para cada operação de página
PAGE_TIMEOUT = 15_000


# ---------------------------------------------------------------------------
# Helpers de extração de HTML
# ---------------------------------------------------------------------------

def _campo(bloco, rotulo: str) -> str:
    """Extrai texto do campo com o rótulo dado dentro de um resultado."""
    try:
        titulo_el = bloco.query_selector(f"xpath=.//div[@class='docTitulo'][normalize-space()='{rotulo}']")
        if not titulo_el:
            # Tenta variantes com acento/sem acento
            for alt in (rotulo.replace("ó", "o"), rotulo + "a"):
                titulo_el = bloco.query_selector(f"xpath=.//div[@class='docTitulo'][normalize-space()='{alt}']")
                if titulo_el:
                    break
        if not titulo_el:
            return ""
        texto_el = titulo_el.evaluate_handle(
            "el => el.nextElementSibling"
        ).as_element()
        return (texto_el.inner_text() or "").strip() if texto_el else ""
    except Exception:
        return ""


def _extrair_resultado(bloco) -> dict | None:
    """Extrai os campos de um bloco de resultado do SCON."""
    try:
        processo  = _campo(bloco, "Processo")
        if not processo:
            # fallback: pegar texto da âncora do processo
            a = bloco.query_selector("a.docTitulo, a[href*='SCON']")
            processo = (a.inner_text() or "").strip() if a else ""

        relator       = _campo(bloco, "Relator") or _campo(bloco, "Relatora")
        dt_publicacao = _campo(bloco, "Data da Publicação") or _campo(bloco, "Data de Publicação")
        dt_julgamento = _campo(bloco, "Data do Julgamento")
        orgao         = _campo(bloco, "Órgão Julgador")
        ementa        = _campo(bloco, "Ementa")

        if not ementa:
            return None  # sem ementa → não interessa

        return {
            "Titulo"             : processo,
            "Relator"            : relator,
            "Data de publicação" : dt_publicacao,
            "Data de julgamento" : dt_julgamento,
            "Órgão julgador"     : orgao,
            "Ementa"             : ementa,
        }
    except Exception as exc:
        print(f"  [aviso] Erro ao extrair resultado: {exc}")
        return None


# ---------------------------------------------------------------------------
# Core do scraper
# ---------------------------------------------------------------------------

def _aguardar_cloudflare(page, tentativas: int = 10) -> bool:
    """
    Aguarda o Cloudflare resolver o desafio.
    Retorna True se passou, False se esgotou tentativas.
    """
    for i in range(tentativas):
        titulo = page.title().lower()
        url    = page.url
        if "moment" not in titulo and "verificação" not in titulo and "scon" in url.lower():
            return True
        print(f"  [cloudflare] Aguardando desafio... ({i+1}/{tentativas})")
        time.sleep(3)
    return False


def _pagina_tem_resultados(page) -> bool:
    """Verifica se a página atual tem resultados do SCON."""
    return bool(page.query_selector_all(SEL_RESULTADO))


def _total_resultados(page) -> int:
    """Extrai o número total de resultados da página de busca."""
    for sel in ("span.numDocs", "span#numDocs", "span[class*='numDoc']",
                "td.numDocs", "div.numDocs", "#numero-de-documentos"):
        try:
            el = page.query_selector(sel)
            if el:
                texto = el.inner_text().replace(".", "").strip()
                if texto.isdigit():
                    return int(texto)
        except Exception:
            pass
    # Fallback: contar blocos na página × estimar páginas restantes
    blocos = len(page.query_selector_all(SEL_RESULTADO))
    return blocos if blocos > 0 else 0


def _aguardar_resultados(page, timeout_s: int = 20) -> bool:
    """Aguarda até que a página mostre resultados (paragrafoBRS) ou timeout."""
    for _ in range(timeout_s):
        if _pagina_tem_resultados(page):
            return True
        time.sleep(1)
    return False


def _screenshot(page, nome: str) -> None:
    """Salva screenshot para debug."""
    try:
        path = Path(f"/tmp/stj_debug_{nome}.png")
        page.screenshot(path=str(path))
        print(f"  [debug] Screenshot: {path}")
    except Exception:
        pass


def baixar_acordaos(
    data_inicio: str,
    data_fim: str,
    arquivo_saida: Path,
    headless: bool = False,
) -> int:
    """
    Baixa acórdãos do SCON no intervalo de datas especificado.

    Args:
        data_inicio: "DD/MM/AAAA"
        data_fim:    "DD/MM/AAAA"
        arquivo_saida: caminho do CSV de saída
        headless: False = Chrome visível (recomendado para Cloudflare)

    Returns:
        Número de registros baixados.
    """
    arquivo_saida.parent.mkdir(parents=True, exist_ok=True)
    registros: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="pt-BR",
            viewport={"width": 1280, "height": 900},
        )
        ctx.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        page = ctx.new_page()

        # ------------------------------------------------------------------
        # 1. Acesso inicial → passar Cloudflare
        # ------------------------------------------------------------------
        print(f"\n[1/4] Acessando SCON ({SCON_URL})...")
        try:
            page.goto(SCON_URL, timeout=30_000, wait_until="domcontentloaded")
        except PWTimeout:
            print("  [timeout] Tentando novamente...")
            page.goto(SCON_URL, timeout=30_000, wait_until="commit")

        if not _aguardar_cloudflare(page):
            print("\n❌ Cloudflare não resolvido. Execute na sua máquina LOCAL.")
            browser.close()
            return 0

        print("  ✅ Cloudflare resolvido.")
        time.sleep(2)

        # ------------------------------------------------------------------
        # 2. Busca: tenta automático, cai para interativo se falhar
        # ------------------------------------------------------------------
        print(f"\n[2/4] Buscando acórdãos de {data_inicio} a {data_fim}...")

        encontrou = False

        # Tentativa 1: URL direta do toc.jsp com data URL-encoded (DD%2FMM%2FAAAA)
        di_enc = data_inicio.replace("/", "%2F")
        df_enc = data_fim.replace("/", "%2F")
        for url_tentativa in [
            # Formato 1: dtpb/dtde com barras URL-encoded
            f"{TOC_URL}?b=ACOR&livre=&p=true&l=10&i=1&operador=e&tp=T&dtpb={di_enc}&dtde={df_enc}",
            # Formato 2: pesquisar.jsp com barras
            f"{PESQUISA_URL}?b=ACOR&p=true&tp=T&dtpb={di_enc}&dtde={df_enc}&l=10&i=1",
            # Formato 3: campo livre com sintaxe SCON (@DTPB)
            f"{TOC_URL}?b=ACOR&livre=%40DTPB+%3E%3D+{di_enc}+E+%40DTPB+%3C%3D+{df_enc}&p=true&l=10&i=1",
        ]:
            try:
                print(f"  Tentando: {url_tentativa[:90]}...")
                page.goto(url_tentativa, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")
                _aguardar_cloudflare(page, tentativas=5)
                time.sleep(2)
                if _pagina_tem_resultados(page):
                    encontrou = True
                    print("  ✅ Resultados encontrados!")
                    break
            except Exception as exc:
                print(f"  [aviso] {exc}")

        # Tentativa 2: preencher o formulário visualmente
        if not encontrou:
            try:
                print("  Tentando preencher formulário...")
                page.goto(f"{PESQUISA_URL}?b=ACOR", timeout=PAGE_TIMEOUT,
                          wait_until="domcontentloaded")
                _aguardar_cloudflare(page, tentativas=5)
                time.sleep(1)

                # Preencher campos de data com formato DD/MM/AAAA
                for name, val in [("dtpb", data_inicio), ("dtde", data_fim)]:
                    el = page.query_selector(f"input[name='{name}']")
                    if el:
                        el.triple_click()
                        el.type(val)

                # Campo livre vazio
                el = page.query_selector("input[name='livre'], textarea[name='livre']")
                if el:
                    el.fill("")

                # Submeter
                submit = page.query_selector(
                    "input[type='submit'][value*='esquis'], "
                    "input[type='submit'][value*='Esquis'], "
                    "button[type='submit'], input[type='submit']"
                )
                if submit:
                    submit.click()
                    page.wait_for_load_state("domcontentloaded", timeout=PAGE_TIMEOUT)
                    _aguardar_cloudflare(page, tentativas=5)
                    time.sleep(2)
                    if _pagina_tem_resultados(page):
                        encontrou = True
                        print("  ✅ Resultados encontrados via formulário!")
            except Exception as exc:
                print(f"  [aviso] Formulário: {exc}")

        # Tentativa 3 (fallback interativo): o usuário faz a busca manualmente
        if not encontrou:
            _screenshot(page, "antes_busca_manual")
            print("\n" + "=" * 60)
            print("AÇÃO NECESSÁRIA — busca automática não funcionou.")
            print()
            print("Na janela do Chrome aberta:")
            print(f"  1. Certifique-se de estar em 'Pesquisa de Jurisprudência'")
            print(f"  2. Selecione a base: ACÓRDÃOS")
            print(f"  3. Deixe o campo de texto livre VAZIO")
            print(f"  4. Preencha 'Data de Publicação':")
            print(f"     - De: {data_inicio}")
            print(f"     - Até: {data_fim}")
            print(f"  5. Clique em PESQUISAR")
            print(f"  6. Aguarde os resultados carregarem na página")
            print("=" * 60)
            print("\nPressione ENTER aqui quando os resultados aparecerem no Chrome...")
            input()
            _aguardar_cloudflare(page, tentativas=5)
            if _aguardar_resultados(page, timeout_s=15):
                encontrou = True
                print("  ✅ Resultados detectados!")
            else:
                _screenshot(page, "sem_resultados")
                print(f"\n❌ Nenhum resultado detectado.")
                print(f"   Screenshot salvo em /tmp/stj_debug_sem_resultados.png")
                print(f"   URL atual: {page.url}")
                browser.close()
                return 0

        # ------------------------------------------------------------------
        # 3. Paginar e extrair resultados
        # ------------------------------------------------------------------
        total = _total_resultados(page)
        print(f"\n[3/4] Total de resultados detectados: {total or '?'} "
              f"(continuando mesmo se impreciso...)")
        _screenshot(page, "pagina_resultados")

        import re as _re

        pagina     = 1
        idx        = 1        # parâmetro &i= da URL
        sem_novos  = 0        # contador de páginas consecutivas sem novos registros

        while True:
            blocos = page.query_selector_all(SEL_RESULTADO)
            novos = 0
            for bloco in blocos:
                rec = _extrair_resultado(bloco)
                if rec:
                    registros.append(rec)
                    novos += 1

            print(f"  Página {pagina:>4}: {novos:>3} ementas  "
                  f"(acumulado: {len(registros)})")

            if novos == 0:
                sem_novos += 1
                if sem_novos >= 2:
                    print("  Sem novos resultados em 2 páginas consecutivas — fim.")
                    break
            else:
                sem_novos = 0

            if len(registros) >= 10_000:
                print("  [aviso] 10.000 registros — divida o intervalo se precisar de mais.")
                break

            # ── Tentar ir para próxima página ──────────────────────────────
            # 1. Link "Próximo" na página
            proximo = page.query_selector(
                "a[title='Próximo'], a[title='Próxima'], "
                "a.next, a[rel='next'], "
                "a:has-text('Próximo'), a:has-text('›'), a:has-text('>>'), "
                "img[src*='proximo'], img[alt*='roximo']"
            )
            if proximo:
                try:
                    proximo.click()
                    page.wait_for_load_state("domcontentloaded", timeout=PAGE_TIMEOUT)
                    _aguardar_cloudflare(page, tentativas=5)
                    time.sleep(1.5)
                    pagina += 1
                    continue
                except Exception as exc:
                    print(f"  [aviso] Clique em Próximo falhou: {exc}")

            # 2. Manipular parâmetro &i= na URL
            idx += 10
            url_atual = page.url
            if "&i=" in url_atual:
                nova_url = _re.sub(r"[?&]i=\d+", lambda m: m.group().replace(
                    m.group(), ("&" if "&" in m.group() else "?") + f"i={idx}"
                ), url_atual)
            elif "?" in url_atual:
                nova_url = url_atual + f"&i={idx}"
            else:
                nova_url = url_atual + f"?i={idx}"

            try:
                page.goto(nova_url, timeout=PAGE_TIMEOUT, wait_until="domcontentloaded")
                _aguardar_cloudflare(page, tentativas=5)
                time.sleep(1.5)
                pagina += 1
            except Exception as exc:
                print(f"  [aviso] Paginação por URL falhou: {exc}")
                break

        browser.close()

    # ------------------------------------------------------------------
    # 4. Salvar CSV
    # ------------------------------------------------------------------
    if not registros:
        print("\n❌ Nenhuma ementa extraída.")
        return 0

    # Deduplicar por Titulo
    vistos: set[str] = set()
    unicos = []
    for r in registros:
        key = r["Titulo"]
        if key not in vistos:
            vistos.add(key)
            unicos.append(r)

    print(f"\n[4/4] Salvando {len(unicos)} acórdãos únicos em {arquivo_saida}...")
    with open(arquivo_saida, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUNAS)
        writer.writeheader()
        writer.writerows(unicos)

    print(f"✅ Concluído! {len(unicos)} acórdãos salvos em {arquivo_saida}")
    return len(unicos)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baixa acórdãos do STJ/SCON em lote via Playwright.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ano", type=int,
                   help="Ano completo a baixar (ex: 2023)")
    p.add_argument("--inicio", default=None,
                   help="Data inicial DD/MM/AAAA (alternativa a --ano)")
    p.add_argument("--fim", default=None,
                   help="Data final DD/MM/AAAA (alternativa a --ano)")
    p.add_argument("--saida", default=None,
                   help="Caminho do CSV de saída (padrão: data/raw/stj_AAAA.csv)")
    p.add_argument("--headless", action="store_true",
                   help="Rodar sem janela visível (não recomendado — falha no Cloudflare)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.ano:
        inicio = f"01/01/{args.ano}"
        fim    = f"31/12/{args.ano}"
        saida  = Path(args.saida or f"data/raw/stj_{args.ano}.csv")
    elif args.inicio and args.fim:
        inicio = args.inicio
        fim    = args.fim
        ano    = fim.split("/")[-1]
        saida  = Path(args.saida or f"data/raw/stj_{ano}.csv")
    else:
        print("Erro: informe --ano OU --inicio + --fim")
        sys.exit(1)

    n = baixar_acordaos(
        data_inicio=inicio,
        data_fim=fim,
        arquivo_saida=saida,
        headless=args.headless,
    )

    if n > 0:
        print(f"\nPróximo passo — carregue no banco:")
        print(f"  python -m etl.load")
        print(f"  python -m etl.generate_embeddings")


if __name__ == "__main__":
    main()
