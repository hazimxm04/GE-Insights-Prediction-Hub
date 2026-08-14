import sys
import logging
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def task_scrape_news() -> dict:
    """
    Task 1: Scrape latest news from all 5 sources.
    Returns metadata about what was scraped.
    """
    logger.info("Starting news scrape...")

    try:
        from sentiment.scrapers.news_scraper import scrape_all_news
        df = scrape_all_news()

        if df.empty:
            logger.warning("No articles scraped")
            return {'status': 'warning', 'articles': 0}

        save_path = ROOT / "data/raw/news" / f"news_{datetime.now().strftime('%Y%m%d')}.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

        logger.info(f"Scraped {len(df)} articles -> {save_path}")
        return {'status': 'success', 'articles': len(df), 'path': str(save_path)}

    except Exception as e:
        logger.error(f"Scrape failed: {e}")
        return {'status': 'error', 'error': str(e)}


def task_score_sentiment() -> dict:
    logger.info("Starting sentiment scoring...")

    try:
        import glob
        import pandas as pd
        import subprocess

        news_files = sorted(glob.glob(
            str(ROOT / "data/raw/news/news_*.csv")
        ))
        if not news_files:
            return {'status': 'error', 'error': 'No news files found'}

        latest = news_files[-1]
        df = pd.read_csv(latest)
        logger.info(f"Found {len(df)} articles from {latest}")

        # Add state mention columns if missing
        if 'mentions_johor' not in df.columns:
            from sentiment.scrapers.news_scraper import scrape_and_tag_states
            STATE_KEYWORDS = {
                'johor':        ['johor', 'jb', 'iskandar', 'johor bahru'],
                'neg_sembilan': ['negeri sembilan', 'seremban', 'nilai', 'ns'],
                'melaka':       ['melaka', 'malacca'],
            }
            for state, keywords in STATE_KEYWORDS.items():
                col = f'mentions_{state}'
                df[col] = df['title'].str.lower().str.contains(
                    '|'.join(keywords), na=False
                ) | df['description'].str.lower().str.contains(
                    '|'.join(keywords), na=False
                )
            df.to_csv(latest, index=False)
            logger.info("Added state mention columns")

        # Run scorer
        result = subprocess.run(
            [sys.executable,
             str(ROOT / "sentiment/scoring/sentiment_scorer.py")],
            capture_output=True, text=True,
            cwd=str(ROOT), timeout=600
        )

        scored_path = ROOT / "data/processed/scored_articles.csv"
        if scored_path.exists():
            df_scored = pd.read_csv(scored_path)
            logger.info(f"Scored {len(df_scored)} articles")
            return {'status': 'success', 'scored': len(df_scored)}
        else:
            logger.error(f"Scorer error: {result.stderr[:300]}")
            return {'status': 'error', 'error': result.stderr[:300]}

    except subprocess.TimeoutExpired:
        scored_path = ROOT / "data/processed/scored_articles.csv"
        if scored_path.exists():
            import pandas as pd
            df = pd.read_csv(scored_path)
            logger.warning(f"Timeout but {len(df)} articles scored")
            return {'status': 'success', 'scored': len(df)}
        return {'status': 'error', 'error': 'Timeout with no output'}

    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        return {'status': 'error', 'error': str(e)}
    
def task_update_state_scores() -> dict:
    """
    Task 3: Aggregate article scores into state-level scores.
    Writes state_sentiment_scores.csv.
    """
    logger.info("Updating state sentiment scores...")

    try:
        import pandas as pd
        from sentiment.scoring.sentiment_scorer import (
            compute_national_narratives,
            compute_state_scores
        )

        scored_path = ROOT / "data/processed/scored_articles.csv"
        if not scored_path.exists():
            return {'status': 'error', 'error': 'No scored articles found'}

        df_scored = pd.read_csv(scored_path)
        narratives = compute_national_narratives(df_scored)
        state_scores = compute_state_scores(df_scored, narratives)

        state_path = ROOT / "data/processed/state_sentiment_scores.csv"
        state_scores.to_csv(state_path, index=False)

        narrative_path = ROOT / "data/processed/national_narrative_scores.csv"
        pd.DataFrame([narratives]).to_csv(narrative_path, index=False)

        logger.info(f"Updated scores for {len(state_scores)} states")
        return {'status': 'success', 'states': len(state_scores)}

    except Exception as e:
        logger.error(f"State score update failed: {e}")
        return {'status': 'error', 'error': str(e)}


def task_rebuild_rag() -> dict:
    """
    Task 4: Rebuild ChromaDB knowledge base with latest data.
    Ensures chatbot answers reflect today's sentiment scores.
    """
    logger.info("Rebuilding RAG knowledge base...")

    try:
        from chatbot.knowledge_base.builder import build_knowledge_base
        build_knowledge_base()

        logger.info("RAG knowledge base rebuilt successfully")
        return {'status': 'success'}

    except Exception as e:
        logger.error(f"RAG rebuild failed: {e}")
        return {'status': 'error', 'error': str(e)}


def run_dag():
    """
    Run all tasks in sequence.
    This is what APScheduler calls on schedule.
    In Airflow, each task would be a separate operator.
    """
    dag_name = "dag_01_daily_sentiment"
    start_time = datetime.now()
    logger.info(f"{'='*50}")
    logger.info(f"DAG START: {dag_name} at {start_time}")
    logger.info(f"{'='*50}")

    tasks = [
        ("scrape_news",        task_scrape_news),
        ("score_sentiment",    task_score_sentiment),
        ("update_state_scores", task_update_state_scores),
        ("rebuild_rag",        task_rebuild_rag),
    ]

    results = {}
    for task_name, task_fn in tasks:
        logger.info(f"\n[TASK] {task_name}")
        result = task_fn()
        results[task_name] = result

        if result.get('status') == 'error':
            logger.error(f"[TASK FAILED] {task_name}: {result.get('error')}")
            logger.error(f"DAG STOPPED — downstream tasks skipped")
            break
        else:
            logger.info(f"[TASK OK] {task_name}: {result}")

    duration = (datetime.now() - start_time).seconds
    logger.info(f"\nDAG COMPLETE: {dag_name} in {duration}s")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_dag()
    