"""Tests for chatbot/docs_search.py: the written explanations the agent can search."""
import re

from chatbot import docs_search as D


def top(query, k=1):
    return D.search_docs(query, k)['results']


def test_every_doc_loads_into_sections():
    chunks = D.load_chunks()
    assert {c['source'] for c in chunks} >= {'overview.md', 'model.md', 'fallback_and_confidence.md',
                                              'validation_and_baselines.md', 'sentiment_layer.md',
                                              'limitations.md', 'glossary.md'}
    assert all(c['text'].strip() for c in chunks)


def test_fallback_question_finds_the_fallback_passage():
    assert top('what is a fallback seat')[0]['source'] == 'fallback_and_confidence.md'


def test_confidence_question_finds_the_no_confidence_passage():
    r = top('why is there no confidence level')[0]
    assert r['source'] == 'fallback_and_confidence.md' and 'No confidence level' in r['section']


def test_sentiment_questions_find_the_sentiment_doc():
    assert top('how is the sentiment used')[0]['source'] == 'sentiment_layer.md'
    assert top('are the sentiment effects fitted to data')[0]['source'] == 'sentiment_layer.md'


def test_baseline_question_finds_validation_doc():
    assert top('what are the baselines')[0]['source'] == 'validation_and_baselines.md'


def test_forecast_versus_validation_is_explained():
    assert top('difference between forecast and validation')[0]['source'] == 'overview.md'


def test_nonsense_and_empty_queries_do_not_crash():
    assert D.search_docs('')['results'] == []
    assert D.search_docs(None)['results'] == []
    assert D.search_docs('zzzqqq')['results'] == []
    assert D.search_docs('model', k='bad')['results']


def test_k_is_limited_and_passages_are_cut():
    assert len(D.search_docs('model', k=99)['results']) <= 5
    assert all(len(r['text']) <= D.MAX_CHARS for r in D.search_docs('model', 5)['results'])


def test_docs_hold_no_state_figures():
    """Numbers must come from the tools. A figure in a document goes stale; this keeps them out."""
    bad = [c['source'] for c in D.load_chunks()
           if re.search(r'\b(48|56|31|36)\s*(/|of)\s*\d+|\d+(\.\d+)?%', c['text'])]
    assert bad == [], f'figures found in {bad}'


def test_schema_is_valid_and_agent_registers_it():
    from chatbot import agent
    assert D.SEARCH_DOCS_SCHEMA['function']['name'] == 'search_docs'
    schemas, funcs = agent._registry()
    assert 'search_docs' in funcs and any(s['function']['name'] == 'search_docs' for s in schemas)
