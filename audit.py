from backend.core.pipelines.state_pipeline import StateElectionPipeline

pipeline = StateElectionPipeline('johor')
sentiment = pipeline.load_sentiment_features()
economic = pipeline.load_economic_features()

print("Sentiment keys:", sentiment.keys())
print("Sentiment values sample:", {k: v for k, v in list(sentiment.items())[:3]})
print("\nEconomic pressure:", economic)