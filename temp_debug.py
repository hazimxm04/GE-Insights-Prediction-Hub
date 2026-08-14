import sys
sys.path.insert(0, '.')
from backend.core.pipelines.state_pipeline import StateElectionPipeline

pipeline = StateElectionPipeline('neg_sembilan')
df = pipeline.engineer_features(2023, 2026)

print('coalition_b unique:', df['winner_coalition_b'].unique().tolist())
print('coalition_a unique:', df['winner_coalition_a'].unique().tolist())

bn = df[df['winner_coalition_b'] == 'BN']
print('BN seats 2026:', len(bn))
print('BN with target=0:', (bn['target_non_bn_won']==0).sum())
print('BN with target=1:', (bn['target_non_bn_won']==1).sum())

print()
print(df[['seat','winner_coalition_b','target_non_bn_won']].to_string())