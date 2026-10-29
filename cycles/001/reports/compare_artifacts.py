"""Paired example-bootstrap analysis of fixed fitted models, with no fitting path.

Global context: the frozen protocol defines the statistic. Shared indices retain
dependence between methods and seeds. src-bootstrap documents paired resampling;
explicit NumPy quantiles implement the registered percentile construction.
"""
import numpy as np


def paired_summary(treated, baseline, *, resamples, seed):
    """Validate alignment, then resample examples conditional on all fitted seeds."""
    assert len(treated) == len(baseline) and len(treated) > 1
    reference = treated[0]
    assert len({row['seed'] for row in treated}) == len(treated)
    differences = []
    for left, right in zip(treated, baseline):
        assert left['seed'] == right['seed']
        for row in (left, right):
            assert row['example_ids'] == reference['example_ids']
            assert row['y_true'] == reference['y_true']
            assert len(row['y_pred']) == len(reference['y_true'])
        truth = np.asarray(reference['y_true'])
        differences.append((np.asarray(left['y_pred']) == truth).astype(float)
                           - (np.asarray(right['y_pred']) == truth).astype(float))
    differences = np.asarray(differences)
    effects = differences.mean(axis=1)
    # Average seeds first: linearity makes this exactly the shared-index statistic.
    per_example = differences.mean(axis=0)
    indices = np.random.default_rng(seed).integers(
        0, len(per_example), size=(resamples, len(per_example)))
    distribution = per_example[indices].mean(axis=1)
    interval = np.quantile(distribution, [0.025, 0.975], method='linear')
    interpretation = ('direction_contradicted_conditionally' if interval[1] < 0 else
                      'positive_conditionally' if interval[0] > 0 else 'inconclusive')
    return {'analysis_kind': 'exploratory', 'seed_order': [row['seed'] for row in treated],
            'paired_seed_differences': effects.tolist(),
            'mean_difference': float(effects.mean()),
            'seed_difference_sample_sd': float(effects.std(ddof=1)),
            'conditional_interval': interval.tolist(), 'interpretation': interpretation,
            'bootstrap_seed': seed, 'bootstrap_resamples': resamples,
            'bootstrap_distribution_unique_values': np.unique(distribution).tolist(),
            'correctness_disagreements_by_seed': np.count_nonzero(differences, axis=1).tolist(),
            'interval_scope': 'Percentile test-example bootstrap conditional on the fitted seed set; no training-population uncertainty.'}
