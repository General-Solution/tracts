from tracts import ParametrizedDemography
import tracts
import numpy


def test_founding_2pop():
    model = ParametrizedDemography()
    model.add_founder_event({'A': 'm1_A'}, 'B', 't0')
    model.finalize()
    migration_matrices = model.get_migration_matrices([0.4, 4.5])
    m = migration_matrices[0]
    # The old model used number_of_generations / 100
    m2 = tracts.legacy_models.models_2pop.pp([0.4, 0.045])
    assert numpy.allclose(m, m2)
