
from itertools import chain

import numpy as np

from pgmpy.estimators import ParameterEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator


class SmoothedMaximumLikelihoodEstimator(MaximumLikelihoodEstimator):
    def estimate_cpd(self, node):
        """
            Method to estimate the CPD for a given variable.

            Parameters
            ----------
            node: int, string (any hashable python object)
                The name of the variable for which the CPD is to be estimated.

            Returns
            -------
            CPD: TabularCPD

            Examples
            --------
            >>> import pandas as pd
            >>> from pgmpy.models import BayesianModel
            >>> from pgmpy.estimators import MaximumLikelihoodEstimator
            >>> data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
            >>> model = BayesianModel([('A', 'C'), ('B', 'C')])
            >>> cpd_A = MaximumLikelihoodEstimator(model, data).estimate_cpd('A')
            >>> print(cpd_A)
            ╒══════╤══════════╕
            │ A(0) │ 0.666667 │
            ├──────┼──────────┤
            │ A(1) │ 0.333333 │
            ╘══════╧══════════╛
            >>> cpd_C = MaximumLikelihoodEstimator(model, data).estimate_cpd('C')
            >>> print(cpd_C)
            ╒══════╤══════╤══════╤══════╤══════╕
            │ A    │ A(0) │ A(0) │ A(1) │ A(1) │
            ├──────┼──────┼──────┼──────┼──────┤
            │ B    │ B(0) │ B(1) │ B(0) │ B(1) │
            ├──────┼──────┼──────┼──────┼──────┤
            │ C(0) │ 0.0  │ 0.0  │ 1.0  │ 0.5  │
            ├──────┼──────┼──────┼──────┼──────┤
            │ C(1) │ 1.0  │ 1.0  │ 0.0  │ 0.5  │
            ╘══════╧══════╧══════╧══════╧══════╛
        """

        state_counts = self.state_counts(node)

        # if a column contains only `0`s (no states observed for some configuration
        # of parents' states) fill that column uniformly instead
        state_counts.loc[:, (state_counts == 0).all()] = 1

        parents = sorted(self.model.get_parents(node))
        parents_cardinalities = [len(self.state_names[parent])
                                for parent in parents]
        node_cardinality = len(self.state_names[node])

        # Get the state names for the CPD
        state_names = {node: list(state_counts.index)}
        if parents:
                state_names.update(
                    {
                        state_counts.columns.names[i]: list(
                            state_counts.columns.levels[i])
                        for i in range(len(parents))
                    }
                )

        cpd = TabularCPD(
            node,
            node_cardinality,
            np.array(state_counts) + 1,
            evidence=parents,
            evidence_card=parents_cardinalities,
            state_names={var: self.state_names[var]
                        for var in chain([node], parents)},
        )
        cpd.normalize()
        return cpd
