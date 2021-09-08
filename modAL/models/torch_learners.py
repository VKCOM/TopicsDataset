from typing import Any

from modAL.utils.data import modALinput

from modAL import ActiveLearner


class TorchTopicsActiveLearner(ActiveLearner):

    def _fit_to_known(self, bootstrap: bool = False, **fit_kwargs) -> 'BaseLearner':
        pass

    def _fit_on_new(self, X: modALinput, y: modALinput, bootstrap: bool = False, **fit_kwargs) -> 'BaseLearner':
        pass

    def score(self, X: modALinput, y: modALinput, **score_kwargs) -> Any:
        pass
        # return self.estimator.evaluate(X, y, verbose=0, **score_kwargs)[1]
