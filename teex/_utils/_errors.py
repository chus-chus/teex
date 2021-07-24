""" Custom error classes """


class MetricNotAvailableError(Exception):

    def __init__(self, metric: str, message=None):
        super(MetricNotAvailableError, self).__init__(f"Metric {metric} not available" if message is None else message)


class IncompatibleGTAndPredError(TypeError):

    def __init__(self, message: str = "Ground truth/s and prediction/s are not of the same type."):
        super(IncompatibleGTAndPredError, self).__init__(message)
