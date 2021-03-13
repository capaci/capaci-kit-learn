class BaseException(Exception):
    default_message: str = ''

    def __init__(self, message: str = ''):
        if not message:
            message = self.default_message

        super().__init__(message)


class InvalidMetric(BaseException):
    default_message: str = 'Invalid metric passed as argument'
