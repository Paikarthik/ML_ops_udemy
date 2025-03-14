import traceback
import sys


class CustomException(Exception):

    def __init__(self, error_message, error_detail: sys):
        super.__init__(error_message)
        self.error_message = self.get_detailed_error_message(
            error_message, error_detail
        )

    @staticmethod
    def get_detailed_error_message(error_message, error_detail):
        """
        Returns detailed error message string
        """

        _, _, error_traceback = error_detail.exc.info()
        file_name = error_traceback.tb_frame.f_code.co_filename
        line_number = error_traceback.tb_lineno

        return f"Error in {file_name}, line {line_number}: {error_message}"

    def __str__(self):
        return self.error_message
