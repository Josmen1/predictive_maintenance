import sys

class PredictiveMaintenanceException(Exception):
    """
    Custom exception class for the Predictive Maintenance project.
    Captures the message, filename, and line number where the exception occurred.
    """

    def __init__(self, message: str, error_details: sys):
        """
        Args:
            message (str): Human-readable error message.
            error_details (sys): Typically pass 'sys' so we can extract traceback info.
        """
        # Store the custom message
        self.message = message

        # Extract traceback details: type, value, traceback
        _, _, exc_tb = error_details.exc_info()

        # Extract the exact line number where the exception occurred
        self.line_number = exc_tb.tb_lineno

        # Extract the file name where the exception occurred
        self.file_name = exc_tb.tb_frame.f_code.co_filename

        # Initialize the base Exception with the message
        super().__init__(self.message)

    def __str__(self) -> str:
        """
        Returns a clear, formatted error message for logs or console output.
        """
        return (
            f"Error occurred in script: {self.file_name} "
            f"at line number: {self.line_number} "
            f"with message: {self.message}"
        )
