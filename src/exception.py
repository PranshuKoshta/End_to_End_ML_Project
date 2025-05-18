import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    
    # This function takes an error and its details and returns a formatted string with the error message.
    
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{str(error)}]"
    return error_message

class CustomException(Exception):
    # This class inherits from the built-in Exception class and is used to raise custom exceptions with detailed error messages.
    
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
# The CustomException class is used to handle exceptions in a more informative way by providing the file name, line number, and error message.
# This can be useful for debugging and logging purposes.
# The error_message_detail function formats the error message with the file name, line number, and error message.
# This can help developers quickly identify where the error occurred and what the error was.        


if __name__ == '__main__':
    try:
        a= 1/0
    except Exception as e:
        logging.info("Divide by zero")
        raise CustomException(e,sys)