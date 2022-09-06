import os
import sys

class sales_project_exception(Exception):
    
    def __init__(self,error_msg:Exception,error_detail:sys):
        super().__init__(error_msg)
        self.error_msg=sales_project_exception.get_detailed_error_message(error_msg=error_msg,error_detail=error_detail)
    
    @staticmethod
    def get_detailed_error_message(error_msg:Exception,error_detail:sys)->str:
        _,_,exec_tb=error_detail.exc_info()
        line_number=exec_tb.tb_lineno
        file_name=exec_tb.tb_frame.f_code.co_filename

        error_msg=f"Error occured in script : [{file_name}] at line number: [{line_number} error message: [{error_msg}]]"
        return error_msg
    def __str__(self):
        return self.error_msg
    def __repr__(self):
        return sales_project_exception.__name__.str()



