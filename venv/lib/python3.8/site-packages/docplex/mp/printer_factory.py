# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

from docplex.mp.lp_printer import LPModelPrinter
from docplex.mp.ppretty import ModelPrettyPrinter


class ModelPrinterFactory(object):
    __printer_ext_map = \
        {printer().get_format(): printer for printer in {LPModelPrinter}}
    default_printer_type = LPModelPrinter

    @staticmethod
    def new_printer(exchange_format, **kwargs):
        """
        returns a new printer
        :param exchange_format:
        :return:
        """
        printer_type = ModelPrinterFactory.__printer_ext_map.get(exchange_format)

        if not printer_type:
            return None
        else:
            printer = printer_type(**kwargs)
            return printer

    @staticmethod
    def new_pretty_printer():
        return ModelPrettyPrinter()
