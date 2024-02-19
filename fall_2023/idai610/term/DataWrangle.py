#!/usr/bin/env python
import pandas as pd
from dateutil.parser import parse


class DataWrangle(object):

    def __init__(self, dataset):
        df = pd.read_csv(dataset)

    def data_inspect(self):
        """
        Output dataset's basic structure.
        """
        pass

    def missing_values(self):
        """
        Identifies and deals with missing data.
        """
        pass

    def data_cleaning(self):
        """
        Remove irrelevant data and correct for inconsistencies,
        standardize formats of strings and dates, check for duplicates, etc.
        """
        pass

    def data_transformation(self):
        """
        Convert categorical data to a numerical format and normalize numerical
        data.
        """
        pass

    def outliers(self):
        """
        Detect and deal with outliers.
        """
        pass

    def feature_engineer(self):
        """
        Create new features that might be relevant for analysis.
        """
        pass

    def date_transformation(self, date_column):
        """
        Converting data columns to actual date datatype.
        -----------------------------------------------
        INPUT:
            Date: (pandas series: str)

        OUTPUT:
            parsed_series: (pandas datetime series: default = str) Shoujld
            return pandas series of datetime datatypes unless proper format not
        found then "Format not found" is returned as a string.
        """
        date_formats = [
            "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
            "%m-%d-%Y", "%d-%m-%Y", "%Y-%m-%d",
            "%m.%d.%Y", "%d.%m.%Y", "%Y.%m.%d",
            "%m/%d/%y", "%d/%m/%y", "%y/%m/%d",
            "%m-%d-%y", "%d-%m-%y", "%y-%m-%d",
            "%m.%d.%y", "%d.%m.%y", "%y.%m.%d"
        ]

        for formats in date_formats:
            try:
                converted_series = pd.to_datetime(df[date_column], format=formats)
                return converted_series

            except ValueError:
                # If there's an error try next format
                continue

        return "Format not found"


data = DataWrangle()
