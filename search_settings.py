import os
import time
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, List


class PropertyModel(BaseModel):
    street_number:  str = Field(..., alias='Street #')
    street_name:    str = Field(..., alias='Street name')
    municipality:   str = Field(..., alias='Municipality')
    street_abbr:          Optional[str] = Field(None, alias='Street abbr')
    street_direction:     Optional[str] = Field(None, alias='Street direction')
    apt_unit:             Optional[str] = Field(None, alias='Apt/Unit')
    penthouse:            Optional[bool] = Field(None, alias='Penthouse')
    sub_penthouse:        Optional[bool] = Field(None, alias='Sub-penthouse')
    penthouse_collection: Optional[bool] = Field(None, alias='Penthouse collection')


# Path and sheet name for the Excel file
phl_path = os.path.join('data', 'PHL - List - TRREB.xlsx')
phl_sheet_name = 'List'


def safe_nan_check(value, expected_type):
    """Convert NaN values to None and ensure the value matches the expected type."""
    if isinstance(value, (float, np.float64)) and np.isnan(value):
        return None
    if expected_type == str:
        return str(value) if value not in (None, np.nan) else None
    if expected_type == bool:
        return bool(value) if value not in (None, np.nan) else None
    return value


def convert_and_validate(item: dict) -> PropertyModel:
    """Convert and validate the item using PropertyModel."""
    item['Street direction']     = safe_nan_check(item.get('Street direction'), str)
    item['Street abbr']          = safe_nan_check(item.get('Street abbr'), str)
    item['Sub-penthouse']        = safe_nan_check(item.get('Sub-penthouse'), bool)
    item['Penthouse collection'] = safe_nan_check(item.get('Penthouse collection'), bool)

    item['Street #'] = (
        str(int(item['Street #']))
        if not isinstance(item['Street #'], str)
        else item['Street #']
    )

    item['Penthouse'] = (
        bool(item['Penthouse'])
        if isinstance(item.get('Penthouse'), (float, np.float64))
        else item.get('Penthouse')
    )

    return PropertyModel(**item)


def boilerplate_read_filter_settings() -> List[PropertyModel]:
    """
        Return a list of PropertyModel instances with predefined properties.
    """
    properties = [
        {
            'Street #': '6801',
            'Street name': 'Queen',
            'Street abbr': 'St',
            'Street direction': None,
            'Apt/Unit': None,
            'Municipality': 'Toronto',
            'Penthouse': False,
            'Sub-penthouse': None,
            'Penthouse collection': None,
        },
        {
            'Street #': '7119',
            'Street name': 'Bloor',
            'Street abbr': 'St',
            'Street direction': None,
            'Apt/Unit': None,
            'Municipality': 'Toronto',
            'Penthouse': False,
            'Sub-penthouse': None,
            'Penthouse collection': None,
        },
        {
            'Street #': '9864',
            'Street name': 'Yonge',
            'Street abbr': 'St',
            'Street direction': None,
            'Apt/Unit': None,
            'Municipality': 'Toronto',
            'Penthouse': False,
            'Sub-penthouse': None,
            'Penthouse collection': None,
        },
    ]
    
    return [PropertyModel(**prop) for prop in properties]



if __name__ == '__main__':
    from pprint import pprint

    properties = boilerplate_read_filter_settings()

    if len(properties) > 10:
        pprint(properties[:10], indent=4)
    else:
        pprint(properties, indent=4)
