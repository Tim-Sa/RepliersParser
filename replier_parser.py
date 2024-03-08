import re
import json
import asyncio
import platform
from functools import wraps
from json import JSONDecodeError
from asyncio.proactor_events import _ProactorBasePipeTransport

import aiohttp


def silence_event_loop_closed(func):
    '''
        Wrapper for disable raising of 'Event loop is closed' error.
    '''
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except RuntimeError as e:
            if str(e) != 'Event loop is closed':
                raise
    return wrapper


 # disable raising of 'Event loop is closed' errors with all async functions
_ProactorBasePipeTransport.__del__ = silence_event_loop_closed(_ProactorBasePipeTransport.__del__)
if platform.system()=='Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def replier_request(session: aiohttp.ClientSession, 
                          token: str, 
                          city: str, street_name: str, street_number: str) -> (dict, int):
    
    '''
    get from repliers api info about buildings on specific streets

    params:
        session: iohttp.ClientSession - aiohttp session for async requests to API
        token: str - API key for api.repliers.io
        city: str - city to search
        street_name: str - street name to search
        street_number: str - street number to search

    returns
        dictionary - info about finded buildings
        int - response status code
    '''

    # headers for request, there we put API key and type of recieving data
    headers = {'REPLIERS-API-KEY': token, 
               "content-type": "application/json", 
               'accept': 'application/json'}
    
    # url for request with specified params (city, street, etc.)
    params = f'city={city}&streetName={street_name}&streetNumber={street_number}'
    url = f'https://api.repliers.io/listings/?listings=true&{params}'

    # request to repliers API with specified params
    async with session.get(url, headers=headers) as response:
        try:
            json_result = await response.json()
        except JSONDecodeError:
            # TODO: signal about problem
            json_result = {}

        status = response.status
    
    return json_result, status


def filter_buildings(data: dict, 
                     f_city: str, 
                     f_street_number: str, 
                     f_street_name: str, 
                     f_street_abbr: str, 
                     f_apt_unit: str) -> list[dict]:
    
    '''
    filter of items by street abbreviature and unit code.

    params:
        data: dict - data about buildings from API response with this structure - 
            {
                'listings': 
                        [item1(dict), item2(dict), ..., itemN(dict)]
            }
        f_street_number: str - street number to for filtering
        f_street_name: str - street name for filtering
        f_street_abbr: str - street abbreviature for filtering
        f_apt_unit: str - unit code for filtering

    returns
        list of dictionaries (hash map) - info about finded and filtered buildings
    '''

    # unit code for filtering must be in same format with unit code from API response
    #  Ph1004, PH 1004, ph-1004, ph#1004, #ph1004 -> ph1004
    if len(f_apt_unit) > 0:
        f_apt_unit = re.findall(r'[a-zA-Z0-9]+', f_apt_unit)[0].lower()

    filtered = []

    # parse info about every item from API response
    for build in data['listings']:
        addr = build["address"]
        city = addr["city"]
        street_number = addr['streetNumber']
        street_name = addr['streetName']
        street_abbr = addr["streetSuffix"]
        apt_unit = addr["unitNumber"]

        # unit code from API must be in same format with unit code for filtering
        #  Ph1004, PH 1004, ph-1004, ph#1004, #ph1004 -> ph1004
        if len(apt_unit) > 0:
            apt_unit = re.findall(r'[a-zA-Z0-9]+', apt_unit)[0].lower()

        # compare data from API with filter fields
        if (f_city == city and
            f_street_number == street_number and 
            f_street_name == street_name and 
            f_street_abbr == street_abbr and 
            f_apt_unit == apt_unit):

            filtered.append(build)

    return filtered