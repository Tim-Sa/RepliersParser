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

