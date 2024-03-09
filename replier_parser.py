#!/venv/bin/python
import os
import re
import time
import json
import asyncio
import platform
from functools import wraps
from json import JSONDecodeError
from asyncio.proactor_events import _ProactorBasePipeTransport

import aiohttp
import redis.asyncio as aioredis
from dotenv import load_dotenv


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


# Cache for storing requests results for preventing reply of similar requests
CACHE_REDIS_HOST = 'redis://localhost'
redis = aioredis.from_url(CACHE_REDIS_HOST, 
                          encoding="utf8", 
                          decode_responses=True)


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
        f_street_number: str - street number for filtering
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


def read_filter_settings(path: str) -> list[dict]:
    '''
    read json file with filter params
    '''
    
    with open(path, "r", encoding='utf-8') as f:
        settings = json.load(f)
 
    return settings


async def get_buildings_info(session: aiohttp.ClientSession, 
                             city: str, 
                             street_name: str, 
                             street_number: str, 
                             street_abbr: str, 
                             apt_unit: str) -> dict:
    
    '''
    get info about buildings from specified street and city, 
    filter buildings with specified street abbreviature and unit code

    params:
        session: iohttp.ClientSession - aiohttp session for async requests to API
        token: str - API key for api.repliers.io
        city: str - city to search
        street_name: str - street name to search
        street_number: str - street number to search
        street_abbr: str - street abbreviature for filtering
        apt_unit: str - unit code for filtering

    returns
        dictionary (hash map) - info about finded and filtered buildings
    '''
    
    # get from envirement API Key (don't want store it in code text)
    load_dotenv()
    token = os.getenv('replier_token')

    # generation of redis DB key for an item
    redis_key = f'{city}.{street_number}.{street_name}'

    # try get request result from item
    data = await redis.get(redis_key)

    # if not request result in redis just send request
    if data is None:
        print(f'start request for: \n\t{city} - {street_number} - {street_name}')
        response = await replier_request(session=session, 
                                         token=token, 
                                         city=city, 
                                         street_name=street_name, 
                                         street_number=street_number)

        status_code = response[1]
        print('status code is:', status_code)

        # if all good, just serialize JSON response
        if 199 < status_code <= 299:
            data = response[0]
            data_str = json.dumps(data)

            # Use Redis pipeline for bulk data insertion (save runtime)
            pipe = redis.pipeline()
            pipe.set(redis_key, data_str)
            await pipe.execute()
            
        else:
            print(f'bad request for: {city} - {street_number} - {street_name}')
            data = None
            return {}
        

async def parse():
    '''
    read filter params -> create async tasks for make requests and filter buildings ->
    gathering info from tasks -> create json file with output results
    '''
    start = time.time()

    filtered_data = {'listings': []}

    async with aiohttp.ClientSession() as session:
        tasks = []

        # read data source with filter params
        filter_settings = read_filter_settings('search_settings.json')

        # for every line from table get filter fields
        for filter_build_params in filter_settings:
            city = filter_build_params['Municipality']
            street_name = filter_build_params['Street name']
            street_number = str(filter_build_params['Street #'])
            street_abbr = filter_build_params['Street abbr']
            apt_unit = filter_build_params['Apt/Unit']
            # for every line create async task with specific filter fields
            # task will make request and filtering of request response
            task = asyncio.create_task(get_buildings_info(session, city, street_name, street_number, street_abbr, apt_unit))
            tasks.append(task)

        # gather results from every tasks
        results = await asyncio.gather(*tasks)
        for data in results:
            if not isinstance(data, type(None)):
                filtered_data['listings'] += data
            else:
                print(f"can't read data for {city} - {street_number} - {street_name}")    

    # write result data to json file
    with open('filtered.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4)

    total = round(time.time() - start, 2)
    print(f'run in {total} secs.')


def main():
    asyncio.run(parse())


if __name__ == '__main__':
    main()