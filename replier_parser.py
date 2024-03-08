from json import JSONDecodeError

import aiohttp


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
            json = await response.json()
        except JSONDecodeError:
            # TODO: signal about problem
            json = {}

        status = response.status
    
    return json, status

