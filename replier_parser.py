#!/venv/bin/python
import os
import re
import time
import json
import asyncio
import platform
import logging
from functools import wraps
from json import JSONDecodeError
from asyncio.proactor_events import _ProactorBasePipeTransport
from typing import List, Dict, Tuple, Any, Optional

import aiohttp
import redis.asyncio as aioredis
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

from search_settings import boilerplate_read_filter_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for configuration
CACHE_REDIS_HOST = 'redis://localhost'
TIME_DELAY = 15
RETRIES = 5
USE_CACHE = True

# Redis client setup for caching results
redis = aioredis.from_url(CACHE_REDIS_HOST, encoding="utf8", decode_responses=True)


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


class AddressModel(BaseModel):
    street_number: str
    street_name:   str
    city:          str
    street_abbr: Optional[str] = None
    apt_unit:    Optional[str] = None

    @validator('apt_unit', pre=True, always=True)
    def validate_apt_unit(cls, value: Optional[str]) -> Optional[str]:
        """Ensure the apartment unit string is clean and lowercase."""
        if value:
            return re.sub(r'[^a-zA-Z0-9]', '', value).lower()
        return value


class BuildingModel(BaseModel):
    address: AddressModel
    other_info: Dict[str, Any]


def silence_event_loop_closed(func):
    """Wrapper to silence 'Event loop is closed' errors."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except RuntimeError as e:
            if str(e) != 'Event loop is closed':
                raise     
    return wrapper


def set_event_loop_policy():
    """Set event loop policy for Windows."""
    _ProactorBasePipeTransport.__del__ = silence_event_loop_closed(_ProactorBasePipeTransport.__del__)
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def handle_rate_limiting(attempt: int, retries: int, delay: int):
    """Handle rate limiting by waiting and retrying."""
    if attempt < retries - 1:
        logger.warning("Received 429 error. Retrying in %d seconds...", delay)
        await asyncio.sleep(delay)
        return True
    return False


async def replier_request(
    session: aiohttp.ClientSession,
    token:   str,
    address: AddressModel,
    retries: int = RETRIES,
    delay:   int = TIME_DELAY
) -> Tuple[Dict[str, Any], int]:
    """
        Send a request to repliers API for building info.
    """    
    headers = {
        'REPLIERS-API-KEY': token,
        "content-type": "application/json",
        'accept': 'application/json'
    }

    params = f'city={address.city}&streetName={address.street_name}&streetNumber={address.street_number}'
    url = f'https://api.repliers.io/listings/?listings=true&{params}'

    for attempt in range(retries):
        async with session.get(url, headers=headers) as response:
            status = response.status

            if status == 429:  # Too many requests
                if await handle_rate_limiting(attempt, retries, delay):
                    continue
                return {}, status

            try:
                json_result = await response.json()
                return json_result, status
            except JSONDecodeError:
                return {}, status

    return {}, status


def are_anagrams(street1: str, street2: str) -> bool:
    """Compare two street names to check if they are anagrams."""
    # Remove spaces and convert to lower case for comparison
    normalized_street1 = ''.join(sorted(street1.replace(" ", "").lower()))
    normalized_street2 = ''.join(sorted(street2.replace(" ", "").lower()))
    return normalized_street1 == normalized_street2


def filter_buildings(
    data:          Dict[str, Any], 
    filter_params: PropertyModel
) -> List[BuildingModel]:
    """Filter buildings based on search criteria."""
    filtered_buildings = []

    for building_data in data.get('listings', []):
        address_data = building_data.get("address", {})
        street_number = address_data.get("streetNumber")
        street_name = address_data.get("streetName")

        if not street_number or not street_name:
            logger.warning("Missing street_number or street_name in %s", address_data)
            continue  # Skip if essential fields are missing

        address = AddressModel(
            city=address_data.get("city", "Unknown City"),
            street_number=street_number,
            street_name=street_name,
            street_abbr=address_data.get("streetDirection"),
            apt_unit=address_data.get("unitNumber")
        )

        if (
                filter_params.municipality.lower()  == address.city.lower()  and
                filter_params.street_number.lower() == street_number.lower() and
                are_anagrams(filter_params.street_name, street_name) and
                (
                    filter_params.street_abbr is None or filter_params.street_abbr.lower() == address.street_abbr.lower()
                ) and
                (
                    filter_params.apt_unit.lower() == address.apt_unit.lower() if filter_params.apt_unit else True
                )
            ):
            # Create a BuildingModel instance
            filtered_buildings.append(BuildingModel(address=address, other_info=building_data))

    return filtered_buildings


async def get_buildings_info(
    session:       aiohttp.ClientSession,
    filter_params: PropertyModel
) -> List[BuildingModel]:
    """Fetch building information and filter results."""
    
    load_dotenv()
    token = os.getenv('replier_token')

    redis_key = f'{filter_params.municipality}.{filter_params.street_number}.{filter_params.street_name}'
    cached_data = await redis.get(redis_key) if USE_CACHE else None

    if cached_data is None:
        logger.info('Starting request for: %s - %s - %s',
                    filter_params.municipality,
                    filter_params.street_number,
                    filter_params.street_name)
        address_model = AddressModel(
            city=filter_params.municipality,
            street_number=filter_params.street_number,
            street_name=filter_params.street_name,
            street_abbr=filter_params.street_abbr,
            apt_unit=filter_params.apt_unit
        )

        response = await replier_request(session=session, token=token, address=address_model)

        status_code = response[1]
        logger.info('Status code: %d', status_code)

        if 199 < status_code <= 299:
            data = response[0]
            if USE_CACHE:
                await redis.set(redis_key, json.dumps(data))  # Cache the result
            return filter_buildings(data, filter_params)
        else:
            logger.error('Bad request: %s - %s - %s', 
                         filter_params.municipality,
                         filter_params.street_number, 
                         filter_params.street_name)
            return []

    return json.loads(cached_data) if cached_data else []  # Parse cached data if exists


async def parse(filter_settings: List[PropertyModel]):
    """Main logic for making requests and gathering results."""
    set_event_loop_policy()

    start_time = time.time()
    filtered_data = {
        'apiVersion': '1.0',
        'page': 1,
        'numPages': 1, 
        'pageSize': len(filter_settings),
        'count': 0,
        'statistics': { 
            'totalBuildings': 0,
            'filteredBuildings': 0
        },
        'listings': []
    }

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(
                get_buildings_info(session, params)) 
                for params in filter_settings
        ]

        results = await asyncio.gather(*tasks)
        
        from pprint import pprint
        pprint(results, indent=8)
        exit(0)

        for buildings in results:
            if buildings:
                filtered_data['listings'].extend(buildings)
                filtered_data['statistics']['filteredBuildings'] += len(buildings)
            else:
                logger.warning("Can't read data for %s", filter_settings)

    # Update count
    filtered_data['count'] = len(filtered_data['listings'])

    # Save filtered data to a JSON file
    with open('filtered.json', 'w', encoding='utf-8') as file:
        json.dump(filtered_data, file, indent=4)

    elapsed_time = round(time.time() - start_time, 2)
    logger.info('Run completed in %d seconds.', elapsed_time)

    return filtered_data


def main():
    """Main entry point for the script."""
    property_models = boilerplate_read_filter_settings()
    return asyncio.run(parse(property_models))


if __name__ == '__main__':
    from pprint import pprint
    pprint(main(), indent=4)
