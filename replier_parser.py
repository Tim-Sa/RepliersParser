#!/venv/bin/python
import os
import re
import time
import json
import asyncio
import platform
import logging
from functools import wraps
from collections import Counter
from json import JSONDecodeError
from typing import List, Dict, Any, Tuple, Optional

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
    street_name: str = Field(..., alias='Street name')
    municipality: str = Field(..., alias='Municipality')
    street_number: str = Field(..., alias='Street #')
    apt_unit: Optional[str] = Field(None, alias='Apt/Unit')
    street_abbr: Optional[str] = Field(None, alias='Street abbr')
    street_direction: Optional[str] = Field(None, alias='Street direction')


class AddressModel(BaseModel):
    street_number: str
    street_name: str
    city: str
    neighborhood: Optional[str] = None
    street_abbr: Optional[str] = None
    apt_unit: Optional[str] = None

    @validator('apt_unit', pre=True, always=True)
    def validate_apt_unit(cls, value: Optional[str]) -> Optional[str]:
        """Clean and lowercase the apartment unit string.

        Args:
            value (Optional[str]): The apartment unit value to validate.

        Returns:
            Optional[str]: Cleaned and lowercased apartment unit, or None if empty.
        """
        if value:
            return re.sub(r'[^a-zA-Z0-9]', '', value).lower()
        return value


class BuildingModel(BaseModel):
    address: AddressModel
    status: str
    list_price: float
    is_available: bool


def silence_event_loop_closed(func):
    """Decorator to silence 'Event loop is closed' errors."""
    
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except RuntimeError as e:
            if str(e) != 'Event loop is closed':
                raise
    return wrapper


def set_event_loop_policy():
    """Set event loop policy for Windows if necessary."""
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def replier_request(
    session: aiohttp.ClientSession, 
    token: str, 
    address: AddressModel, 
    retries: int = RETRIES, 
    delay: int = TIME_DELAY) -> Tuple[Dict[str, Any], int]:
    """Make a request to the repliers API for building listings.

    Args:
        session (aiohttp.ClientSession): The active aiohttp session.
        token (str): The API key for authentication.
        address (AddressModel): The address to query.
        retries (int, optional): Number of retry attempts. Defaults to RETRIES.
        delay (int, optional): Delay between retries in seconds. Defaults to TIME_DELAY.

    Returns:
        Tuple[Dict[str, Any], int]: The JSON response and the HTTP status code.
    """
    
    headers = {
        'REPLIERS-API-KEY': token,
        "content-type": "application/json",
        'accept': 'application/json'
    }
    
    params = f'city={address.city}&streetName={address.street_name}'
    url = f'https://api.repliers.io/listings/?listings=true&{params}'

    for attempt in range(retries):
        async with session.get(url, headers=headers) as response:
            status = response.status
            logger.info('Response status code: %d', status)

            try:
                json_result = await response.json()
                return json_result, status
            except JSONDecodeError:
                logger.error('Failed to decode JSON response')
                return {}, status


def filter_buildings(data: Dict[str, Any]) -> List[BuildingModel]:
    """Filter the API response to create a list of BuildingModel instances.

    Args:
        data (Dict[str, Any]): The raw data from the API response.

    Returns:
        List[BuildingModel]: A list of filtered and formatted BuildingModel instances.
    """
    
    filtered_buildings = []

    for building_data in data.get('listings', []):
        address_data = building_data.get("address", {})
        list_price = float(building_data.get("listPrice", 0).replace(",", ""))
        status = building_data.get("status", "Unknown")
        street_number = address_data.get("streetNumber")
        street_name = address_data.get("streetName")
        city = address_data.get("city", "Unknown City")

        if not street_number or not street_name:
            logger.warning("Missing street_number or street_name in %s", address_data)
            continue

        address = AddressModel(
            city=city,
            street_number=street_number,
            street_name=street_name,
            street_abbr=address_data.get("streetDirection"),
            apt_unit=address_data.get("unitNumber"),
            neighborhood=address_data.get("neighborhood")
        )

        is_available = status in ["A", "Active"]

        filtered_buildings.append(BuildingModel(
            address=address,
            status=status,
            list_price=list_price,
            is_available=is_available
        ))

    return filtered_buildings


async def get_buildings_info(
    session: aiohttp.ClientSession, 
    filter_params: PropertyModel) -> List[BuildingModel]:
    """Fetch building information based on provided filter parameters.

    Args:
        session (aiohttp.ClientSession): The active aiohttp session.
        filter_params (PropertyModel): The filter settings for querying.

    Returns:
        List[BuildingModel]: A list of BuildingModel instances retrieved from the API.
    """
    
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

        if 199 < response[1] <= 299:
            data = response[0]
            if USE_CACHE:
                await redis.set(redis_key, json.dumps(data))  # Cache the result
            return filter_buildings(data)
        else:
            logger.error('Bad request: %s - %s - %s',
                         filter_params.municipality,
                         filter_params.street_number,
                         filter_params.street_name)
            return []

    return json.loads(cached_data) if cached_data else []


def normalize_key(key: str) -> str:
    """Normalize a string key by stripping whitespace and converting to lowercase.

    Args:
        key (str): The key to normalize.

    Returns:
        str: The normalized key.
    """
    return re.sub(r'\s+', ' ', key.strip()).lower()


def extract_street_name(key: str) -> str:
    """Extract the street name from a full address key, ignoring numeric prefixes.

    Args:
        key (str): The full address key.

    Returns:
        str: The extracted street name.
    """
    parts = key.strip().split()
    return ' '.join(parts[1:])


def are_keys_permutations(key1: str, key2: str) -> bool:
    """Check if two keys are permutations of each other.

    Args:
        key1 (str): The first key.
        key2 (str): The second key.

    Returns:
        bool: True if the keys are permutations of each other, False otherwise.
    """
    return Counter(key1) == Counter(key2)


async def form_result_buildings(
    filter_settings: List[PropertyModel], 
    grouped_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Formulate a list of result buildings based on filter settings and grouped data.

    Args:
        filter_settings (List[PropertyModel]): The filter settings for matching.
        grouped_data (Dict[str, List[Dict[str, Any]]]): The grouped building data.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing formatted building models.
    """
    
    building_models = []
    normalized_filter_settings = {
        extract_street_name(normalize_key(f"{params.street_number} {params.street_name}")): params for params in filter_settings
    }

    for street_key, info_list in grouped_data.items():
        normalized_key = normalize_key(street_key)
        street_name = extract_street_name(normalized_key)
        
        matched_params = None
        for normalized_filter_key, params in normalized_filter_settings.items():
            if are_keys_permutations(street_name, normalized_filter_key):
                matched_params = params
                break
        
        if matched_params:
            address = {
                "street_number": matched_params.street_number,
                "street_name": matched_params.street_name,
                "city": matched_params.municipality,
                "apt_unit": None,
                "neighborhood": None
            }
            
            for info in info_list:
                address["apt_unit"] = info["apt_unit"]

                building_model = {
                    "address": address,
                    "status": info["status"],
                    "list_price": info["price"],
                    "is_available": info["status"] == "Available"
                }
                building_models.append(building_model)
        else:
            logger.warning("No matching PropertyModel found for street key: %s", street_key)

    return building_models


async def parse(filter_settings: List[PropertyModel]):
    """Main function to parse filter settings, make API requests, and save results.

    Args:
        filter_settings (List[PropertyModel]): The filter settings for querying buildings.

    Returns:
        List[Dict[str, Any]]: A list of result buildings after processing.
    """
    
    set_event_loop_policy()
    start_time = time.time()
    grouped_data = {}

    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(get_buildings_info(session, params)) for params in filter_settings]
        results = await asyncio.gather(*tasks)

        for buildings in results:
            for building in buildings:
                street_key = f"{building.address.street_number} {building.address.street_name}"
                if street_key not in grouped_data:
                    grouped_data[street_key] = []
                grouped_data[street_key].append({
                    "number": building.address.street_number,
                    "price": building.list_price,
                    "status": "Available" if building.is_available else "Occupied",
                    "apt_unit": building.address.apt_unit,
                })

    result = await form_result_buildings(filter_settings, grouped_data)

    with open('grouped_buildings.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=4)

    elapsed_time = round(time.time() - start_time, 2)
    logger.info('Run completed in %d seconds.', elapsed_time)

    return result


def main():
    """Main entry point for the script to run."""
    
    property_models = boilerplate_read_filter_settings()
    asyncio.run(parse(property_models))


if __name__ == '__main__':
    main()
