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
from pydantic import BaseModel, Field, field_validator


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Configuration constants
load_dotenv()

CACHE_REDIS_HOST = os.getenv('CACHE_REDIS_HOST', 'redis://localhost')
USE_CACHE = os.getenv('USE_CACHE', 'True').lower() in ['true', '1', 't'] 

TIME_DELAY = int(os.getenv('TIME_DELAY', 15))
RETRIES = int(os.getenv('RETRIES', 5))

TOKEN = os.getenv('REPLIERS_API_TOKEN')

# Set up Redis client for caching results
redis = aioredis.from_url(CACHE_REDIS_HOST, encoding="utf8", decode_responses=True)


# Model representing the input details of an interest in a property
class InquiryModel(BaseModel):
    street_name:   str = Field(..., alias='Street name')
    municipality:  str = Field(..., alias='Municipality')
    street_number: str = Field(..., alias='Street #')
    apt_unit:         Optional[str] = Field(None, alias='Apt/Unit')
    street_abbr:      Optional[str] = Field(None, alias='Street abbreviation')
    street_direction: Optional[str] = Field(None, alias='Street direction')


# Model representing a detailed address
class AddressDetailsModel(BaseModel):
    street_number: str
    street_name: str
    city: str
    apt_unit:     Optional[str] = None
    street_abbr:  Optional[str] = None
    neighborhood: Optional[str] = None

    @field_validator('apt_unit')
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


# Model representing the details of a building's availability and pricing
class BuildingDetailsModel(BaseModel):
    address: AddressDetailsModel
    status: str
    rental_price: float
    is_available: bool


def silence_event_loop_closed(func):
    """Decorator to suppress 'Event loop is closed' errors."""
    
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except RuntimeError as e:
            if str(e) != 'Event loop is closed':
                raise
    return wrapper


def set_event_loop_policy():
    """Set the event loop policy for Windows, if necessary."""
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def fetch_building_info(
    session: aiohttp.ClientSession, 
    token: str, 
    address: AddressDetailsModel, 
    retries: int = RETRIES, 
    delay: int = TIME_DELAY) -> Tuple[Dict[str, Any], int]:
    """Send the API request to get current rental information for specific addresses.

    Args:
        session (aiohttp.ClientSession): The active aiohttp session.
        token (str): The API key for authentication.
        address (AddressDetailsModel): The address to query.
        retries (int, optional): Number of retry attempts. Defaults to RETRIES.
        delay (int, optional): Delay between retries in seconds. Defaults to TIME_DELAY.

    Returns:
        Tuple[Dict[str, Any], int]: JSON response and HTTP status code.
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


def parse_building_data(data: Dict[str, Any]) -> List[BuildingDetailsModel]:
    """Convert the raw API response into structured building details.

    Args:
        data (Dict[str, Any]): The raw data from the API response.

    Returns:
        List[BuildingDetailsModel]: A list of filtered and formatted BuildingDetailsModel instances.
    """
    parsed_buildings = []
    
    for building_data in data.get('listings', []):
        parsed_buildings.append(BuildingDetailsModel(
            address=AddressDetailsModel(
                street_number=building_data['address']['streetNumber'],
                street_name=building_data['address']['streetName'],
                city=building_data['address'].get('city', 'Unknown City'),
                apt_unit=building_data['address'].get('unitNumber'),
                street_abbr=building_data['address'].get('streetDirection'),
                neighborhood=building_data['address'].get('neighborhood')
            ),
            status=building_data.get('status', 'Unknown'),
            rental_price=float(building_data.get('listPrice', 0).replace(',', '')),
            is_available=building_data.get('status') in ["A", "Active"]
        ))

    return parsed_buildings



async def retrieve_building_info(
    session: aiohttp.ClientSession, 
    inquiry_params: InquiryModel) -> List[BuildingDetailsModel]:
    """Get fresh data from the API or retrieve cached data for the specified address.

    Args:
        session (aiohttp.ClientSession): The active aiohttp session.
        inquiry_params (InquiryModel): The inquiry details for querying.

    Returns:
        List[BuildingDetailsModel]: A list of BuildingDetailsModel instances retrieved from the API.
    """
    
    token = TOKEN

    redis_key = f'{inquiry_params.municipality}.{inquiry_params.street_number}.{inquiry_params.street_name}'
    cached_data = await redis.get(redis_key) if USE_CACHE else None

    # If not cached, perform API request
    if cached_data is None:
        logger.info('Starting request for: %s - %s - %s',
                    inquiry_params.municipality,
                    inquiry_params.street_number,
                    inquiry_params.street_name)
        address_model = AddressDetailsModel(
            city=inquiry_params.municipality,
            street_number=inquiry_params.street_number,
            street_name=inquiry_params.street_name,
            street_abbr=inquiry_params.street_abbr,
            apt_unit=inquiry_params.apt_unit
        )

        response = await fetch_building_info(session=session, token=token, address=address_model)

        # Process successful response
        if 199 < response[1] <= 299:
            data = response[0]
            if USE_CACHE:
                await redis.set(redis_key, json.dumps(data))  # Cache the result
            return parse_building_data(data)
        else:
            logger.error('Bad request: %s - %s - %s',
                         inquiry_params.municipality,
                         inquiry_params.street_number,
                         inquiry_params.street_name)
            return []

    # Return cached data if available
    return json.loads(cached_data) if cached_data else []


def normalize_address_key(key: str) -> str:
    """Normalize a string key by stripping whitespace and converting to lowercase.

    This ensures a consistent comparison format for address keys.

    Args:
        key (str): The key to normalize.

    Returns:
        str: The normalized key.
    """
    return re.sub(r'\s+', ' ', key.strip()).lower()


def extract_street_name(key: str) -> str:
    """Extract the street name from a full address key, ignoring numeric prefixes.

    This helps in matching and comparing street names accurately.

    Args:
        key (str): The full address key.

    Returns:
        str: The extracted street name.
    """
    parts = key.strip().split()
    return ' '.join(parts[1:])


def are_keys_permutations(key1: str, key2: str) -> bool:
    """Check if two address representations are permutations of each other.

    Args:
        key1 (str): The first key.
        key2 (str): The second key.

    Returns:
        bool: True if the keys are permutations of each other, False otherwise.
    """
    return Counter(key1) == Counter(key2)


async def construct_building_results(
    inquiry_addresses: List[InquiryModel], 
    building_info: Dict[str, List[BuildingDetailsModel]]) -> List[Dict[str, Any]]:
    """Prepare the final output format for building information.

    Args:
        inquiry_addresses (List[InquiryModel]): The inquiry addresses for matching.
        building_info (Dict[str, List[BuildingDetailsModel]]): The grouped building data.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing formatted building models.
    """
    
    building_models = []
    normalized_filter_settings = {
        extract_street_name(normalize_address_key(f"{params.street_number} {params.street_name}")): params for params in inquiry_addresses
    }

    for street_key, info_list in building_info.items():
        normalized_key = normalize_address_key(street_key)
        street_name = extract_street_name(normalized_key)
        
        matched_params = None
        for normalized_filter_key, params in normalized_filter_settings.items():
            if are_keys_permutations(street_name, normalized_filter_key):
                matched_params = params
                break
        
        if matched_params:
            for info in info_list:  # `info` now is a BuildingDetailsModel instance
                building_model = {
                    "address": {
                        "street_number": matched_params.street_number,
                        "street_name": matched_params.street_name,
                        "city": matched_params.municipality,
                        "apt_unit": info.address.apt_unit,
                        "neighborhood": info.address.neighborhood,
                    },
                    "status": info.status,
                    "rental_price": info.rental_price,
                    "is_available": info.is_available
                }
                building_models.append(building_model)
        else:
            logger.warning("No matching InquiryModel found for street key: %s", street_key)

    return building_models


def convert_to_building_details(data: List[dict]) -> List[BuildingDetailsModel]:
    """Convert a list of dictionary entries to a list of BuildingDetailsModel instances.

    Args:
        data (List[dict]): The list of dictionaries with building information.

    Returns:
        List[BuildingDetailsModel]: A list of BuildingDetailsModel instances.
    """
    building_details_list = []

    for item in data:
        address_data = item['address']
        building_detail = BuildingDetailsModel(
            address=AddressDetailsModel(
                apt_unit=address_data.get('apt_unit'),
                city=address_data['city'],
                neighborhood=address_data['neighborhood'],
                street_name=address_data['street_name'],
                street_number=address_data['street_number']
            ),
            status=item['status'],
            rental_price=item['rental_price'],
            is_available=item['is_available']
        )
        building_details_list.append(building_detail)

    return building_details_list


async def parse_inquiries(inquiry_addresses: List[InquiryModel]) -> List[BuildingDetailsModel]:
    """Coordinate the overall handling of property inquiries and ensure results are collected.

    Args:
        inquiry_addresses (List[InquiryModel]): The inquiry addresses for querying buildings.

    Returns:
        List[Dict[str, Any]]: A list of resulting buildings after processing.
    """
    
    set_event_loop_policy()
    start_time = time.time()
    building_info = {}

    async with aiohttp.ClientSession() as session:
        # Create tasks to fetch building information for each inquiry
        tasks = [asyncio.create_task(retrieve_building_info(session, params)) for params in inquiry_addresses]
        results = await asyncio.gather(*tasks)

        # Group building data by street
        for buildings in results:
            for building in buildings:
                street_key = f"{building.address.street_number} {building.address.street_name}"
                if street_key not in building_info:
                    building_info[street_key] = []  # Create a list for buildings in the same location
                building_info[street_key].append(building)  # Append the BuildingDetailsModel instance

    result = await construct_building_results(inquiry_addresses, building_info)

    elapsed_time = round(time.time() - start_time, 2)
    logger.info('Run completed in %d seconds.', elapsed_time)

    return convert_to_building_details(result)


def boilerplate_read_inquiry_addresses() -> List[InquiryModel]:
    """
    Return a list of InquiryModel instances with predefined inquiries.
    """
    inquiries = [
        {
            'Street #': '6801',
            'Street name': 'Queen',
            'Street abbreviation': 'St',
            'Street direction': None,
            'Apt/Unit': None,
            'Municipality': 'Toronto',
        },
        {
            'Street #': '7119',
            'Street name': 'Bloor',
            'Street abbreviation': 'St',
            'Street direction': None,
            'Apt/Unit': None,
            'Municipality': 'Toronto',
        },
        {
            'Street #': '9864',
            'Street name': 'Yonge',
            'Street abbreviation': 'St',
            'Street direction': None,
            'Apt/Unit': None,
            'Municipality': 'Toronto',
        },
    ]
    
    return [InquiryModel(**inquiry) for inquiry in inquiries]


def main():
    from pprint import pprint
    
    property_inquiries = boilerplate_read_inquiry_addresses()
    buildings = asyncio.run(parse_inquiries(property_inquiries))

    pprint(buildings, indent=4)


if __name__ == '__main__':
    main()