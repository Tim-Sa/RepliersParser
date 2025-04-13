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

from search_settings import boilerplate_read_filter_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
CACHE_REDIS_HOST = 'redis://localhost'
TIME_DELAY = 15  # Delay before retrying a request
RETRIES = 5      # Number of retry attempts
USE_CACHE = True  # Use cache for storing data

# Set up Redis client for caching results
redis = aioredis.from_url(CACHE_REDIS_HOST, encoding="utf8", decode_responses=True)


# Model representing the input details of an interest in a property
class InquiryModel(BaseModel):
    street_name: str = Field(..., alias='Street name')
    municipality: str = Field(..., alias='Municipality')
    street_number: str = Field(..., alias='Street #')
    apt_unit: Optional[str] = Field(None, alias='Apt/Unit')
    street_abbr: Optional[str] = Field(None, alias='Street abbreviation')
    street_direction: Optional[str] = Field(None, alias='Street direction')


# Model representing a detailed address
class AddressDetailsModel(BaseModel):
    street_number: str
    street_name: str
    city: str
    neighborhood: Optional[str] = None
    street_abbr: Optional[str] = None
    apt_unit: Optional[str] = None

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
    """Set event loop policy for Windows if necessary."""
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def fetch_building_info(
    session: aiohttp.ClientSession, 
    token: str, 
    address: AddressDetailsModel, 
    retries: int = RETRIES, 
    delay: int = TIME_DELAY) -> Tuple[Dict[str, Any], int]:
    """Send a request to the repliers API to retrieve building information.

    This function handles the API request to get current rental information for specific addresses.

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
    """Filter the API response to construct a list of BuildingDetailsModel instances.

    This function interprets the raw response from the API to create structured building details.

    Args:
        data (Dict[str, Any]): The raw data from the API response.

    Returns:
        List[BuildingDetailsModel]: A list of filtered and formatted BuildingDetailsModel instances.
    """
    
    parsed_buildings = []

    for building_data in data.get('listings', []):
        address_data = building_data.get("address", {})
        rental_price = float(building_data.get("listPrice", 0).replace(",", ""))
        status = building_data.get("status", "Unknown")
        street_number = address_data.get("streetNumber")
        street_name = address_data.get("streetName")
        city = address_data.get("city", "Unknown City")

        if not street_number or not street_name:
            logger.warning("Missing street_number or street_name in %s", address_data)
            continue

        address = AddressDetailsModel(
            city=city,
            street_number=street_number,
            street_name=street_name,
            street_abbr=address_data.get("streetDirection"),
            apt_unit=address_data.get("unitNumber"),
            neighborhood=address_data.get("neighborhood")
        )

        is_available = status in ["A", "Active"]

        parsed_buildings.append(BuildingDetailsModel(
            address=address,
            status=status,
            rental_price=rental_price,
            is_available=is_available
        ))

    return parsed_buildings


async def retrieve_building_info(
    session: aiohttp.ClientSession, 
    inquiry_params: InquiryModel) -> List[BuildingDetailsModel]:
    """Fetch building information based on provided inquiry parameters.

    This function combines getting fresh data from the API or retrieving cached data for the specified address.

    Args:
        session (aiohttp.ClientSession): The active aiohttp session.
        inquiry_params (InquiryModel): The inquiry details for querying.

    Returns:
        List[BuildingDetailsModel]: A list of BuildingDetailsModel instances retrieved from the API.
    """
    
    load_dotenv()
    token = os.getenv('replier_token')

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

    This function ensures a consistent comparison format for address keys.

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
    """Check if two keys are permutations of each other.

    This function is used to determine if two address representations are similar.

    Args:
        key1 (str): The first key.
        key2 (str): The second key.

    Returns:
        bool: True if the keys are permutations of each other, False otherwise.
    """
    return Counter(key1) == Counter(key2)


async def construct_building_results(
    filter_settings: List[InquiryModel], 
    grouped_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Prepares the final output format of building information.

    Args:
        filter_settings (List[InquiryModel]): The filter settings for matching.
        grouped_data (Dict[str, List[Dict[str, Any]]]): The grouped building data.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing formatted building models.
    """
    
    building_models = []
    normalized_filter_settings = {
        extract_street_name(normalize_address_key(f"{params.street_number} {params.street_name}")): params for params in filter_settings
    }

    for street_key, info_list in grouped_data.items():
        normalized_key = normalize_address_key(street_key)
        street_name = extract_street_name(normalized_key)
        
        matched_params = None
        # Determine if each street key matches filter settings
        for normalized_filter_key, params in normalized_filter_settings.items():
            if are_keys_permutations(street_name, normalized_filter_key):
                matched_params = params
                break
        
        # If matching parameters were found, construct building model
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
                    "rental_price": info["price"],
                    "is_available": info["status"] == "Available"
                }
                building_models.append(building_model)
        else:
            logger.warning("No matching InquiryModel found for street key: %s", street_key)

    return building_models


async def parse_inquiries(filter_settings: List[InquiryModel]):
    """Coordinates the overall handling of property inquiries and ensures results are saved.

    Args:
        filter_settings (List[InquiryModel]): The filter settings for querying buildings.

    Returns:
        List[Dict[str, Any]]: A list of resulting buildings after processing.
    """
    
    set_event_loop_policy()
    start_time = time.time()
    grouped_data = {}

    async with aiohttp.ClientSession() as session:
        # Create tasks to fetch building information for each inquiry
        tasks = [asyncio.create_task(retrieve_building_info(session, params)) for params in filter_settings]
        results = await asyncio.gather(*tasks)

        # Group building data by street
        for buildings in results:
            for building in buildings:
                street_key = f"{building.address.street_number} {building.address.street_name}"
                if street_key not in grouped_data:
                    grouped_data[street_key] = []
                grouped_data[street_key].append({
                    "number": building.address.street_number,
                    "price": building.rental_price,
                    "status": "Available" if building.is_available else "Occupied",
                    "apt_unit": building.address.apt_unit,
                })

    result = await construct_building_results(filter_settings, grouped_data)

    # Save results to a JSON file
    with open('grouped_buildings.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=4)

    elapsed_time = round(time.time() - start_time, 2)
    logger.info('Run completed in %d seconds.', elapsed_time)

    return result


def main():
    property_inquiries = boilerplate_read_filter_settings()
    asyncio.run(parse_inquiries(property_inquiries))
