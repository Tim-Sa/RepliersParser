# RepliersParser

This script provides a Python-based solution for querying rental property information from an API. It leverages asynchronous programming to handle multiple inquiries simultaneously and cache results using Redis. The code is structured using Pydantic models for data validation and serialization.

## Features

- Asynchronous fetching of rental information from an API.
- Use of Redis for caching results to minimize API calls and improve response time.
- Pydantic models for structured data handling and validation.
- Logging for monitoring requests and identifying issues.

## Requirements

- **Docker Compose** for orchestrating the development environment and run redis server
- **Python** 3.7 or newer
- `aiohttp` for asynchronous HTTP requests
- `redis` for caching data
- `pydantic` for data validation
- `python-dotenv` to manage environment variables

## Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/RepliersParser.git
   cd RepliersParser
   ```

2. **Create a virtual environment**:
   ```bash
    python -m venv venv
    . ./venv/bin/activate 
   ```

3. **Install the required packages**:
   ```bash
    pip install -r requirements.txt
   ```

4. **Set up your replier's token**:
    Create a .env file in the root of the project directory with the following content:
   ```bash
    replier_token=YOUR_API_TOKEN
   ```
   Replace YOUR_API_TOKEN with your actual API key.

## Usage
   ```bash
   docker-compose up -d
   python replier_parser.py
   ```
