import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from replier_parser import parse_inquiries, InquiryModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/parse-inquiries/", response_model=List[Dict[str, Any]])
async def handle_inquiries(inquiry_addresses: List[InquiryModel]):
    try:
        results = await parse_inquiries(inquiry_addresses)
        return results

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)