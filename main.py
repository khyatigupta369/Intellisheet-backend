from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import os
import uuid
import tempfile
import requests
import logging
import sys
import traceback
import cloudinary
import cloudinary.uploader
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import OpenAI
import json
import io

# ==========================================================================
# CONFIGURATION & INITIALIZATION
# ==========================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("excel-transformer")
logger.info("Starting Excel Transformer")

# Load environment variables
load_dotenv()

# Configure OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found. Please set it in your .env file.")

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Initialize FastAPI app
app = FastAPI(
    title="Excel Transformer",
    description="API for transforming Excel data with AI based on user prompts",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:5173", 
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:5173",
        "https://intellisheet-frontend.vercel.app"  # Add your Vercel URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================================
# HEALTH CHECK ENDPOINTS
# ==========================================================================

@app.get("/")
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "Excel Transformer API is running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "transform": "/transform-excel",
            "upload": "/upload-excel"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": pd.Timestamp.now().isoformat(),
        "openai_configured": bool(OPENAI_API_KEY),
        "cloudinary_configured": bool(os.getenv("CLOUDINARY_CLOUD_NAME"))
    }

# ==========================================================================
# PYDANTIC MODELS
# ==========================================================================

class ExcelTransformRequest(BaseModel):
    url: str
    prompt: str
    tab_name: Optional[str] = None

class ExcelUrlRequest(BaseModel):
    url: str

class FileUploadResponse(BaseModel):
    file_url: str
    public_id: str
    resource_type: str
    created_at: str
    size: int
    format: Optional[str] = None

class FileDownloadRequest(BaseModel):
    url: str
    filename: Optional[str] = None

# ==========================================================================
# HELPER FUNCTIONS
# ==========================================================================

def call_openai_api(prompt: str, system_message: str = None) -> str:
    """Call the OpenAI API with error handling"""
    try:
        logger.info("Calling OpenAI API")
        
        # API configuration
        client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        
        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        # Make API call
        response = client.chat.completions.create(
            model="o3-mini-2025-01-31",
            messages=messages
        )
        
        # Extract response text
        result = response.choices[0].message.content
        logger.info(f"OpenAI API response received (length: {len(result)} characters)")
        return result
        
    except Exception as e:
        logger.error(f"OpenAI API call failed: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error calling OpenAI API: {str(e)}"

def download_excel_from_url(url: str) -> str:
    """Download Excel file from URL and save to temporary file"""
    logger.info(f"Downloading Excel file from URL: {url}")
    
    try:
        # Parse URL to get filename
        parsed_url = urlparse(url)
        file_name = os.path.basename(parsed_url.path) or f"excel_{uuid.uuid4().hex[:8]}.xlsx"
        
        # Handle missing extension
        if not file_name.lower().endswith(('.xlsx', '.xls', '.xlsm')):
            file_name += '.xlsx'
        
        # Download the file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save content to temporary file
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            
            logger.info(f"Excel file downloaded to {temp_file.name}")
            return temp_file.name
    except Exception as e:
        logger.error(f"Failed to download Excel file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Could not download Excel file from URL: {str(e)}")

def upload_to_cloudinary(file_path: str, folder: str = "excel_transformations") -> str:
    """Upload file to Cloudinary and return URL"""
    try:
        logger.info(f"Uploading file to Cloudinary")
        result = cloudinary.uploader.upload(
            file_path,
            resource_type="raw",
            folder=folder,
            public_id=f"excel_{uuid.uuid4().hex[:8]}",
            use_filename=True
        )
        file_url = result.get('secure_url')
        logger.info(f"File uploaded to Cloudinary: {file_url}")
        return file_url
    except Exception as e:
        logger.error(f"Cloudinary upload failed: {str(e)}")
        raise

def upload_file_to_cloudinary(file_path: str, folder: str = "uploads", resource_type: str = "auto") -> Dict[str, Any]:
    """
    Enhanced upload to Cloudinary with more metadata
    
    Args:
        file_path: Path to file to upload
        folder: Folder in Cloudinary to upload to
        resource_type: Type of resource (auto, image, raw, video)
        
    Returns:
        Dictionary with upload details
    """
    try:
        logger.info(f"Uploading file to Cloudinary: {os.path.basename(file_path)}")
        result = cloudinary.uploader.upload(
            file_path,
            resource_type=resource_type,
            folder=folder,
            public_id=f"file_{uuid.uuid4().hex[:8]}",
            use_filename=True
        )
        
        # Extract useful data from result
        response = {
            "file_url": result.get('secure_url'),
            "public_id": result.get('public_id'),
            "resource_type": result.get('resource_type'),
            "created_at": result.get('created_at'),
            "format": result.get('format'),
            "size": result.get('bytes')
        }
        
        logger.info(f"File uploaded to Cloudinary: {response['file_url']}")
        return response
    except Exception as e:
        logger.error(f"Cloudinary upload failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Cloudinary upload failed: {str(e)}")

def download_file_from_url(url: str, output_filename: Optional[str] = None) -> str:
    """
    Download file from URL and save to temporary file
    
    Args:
        url: URL to download file from
        output_filename: Optional filename to use for the downloaded file
        
    Returns:
        Path to downloaded file
    """
    logger.info(f"Downloading file from URL: {url}")
    
    try:
        # Parse URL to get filename if not provided
        if not output_filename:
            parsed_url = urlparse(url)
            output_filename = os.path.basename(parsed_url.path) or f"file_{uuid.uuid4().hex[:8]}"
        
        # Create temp file with appropriate extension
        file_ext = os.path.splitext(output_filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            # Download the file
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save content to temporary file
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            
            logger.info(f"File downloaded to {temp_file.name}")
            return temp_file.name
    except Exception as e:
        logger.error(f"Failed to download file: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Could not download file from URL: {str(e)}")

def execute_transformation(df: pd.DataFrame, transformation_code: str) -> pd.DataFrame:
    """Execute the transformation code on the dataframe with enhanced logging"""
    try:
        logger.info("Executing transformation code:")
        logger.info(f"Code to execute:\n{transformation_code}")
        
        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()
        
        # Log initial state 
        logger.info(f"Original DataFrame shape: {df.shape}")
        
        # Create a local namespace for execution
        local_namespace = {
            "df": df_copy,
            "pd": pd,
            "np": np
        }
        
        # Execute the transformation code
        exec(transformation_code, {}, local_namespace)
        
        # Get the transformed dataframe (assuming it's named 'result')
        if 'result' in local_namespace:
            transformed_df = local_namespace['result']
            
            # Log transformation effects
            value_changes = 0
            unique_values_before = None
            unique_values_after = None
            
            # If we can detect which columns were modified by analyzing the code
            # This is a simple check for common patterns
            if "loc" in transformation_code or "iloc" in transformation_code:
                # Try to extract the column name
                import re
                col_match = re.search(r"(?:loc|iloc).*['\"]([^'\"]+)['\"]", transformation_code)
                if col_match:
                    col_name = col_match.group(1)
                    if col_name in df.columns:
                        # Count non-matching values
                        value_changes = (df[col_name] != transformed_df[col_name]).sum()
                        unique_values_before = df[col_name].unique()[:10]  # First 10 unique values
                        unique_values_after = transformed_df[col_name].unique()[:10]  # First 10 unique values
                        
                        logger.info(f"Modified column: '{col_name}' - {value_changes} values changed")
                        logger.info(f"Sample unique values before: {unique_values_before}")
                        logger.info(f"Sample unique values after: {unique_values_after}")
            
            logger.info(f"Transformation successful: {transformed_df.shape}")
            return transformed_df
        else:
            logger.warning("Transformation didn't produce a 'result' variable, using original dataframe")
            return df_copy
            
    except Exception as e:
        logger.error(f"Error executing transformation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error transforming data: {str(e)}")

def transform_excel_with_prompt(file_path: str, prompt: str, tab_name: str = None) -> Dict[str, Any]:
    """Transform Excel data with AI based on user prompt"""
    logger.info(f"Transforming Excel file with prompt")
    
    try:
        # Read all sheet names
        xl = pd.ExcelFile(file_path)
        all_sheets = xl.sheet_names
        
        # Determine which sheet to use
        selected_tab = tab_name
        if not selected_tab or selected_tab not in all_sheets:
            # Check if prompt mentions a sheet name
            for sheet in all_sheets:
                if sheet.lower() in prompt.lower():
                    selected_tab = sheet
                    logger.info(f"Found sheet name '{sheet}' in prompt")
                    break
            
            # Fall back to first sheet if no match
            if not selected_tab or selected_tab not in all_sheets:
                selected_tab = all_sheets[0]
                logger.info(f"Using first sheet: {selected_tab}")
        
        # Read the data
        df = pd.read_excel(file_path, sheet_name=selected_tab)
        logger.info(f"Read data from sheet '{selected_tab}' with shape: {df.shape}")
        
        # Get data preview for prompt
        preview_rows = min(10, df.shape[0])
        data_preview = df.head(preview_rows).to_string()
        
        # Create rich context for the AI
        ai_prompt = f"""
You are an Excel data transformation expert. Transform the following Excel data according to the user's requirements.

FILE STRUCTURE: This Excel file contains {len(all_sheets)} sheets: {', '.join(all_sheets)}
CURRENT SHEET: {selected_tab}
DATA SHAPE: {df.shape[0]} rows × {df.shape[1]} columns
COLUMN NAMES: {', '.join(str(col) for col in df.columns)}

DATA PREVIEW:
{data_preview}

USER TRANSFORMATION PROMPT:
{prompt}

I need you to provide Python code that will transform this data according to the user's requirements.
Don't explain what the code does or provide any commentary. Just give me the exact Python code to transform the dataframe.

Your code should:
1. Use the dataframe which is already loaded in a variable named 'df'
2. Store the final transformed dataframe in a variable named 'result'
3. Focus on practical transformations like filtering, adding/modifying columns, aggregations, etc.
4. Not include any code for reading or writing files

IMPORTANT: ONLY PROVIDE THE PYTHON CODE, DO NOT INCLUDE ANY EXPLANATION OR COMMENTARY.
"""

        # Call OpenAI API for transformation code
        system_message = "You are a data transformation expert who provides Python code for Pandas dataframe transformations. ONLY RETURN CODE, NO EXPLANATIONS."
        code_result = call_openai_api(ai_prompt, system_message)
        cleaned_code = code_result.strip()
        if cleaned_code.startswith("```python"):
            cleaned_code = cleaned_code.split("```python", 1)[1]
        if cleaned_code.endswith("```"):
            cleaned_code = cleaned_code.rsplit("```", 1)[0]
        cleaned_code = cleaned_code.strip()

        logger.info("Transformation code generated")
        logger.info(f"Full transformation code:\n{cleaned_code}")  # <-- Add this line

        # Execute the transformation
        transformed_df = execute_transformation(df, cleaned_code)
        
        return {
            "sheet_name": selected_tab,
            "original_df": df,
            "transformed_df": transformed_df,
            "transformation_code": cleaned_code,
            "available_sheets": all_sheets
        }
        
    except Exception as e:
        logger.error(f"Error transforming Excel with prompt: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error transforming Excel: {str(e)}")

def create_output_excel(transformation_result: Dict[str, Any], file_path: str) -> str:
    """Create output Excel file with original and transformed data"""
    logger.info(f"Creating output Excel file")
    
    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Original data sheet
            transformation_result["original_df"].to_excel(
                writer, sheet_name="Original Data", index=False
            )
            
            # Transformed data sheet
            transformation_result["transformed_df"].to_excel(
                writer, sheet_name="Transformed Data", index=False
            )
            
            # Transformation code sheet
            pd.DataFrame({
                "Transformation Code": [transformation_result["transformation_code"]],
                "Sheet Used": [transformation_result["sheet_name"]],
                "Available Sheets": [", ".join(transformation_result["available_sheets"])]
            }).to_excel(writer, sheet_name="Transformation Details", index=False)
            
        logger.info(f"Output Excel file created: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error creating output Excel: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating output file: {str(e)}")

# ==========================================================================
# API ENDPOINTS
# ==========================================================================

@app.post("/list-tabs")
async def list_tabs(request: ExcelUrlRequest):
    """List all available tabs/sheets in an Excel file"""
    request_id = uuid.uuid4().hex[:8]
    temp_file = None
    
    try:
        logger.info(f"[{request_id}] Listing tabs for Excel URL: {request.url}")
        
        # Download Excel file
        temp_file = download_excel_from_url(request.url)
        
        # Read sheet names
        xl = pd.ExcelFile(temp_file)
        sheet_names = xl.sheet_names
        
        # Get sheet previews
        sheet_previews = {}
        for sheet in sheet_names:
            try:
                df = pd.read_excel(temp_file, sheet_name=sheet, nrows=3)
                if not df.empty:
                    # Get column names and row count
                    cols = list(df.columns)
                    row_count = len(pd.read_excel(temp_file, sheet_name=sheet))
                    sheet_previews[sheet] = {
                        "columns": cols[:10],  # First 10 columns
                        "column_count": len(cols),
                        "row_count": row_count
                    }
                else:
                    sheet_previews[sheet] = {"columns": [], "column_count": 0, "row_count": 0}
            except Exception as e:
                logger.error(f"Error reading sheet {sheet}: {str(e)}")
                sheet_previews[sheet] = {"error": str(e)}
        
        response = {
            "filename": os.path.basename(urlparse(request.url).path),
            "total_sheets": len(sheet_names),
            "sheet_names": sheet_names,
            "sheet_details": sheet_previews
        }
        
        logger.info(f"[{request_id}] Found {len(sheet_names)} sheets")
        return response
        
    except Exception as e:
        logger.error(f"[{request_id}] Error listing tabs: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass

@app.post("/transform-excel")
async def transform_excel(request: ExcelTransformRequest):
    """
    Transform Excel file from URL based on user prompt and return Cloudinary URL of output file
    """
    request_id = uuid.uuid4().hex[:8]
    temp_file = None
    temp_output = None
    
    try:
        logger.info(f"[{request_id}] Processing transformation request - URL: {request.url}, Prompt: {request.prompt}")
        
        # Download Excel file
        temp_file = download_excel_from_url(request.url)
        
        # Transform data with AI
        transformation_result = transform_excel_with_prompt(temp_file, request.prompt, request.tab_name)
        
        # Create output file
        output_uuid = uuid.uuid4().hex[:8]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'_transformed_{output_uuid}.xlsx') as output_file:
            temp_output = output_file.name
        
        # Create output Excel
        create_output_excel(transformation_result, temp_output)
        
        # Upload to Cloudinary
        output_url = upload_to_cloudinary(temp_output)
        
        # Create response
        original_shape = transformation_result["original_df"].shape
        transformed_shape = transformation_result["transformed_df"].shape
        
        # Detect changes between original and transformed
        changes = {
            "rows": {
                "original": original_shape[0],
                "transformed": transformed_shape[0],
                "difference": transformed_shape[0] - original_shape[0]
            },
            "columns": {
                "original": original_shape[1],
                "transformed": transformed_shape[1],
                "difference": transformed_shape[1] - original_shape[1]
            }
        }
        
        # Detect new/removed columns
        original_cols = set(transformation_result["original_df"].columns)
        transformed_cols = set(transformation_result["transformed_df"].columns)
        
        new_columns = [col for col in transformed_cols if col not in original_cols]
        removed_columns = [col for col in original_cols if col not in transformed_cols]
        
        changes["columns"]["added"] = new_columns
        changes["columns"]["removed"] = removed_columns
        
        # Create response with more detailed information
        response = {
            "output_url": output_url,
            "sheet_transformed": transformation_result["sheet_name"],
            "available_sheets": transformation_result["available_sheets"],
            "prompt": request.prompt,
            "changes": changes,
            "code_preview": transformation_result["transformation_code"][:200] + "..." 
                if len(transformation_result["transformation_code"]) > 200 
                else transformation_result["transformation_code"],
            "transformation_code": transformation_result["transformation_code"],  # <-- Add this line
            "data_changes": {
                "rows_modified": 0,  # Will be updated if we can calculate this
                "transformation_details": "Value modifications in existing columns don't change row/column counts"
            }
        }
        
        # Try to extract more specific changes
        if "mask" in transformation_result["transformation_code"]:
            try:
                # Re-execute just the mask part to get count of affected rows
                locals_dict = {"df": transformation_result["original_df"], "pd": pd, "np": np}
                mask_line = next((line for line in transformation_result["transformation_code"].split("\n") if "mask" in line), None)
                if mask_line:
                    exec(mask_line, {}, locals_dict)
                    if "mask" in locals_dict:
                        affected_rows = locals_dict["mask"].sum()
                        response["data_changes"]["rows_modified"] = int(affected_rows)
                        response["data_changes"]["transformation_details"] = f"{affected_rows} rows had values modified in existing columns"
            except Exception as mask_e:
                logger.warning(f"Could not calculate affected rows: {str(mask_e)}")
        
        logger.info(f"[{request_id}] Transformation completed successfully - Output URL: {output_url}")
        return response
        
    except Exception as e:
        logger.error(f"[{request_id}] Error processing transformation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp files
        for temp in [temp_file, temp_output]:
            if temp and os.path.exists(temp):
                try:
                    os.unlink(temp)
                except:
                    pass

# ==========================================================================
# FILE UPLOAD/DOWNLOAD API ENDPOINTS
# ==========================================================================

@app.post("/upload-to-cloudinary")
async def upload_to_cloudinary_endpoint(request: FileDownloadRequest):
    """
    Upload a file from a URL to Cloudinary and return the Cloudinary URL
    
    This endpoint takes a URL to a file, downloads it, and uploads it to Cloudinary.
    It returns the Cloudinary URL and additional metadata about the uploaded file.
    """
    request_id = uuid.uuid4().hex[:8]
    temp_file = None
    
    try:
        logger.info(f"[{request_id}] Processing upload request - URL: {request.url}")
        
        # Download file from URL
        temp_file = download_file_from_url(request.url, request.filename)
        
        # Upload to Cloudinary
        upload_result = upload_file_to_cloudinary(
            temp_file, 
            folder="uploads",
            resource_type="auto"
        )
        
        logger.info(f"[{request_id}] Upload completed successfully - Output URL: {upload_result['file_url']}")
        return FileUploadResponse(**upload_result)
        
    except Exception as e:
        logger.error(f"[{request_id}] Error processing upload: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass

@app.post("/download-from-url")
async def download_from_url_endpoint(request: FileDownloadRequest):
    """
    Download a file from a URL and upload it to Cloudinary
    
    This endpoint is similar to upload-to-cloudinary, but provides more detailed
    information about the download process and supports specifying a custom filename.
    """
    request_id = uuid.uuid4().hex[:8]
    temp_file = None
    
    try:
        logger.info(f"[{request_id}] Processing download request - URL: {request.url}")
        
        # Get filename from request or generate one
        filename = request.filename
        if not filename:
            parsed_url = urlparse(request.url)
            filename = os.path.basename(parsed_url.path) or f"file_{uuid.uuid4().hex[:8]}"
        
        # Download file
        temp_file = download_file_from_url(request.url, filename)
        
        # Get file size and info
        file_size = os.path.getsize(temp_file)
        file_ext = os.path.splitext(filename)[1].lstrip('.')
        
        # Upload to Cloudinary with appropriate resource type
        resource_type = "raw"
        if file_ext.lower() in ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'tiff', 'svg']:
            resource_type = "image"
        elif file_ext.lower() in ['mp4', 'mov', 'avi', 'wmv', 'flv', 'webm']:
            resource_type = "video"
        
        upload_result = upload_file_to_cloudinary(
            temp_file,
            folder="downloads",
            resource_type=resource_type
        )
        
        # Add additional info to response
        response = FileUploadResponse(**upload_result)
        
        logger.info(f"[{request_id}] Download and upload completed - Output URL: {response.file_url}")
        return response
        
    except Exception as e:
        logger.error(f"[{request_id}] Error processing download: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass

@app.post("/upload-file", response_model=FileUploadResponse)
async def upload_file_endpoint(
    file: UploadFile = File(...),
    folder: str = Form("uploads"),
    resource_type: str = Form("auto")
):
    """
    Upload a file directly to Cloudinary
    
    This endpoint accepts a file upload and sends it to Cloudinary.
    It supports specifying a folder and resource type.
    """
    request_id = uuid.uuid4().hex[:8]
    temp_file = None
    
    try:
        logger.info(f"[{request_id}] Processing direct file upload - Name: {file.filename}")
        
        # Save uploaded file to temp file
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            temp.write(content)
            temp_file = temp.name
        
        # Upload to Cloudinary
        upload_result = upload_file_to_cloudinary(
            temp_file, 
            folder=folder,
            resource_type=resource_type
        )
        
        logger.info(f"[{request_id}] Upload completed successfully - Output URL: {upload_result['file_url']}")
        return FileUploadResponse(**upload_result)
        
    except Exception as e:
        logger.error(f"[{request_id}] Error processing upload: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass

# ==========================================================================
# MAIN ENTRY POINT
# ==========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("Excel Transformer API started")