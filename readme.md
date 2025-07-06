# Excel Transformer API

AI-powered Excel transformation service that converts natural language prompts into data transformations using OpenAI's API.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the server
python main.py
```

Server runs on `http://localhost:8000`

## 📋 Required Environment Variables

```env
OPENAI_API_KEY=your_openai_api_key
CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_api_key
CLOUDINARY_API_SECRET=your_cloudinary_api_secret
```

## 🔧 Core API Endpoints

### `POST /transform-excel`
Transform Excel data using natural language prompts.

**Request:**
```json
{
  "url": "https://example.com/data.xlsx",
  "prompt": "Filter rows where sales > 1000 and add profit margin column",
  "tab_name": "Sales_Data"  // optional
}
```

**Response:**
```json
{
  "output_url": "https://cloudinary.com/transformed.xlsx",
  "sheet_transformed": "Sales_Data",
  "changes": {
    "rows": {"original": 1000, "transformed": 750, "difference": -250},
    "columns": {"original": 5, "transformed": 6, "added": ["profit_margin"]}
  },
  "transformation_code": "# Generated Python code"
}
```

### `POST /list-tabs`
List all sheets in an Excel file.

**Request:**
```json
{
  "url": "https://example.com/data.xlsx"
}
```

## 🧠 How It Works

1. **Download** Excel file from provided URL
2. **Analyze** data structure and user prompt
3. **Generate** Python transformation code via OpenAI API
4. **Execute** code safely in isolated environment
5. **Create** output Excel with original + transformed data
6. **Upload** to Cloudinary and return URL

## 🎯 Transformation Examples

| Prompt | Result |
|--------|--------|
| `"Filter sales > 1000, sort by date"` | Filters and sorts data |
| `"Add profit margin column (30% of sales)"` | Adds calculated column |
| `"Group by region, show total sales"` | Creates aggregation |
| `"Format currency columns with $ symbol"` | Formats data types |

## 📊 Sample Data Generation

```bash
# Generate test Excel files
python sample_data.py

# Creates:
# - sample_sales_data.xlsx (1000 rows, 3 sheets)
# - sample_employee_data.xlsx (500 rows, 2 sheets)
# - sample_inventory_data.xlsx (300 rows, 2 sheets)
# - sample_financial_data.xlsx (50 accounts, 2000 transactions)
```

## 🧪 Testing

```bash
# Run comprehensive test suite
python api_test_runner.py

# Test specific prompts
python test_prompts.py

# Available test categories:
# - Sales data (filtering, aggregation, performance analysis)
# - Employee data (salary analysis, performance metrics)
# - Inventory data (stock management, profitability)
# - Financial data (account analysis, transaction patterns)
```

## 🏗️ Architecture

```
main.py
├── FastAPI app with CORS
├── OpenAI integration (o3-mini model)
├── Cloudinary file storage
├── Pandas data processing
└── Error handling & logging

Key Functions:
├── transform_excel_with_prompt() - Core transformation logic
├── execute_transformation() - Safe code execution
├── call_openai_api() - AI prompt processing
└── upload_to_cloudinary() - File storage
```

## 🔒 Security Features

- **Sandboxed execution** of generated code
- **Temporary file cleanup** after processing
- **Input validation** for URLs and prompts
- **Error isolation** prevents crashes
- **Request tracking** with unique IDs

## 📈 Performance

- **Concurrent requests** supported via FastAPI
- **Streaming downloads** for large files
- **Efficient memory usage** with temporary files
- **Rate limiting** considerations for OpenAI API

## 🐛 Troubleshooting

**Common Issues:**
- `OPENAI_API_KEY not found` → Set in `.env` file
- [`Cloudinary upload failed`](backend/api_test_runner.py ) → Check credentials
- [`Excel file download failed`](backend/api_test_runner.py ) → Verify URL accessibility
- [`Transformation failed`](backend/api_test_runner.py ) → Check prompt clarity

**Debug Mode:**
```python
# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Dependencies

- **FastAPI** - Web framework
- **OpenAI** - AI transformation engine
- **Pandas** - Data manipulation
- **Cloudinary** - File storage
- **openpyxl** - Excel file handling

## 🚀 Deployment

The API is designed for cloud deployment with:
- Environment variable configuration
- Stateless design
- External file storage
- Comprehensive logging

**Health Check:** `GET /health`
**API Documentation:** `GET /docs` (Swagger UI)

---

*Built with Python 3.9+, FastAPI, and OpenAI API*