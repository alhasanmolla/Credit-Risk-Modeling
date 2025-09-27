# Credit Risk Prediction FastAPI Application

This FastAPI application provides a web interface and REST API for credit risk prediction using the trained RNN model.

## Features

- **Web Interface**: User-friendly form for single customer prediction
- **Batch Prediction**: Upload CSV files for multiple customer predictions
- **REST API**: Programmatic access to prediction endpoints
- **Health Check**: Monitor system status and model loading
- **Real-time Validation**: Form validation with immediate feedback
- **Results Export**: Download batch prediction results as CSV

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the model files are available:
   - `../models/rnn_model.h5` - Trained Keras model
   - `../models/feature_scaler.pkl` - Feature scaler

## Running the Application

```bash
python app.py
```

The application will start on `http://localhost:8000`

## API Endpoints

### Web Interface
- `GET /` - Home page with prediction form

### API Endpoints
- `GET /health` - System health check
- `POST /predict/single` - Single customer prediction
- `POST /predict/batch` - Batch prediction from CSV
- `GET /api/docs` - API documentation

### Single Prediction Parameters
```json
{
  "age": 25,
  "sex": "male",
  "job": 2,
  "housing": "own",
  "saving_accounts": "little",
  "checking_account": "moderate",
  "credit_amount": 1000,
  "duration": 24,
  "purpose": "car"
}
```

### Response Format
```json
{
  "status": "success",
  "prediction": 1,
  "probability": 0.85,
  "risk_level": "Low Risk"
}
```

## Usage Examples

### Web Interface
1. Navigate to `http://localhost:8000`
2. Fill out the prediction form
3. Click "Predict Credit Risk"
4. View results and recommendations

### Batch Prediction
1. Prepare a CSV file with customer data
2. Use the batch upload form or API endpoint
3. Download results as CSV

### API Usage (curl)
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict/single" \
  -F "age=25" -F "sex=male" -F "job=2" \
  -F "housing=own" -F "saving_accounts=little" \
  -F "checking_account=moderate" -F "credit_amount=1000" \
  -F "duration=24" -F "purpose=car"

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -F "file=@sample_data.csv"
```

## Sample Data

A sample CSV file (`sample_data.csv`) is provided in the templates directory for testing batch predictions.

## Error Handling

The application includes comprehensive error handling for:
- Missing or invalid model files
- Invalid input data
- Network errors
- File upload issues

## Security Notes

- Input validation is performed on all endpoints
- File uploads are restricted to CSV format
- Error messages are sanitized to prevent information leakage

## Performance

- Model loading is optimized with lazy initialization
- Batch processing supports large CSV files
- Results are cached for improved response times