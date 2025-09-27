// Credit Risk Prediction App JavaScript

// Check system health on page load
document.addEventListener('DOMContentLoaded', function() {
    checkHealth();
    setupEventListeners();
});

// Check system health
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        const healthStatus = document.getElementById('health-status');
        if (data.status === 'healthy') {
            if (data.model_loaded && data.scaler_loaded) {
                healthStatus.className = 'alert alert-success';
                healthStatus.innerHTML = '<strong>System Status:</strong> All systems operational ✅';
            } else {
                healthStatus.className = 'alert alert-warning';
                healthStatus.innerHTML = '<strong>System Status:</strong> Partially operational ⚠️';
            }
        } else {
            healthStatus.className = 'alert alert-danger';
            healthStatus.innerHTML = '<strong>System Status:</strong> System error ❌';
        }
    } catch (error) {
        const healthStatus = document.getElementById('health-status');
        healthStatus.className = 'alert alert-danger';
        healthStatus.innerHTML = '<strong>System Status:</strong> Unable to connect to server ❌';
    }
}

// Setup event listeners
function setupEventListeners() {
    // Single prediction form
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handleSinglePrediction);
    }
    

    
    // Real-time form validation
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('blur', validateField);
        input.addEventListener('input', clearFieldError);
    });
}

// Handle single prediction
async function handleSinglePrediction(event) {
    event.preventDefault();
    
    if (!validateForm('predictionForm')) {
        return;
    }
    
    const formData = new FormData(event.target);
    const submitButton = event.target.querySelector('button[type="submit"]');
    
    // Show loading state
    submitButton.innerHTML = '<span class="loading"></span> Predicting...';
    submitButton.disabled = true;
    hideResults();
    
    try {
        const response = await fetch('/predict/single', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            displaySingleResult(data);
        } else {
            displayError(data.message || 'Prediction failed');
        }
    } catch (error) {
        displayError('Network error: ' + error.message);
    } finally {
        // Reset button state
        submitButton.innerHTML = 'Predict Credit Risk';
        submitButton.disabled = false;
    }
}



// Display single prediction result
function displaySingleResult(data) {
    const resultsDiv = document.getElementById('results');
    const resultAlert = document.getElementById('resultAlert');
    const resultContent = document.getElementById('resultContent');
    
    const isHighRisk = data.prediction === 0;
    const riskColor = isHighRisk ? 'danger' : 'success';
    const riskIcon = isHighRisk ? '⚠️' : '✅';
    
    resultAlert.className = `alert alert-${riskColor}`;
    
    resultContent.innerHTML = `
        <div class="result-metric">
            <strong>Prediction:</strong> ${riskIcon} ${data.risk_level}
        </div>
        <div class="result-metric">
            <strong>Risk Probability:</strong> ${(data.probability * 100).toFixed(2)}%
        </div>
        <div class="result-metric">
            <strong>Recommendation:</strong> ${getRecommendation(data.prediction)}
        </div>
    `;
    
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}



// Display error message
function displayError(message) {
    const resultsDiv = document.getElementById('results');
    const resultAlert = document.getElementById('resultAlert');
    const resultContent = document.getElementById('resultContent');
    
    resultAlert.className = 'alert alert-danger';
    resultContent.innerHTML = `
        <div class="result-metric">
            <strong>Error:</strong> ${message}
        </div>
        <div class="result-metric">
            <strong>Recommendation:</strong> Please check your input data and try again. If the problem persists, contact support.
        </div>
    `;
    
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

// Hide results
function hideResults() {
    document.getElementById('results').style.display = 'none';
}

// Validate form
function validateForm(formId) {
    const form = document.getElementById(formId);
    const inputs = form.querySelectorAll('input[required], select[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!input.value) {
            input.classList.add('is-invalid');
            isValid = false;
        } else {
            input.classList.remove('is-invalid');
            input.classList.add('is-valid');
        }
    });
    
    return isValid;
}

// Validate individual field
function validateField(event) {
    const field = event.target;
    const value = field.value.trim();
    
    // Clear previous validation
    field.classList.remove('is-valid', 'is-invalid');
    
    // Basic validation
    if (field.hasAttribute('required') && !value) {
        field.classList.add('is-invalid');
        return false;
    }
    
    // Numeric validation
    if (field.type === 'number') {
        const numValue = parseFloat(value);
        const min = parseFloat(field.min);
        const max = parseFloat(field.max);
        
        if (isNaN(numValue) || (min && numValue < min) || (max && numValue > max)) {
            field.classList.add('is-invalid');
            return false;
        }
    }
    
    field.classList.add('is-valid');
    return true;
}

// Clear field error
function clearFieldError(event) {
    const field = event.target;
    if (field.classList.contains('is-invalid')) {
        field.classList.remove('is-invalid');
    }
}

// Get recommendation based on prediction
function getRecommendation(prediction) {
    if (prediction === 0) {
        return "High risk detected. Consider additional verification, collateral, or loan denial.";
    } else {
        return "Low risk profile. Standard loan approval process recommended.";
    }
}



// Auto-refresh health status every 30 seconds
setInterval(checkHealth, 30000);