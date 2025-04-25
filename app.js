// Global variables
let trainedModel = false;
let trainingInProgress = false;
let audioRecorder = null;
let audioStream = null;
let recordedChunks = [];

// Track metrics across epochs
const trainingMetrics = {
    iterations: [],
    accuracy: [],
    val_accuracy: [],
    loss: [],
    val_loss: []
};

document.addEventListener('DOMContentLoaded', function() {
    // Initialize upload form event listeners
    const uploadForm = document.getElementById('uploadForm');
    uploadForm.addEventListener('submit', handleAudioAnalysis);

    // Initialize record button event listeners
    const recordButton = document.getElementById('recordButton');
    recordButton.addEventListener('click', toggleRecording);

    // Initialize new analysis button
    const newAnalysisBtn = document.getElementById('newAnalysisBtn');
    newAnalysisBtn.addEventListener('click', resetAnalysis);

    // Create dataset upload container and add it to the page
    createDatasetUploadContainer();
    
    // Add event listeners for advanced options if available
    const toggleAdvancedBtn = document.getElementById('toggleAdvancedBtn');
    if (toggleAdvancedBtn) {
        toggleAdvancedBtn.addEventListener('click', toggleAdvancedOptions);
    }
    
    // Check if we're resuming a previous training session
    if (document.getElementById('trainingProgress') && 
        !document.getElementById('trainingProgress').classList.contains('d-none')) {
        trainingInProgress = true;
        createMetricsContainer();
        
        // Connect to training updates stream
        setupTrainingEventSource();
    }
});

function setupTrainingEventSource() {
    const eventSource = new EventSource('http://localhost:8800/training-updates');
    
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        // Update progress bar
        const progressBar = document.getElementById('trainingProgressBar');
        progressBar.style.width = `${data.progress}%`;
        progressBar.textContent = `${data.progress}%`;
        
        // Add to training log
        const trainingLog = document.getElementById('trainingLog');
        if (data.log) {
            trainingLog.innerHTML += `<div>${data.log}</div>`;
            trainingLog.scrollTop = trainingLog.scrollHeight;
        }
        
        // Update charts
        if (data.accuracy !== undefined) {
            updateCharts(
                data.iteration || 0,
                data.accuracy || 0,
                data.val_accuracy || 0,
                data.loss || 0,
                data.val_loss || 0
            );
        }
        
        // Handle completion or errors
        if (data.status === 'complete' || data.status === 'error') {
            trainingInProgress = false;
            resetFormState();
            eventSource.close();
            
            if (data.status === 'complete') {
                trainedModel = true;
                notifyUser('success', 'Training Complete', 'Model training has finished successfully');
            } else {
                notifyUser('danger', 'Training Error', data.log || 'An error occurred during training');
            }
        }
    };
    
    eventSource.onerror = function(event) {
        console.error('SSE Error:', event);
        notifyUser('danger', 'Connection Error', 'Lost connection to training process');
        resetFormState();
        trainingInProgress = false;
        eventSource.close();
    };
}

function createDatasetUploadContainer() {
    // Create the dataset upload card
    const datasetCard = document.createElement('div');
    datasetCard.className = 'card shadow-sm mb-5';
    datasetCard.innerHTML = `
        <div class="card-body text-center">
            <h3><i class="fas fa-database text-primary"></i> Upload Training Dataset</h3>
            <p>Upload a zip file containing categorized baby cry audio samples</p>
            <form id="datasetUploadForm" enctype="multipart/form-data">
                <div class="mb-3">
                    <input type="file" class="form-control" id="datasetFile" name="dataset" accept=".zip">
                </div>
                <button type="submit" class="btn btn-primary" id="trainModelBtn">
                    <i class="fas fa-brain"></i> Train Model
                </button>
            </form>
            <div id="trainingProgress" class="mt-3 d-none">
                <div class="progress mb-2">
                    <div id="trainingProgressBar" class="progress-bar progress-bar-striped progress-bar-animated" 
                         role="progressbar" style="width: 0%"></div>
                </div>
                <div id="trainingLog" class="text-start small bg-light p-2 mt-2" 
                     style="max-height: 150px; overflow-y: auto; font-family: monospace;">
                </div>
            </div>
        </div>
    `;

    // Insert the dataset card before the row with the upload/record options
    const heroSection = document.querySelector('.hero-section');
    if (heroSection) {
        heroSection.parentNode.insertBefore(datasetCard, heroSection.nextSibling);
    }

    // Add event listener for dataset upload form
    const datasetUploadForm = document.getElementById('datasetUploadForm');
    datasetUploadForm.addEventListener('submit', handleDatasetUpload);
}

// This function handles dataset upload and model training
function handleDatasetUpload(event) {
    event.preventDefault();

    // Disable the normal audio upload and recording until training is complete
    document.getElementById('audioFile').disabled = true;
    document.getElementById('uploadBtn').disabled = true;
    document.getElementById('recordButton').disabled = true;
    document.getElementById('trainModelBtn').disabled = true;

    const datasetFile = document.getElementById('datasetFile').files[0];
    if (!datasetFile) {
        alert('Please select a dataset file to upload');
        resetFormState();
        return;
    }

    // Show training progress elements
    const trainingProgress = document.getElementById('trainingProgress');
    trainingProgress.classList.remove('d-none');
    const progressBar = document.getElementById('trainingProgressBar');
    progressBar.style.width = '0%';
    progressBar.textContent = '0%';

    const trainingLog = document.getElementById('trainingLog');
    trainingLog.innerHTML = "<div>Starting training process...</div>";
    
    trainingInProgress = true;

    // Create a FormData object to send the file
    const formData = new FormData();
    formData.append('dataset', datasetFile);
    
    // Create metrics container for displaying accuracy trends
    createMetricsContainer();

    // Send to backend for training
    fetch('http://localhost:8800/train-model', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        // Set up SSE for receiving training updates
        setupTrainingEventSource();
        return response.json();
    })
    .then(data => {
        console.log('Initial response:', data);
    })
    .catch(error => {
        console.error('Error:', error);
        notifyUser('danger', 'Upload Error', 'Failed to upload dataset: ' + error.message);
        resetFormState();
        trainingInProgress = false;
    });
}

// Training metrics visualization enhancement
function createMetricsContainer() {
    // Remove existing metrics container if it exists
    const existingContainer = document.getElementById('metricsContainer');
    if (existingContainer) {
        existingContainer.remove();
    }
    
    // Create metrics container with tabs for different visualizations
    const metricsContainer = document.createElement('div');
    metricsContainer.id = 'metricsContainer';
    metricsContainer.className = 'card shadow-sm mb-5 mt-3';
    metricsContainer.innerHTML = `
        <div class="card-body">
            <h4 class="mb-3">Training Metrics</h4>
            
            <ul class="nav nav-tabs" id="metricsTab" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="accuracy-tab" data-bs-toggle="tab" 
                            data-bs-target="#accuracy-tab-pane" type="button" role="tab" 
                            aria-controls="accuracy-tab-pane" aria-selected="true">
                        Accuracy
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="loss-tab" data-bs-toggle="tab" 
                            data-bs-target="#loss-tab-pane" type="button" role="tab" 
                            aria-controls="loss-tab-pane" aria-selected="false">
                        Loss
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="combined-tab" data-bs-toggle="tab" 
                            data-bs-target="#combined-tab-pane" type="button" role="tab" 
                            aria-controls="combined-tab-pane" aria-selected="false">
                        Combined
                    </button>
                </li>
            </ul>
            
            <div class="tab-content" id="metricsTabContent">
                <!-- Accuracy Chart -->
                <div class="tab-pane fade show active" id="accuracy-tab-pane" role="tabpanel" 
                     aria-labelledby="accuracy-tab" tabindex="0">
                    <div style="height: 300px;" class="mt-3">
                        <canvas id="accuracyChart"></canvas>
                    </div>
                </div>
                
                <!-- Loss Chart -->
                <div class="tab-pane fade" id="loss-tab-pane" role="tabpanel" 
                     aria-labelledby="loss-tab" tabindex="0">
                    <div style="height: 300px;" class="mt-3">
                        <canvas id="lossChart"></canvas>
                    </div>
                </div>
                
                <!-- Combined Chart -->
                <div class="tab-pane fade" id="combined-tab-pane" role="tabpanel" 
                     aria-labelledby="combined-tab" tabindex="0">
                    <div style="height: 300px;" class="mt-3">
                        <canvas id="combinedChart"></canvas>
                    </div>
                </div>
            </div>
            
            <!-- Real-time metrics summary -->
            <div class="row mt-4" id="metricsSummary">
                <div class="col-md-3">
                    <div class="card bg-light">
                        <div class="card-body text-center p-3">
                            <h6 class="card-title">Current Epoch</h6>
                            <h3 id="currentEpoch">0/50</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-light">
                        <div class="card-body text-center p-3">
                            <h6 class="card-title">Training Accuracy</h6>
                            <h3 id="currentAccuracy">0.00%</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-light">
                        <div class="card-body text-center p-3">
                            <h6 class="card-title">Validation Accuracy</h6>
                            <h3 id="currentValAccuracy">0.00%</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-light">
                        <div class="card-body text-center p-3">
                            <h6 class="card-title">Loss</h6>
                            <h3 id="currentLoss">0.00</h3>
                        </div>
                    </div>
                </div>
            </div>
            
            <button class="btn btn-sm btn-outline-secondary mt-3" id="exportMetricsBtn">
                Export Metrics (CSV)
            </button>
        </div>
    `;
    
    // Append after training progress
    const trainingProgress = document.getElementById('trainingProgress');
    if (trainingProgress) {
        trainingProgress.parentNode.insertBefore(metricsContainer, trainingProgress.nextSibling);
    }
    
    // Add export metrics button event listener
    const exportButton = document.getElementById('exportMetricsBtn');
    if (exportButton) {
        exportButton.addEventListener('click', exportMetricsCSV);
    }
    
    // Initialize charts
    initCharts();
}

function initCharts() {
    // Reset metrics
    trainingMetrics.iterations = [];
    trainingMetrics.accuracy = [];
    trainingMetrics.val_accuracy = [];
    trainingMetrics.loss = [];
    trainingMetrics.val_loss = [];
    
    // Initialize accuracy chart
    const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
    window.accuracyChart = new Chart(accuracyCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Accuracy',
                    data: [],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderWidth: 2,
                    tension: 0.1
                },
                {
                    label: 'Validation Accuracy',
                    data: [],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderWidth: 2,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Model Accuracy'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + (context.parsed.y * 100).toFixed(2) + '%';
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Accuracy'
                    },
                    min: 0,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            }
        }
    });
    
    // Initialize loss chart
    const lossCtx = document.getElementById('lossChart').getContext('2d');
    window.lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Loss',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 2,
                    tension: 0.1
                },
                {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: 'rgba(255, 159, 64, 1)',
                    backgroundColor: 'rgba(255, 159, 64, 0.2)',
                    borderWidth: 2,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Model Loss'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Loss'
                    },
                    min: 0
                }
            }
        }
    });
    
    // Initialize combined chart
    const combinedCtx = document.getElementById('combinedChart').getContext('2d');
    window.combinedChart = new Chart(combinedCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Accuracy',
                    data: [],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderWidth: 2,
                    tension: 0.1,
                    yAxisID: 'y'
                },
                {
                    label: 'Validation Accuracy',
                    data: [],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderWidth: 2,
                    tension: 0.1,
                    yAxisID: 'y'
                },
                {
                    label: 'Training Loss',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 2,
                    tension: 0.1,
                    yAxisID: 'y1'
                },
                {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: 'rgba(255, 159, 64, 1)',
                    backgroundColor: 'rgba(255, 159, 64, 0.2)',
                    borderWidth: 2,
                    tension: 0.1,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Model Training Progress'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            const datasetLabel = context.dataset.label;
                            const value = context.parsed.y;
                            
                            if (datasetLabel.includes('Accuracy')) {
                                return datasetLabel + ': ' + (value * 100).toFixed(2) + '%';
                            } else {
                                return datasetLabel + ': ' + value.toFixed(4);
                            }
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Epoch'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Accuracy'
                    },
                    min: 0,
                    max: 1,
                    position: 'left',
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                },
                y1: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Loss'
                    },
                    min: 0,
                    position: 'right',
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

function updateCharts(iteration, accuracy, val_accuracy, loss, val_loss) {
    // Add new metrics
    trainingMetrics.iterations.push(iteration);
    trainingMetrics.accuracy.push(accuracy);
    trainingMetrics.val_accuracy.push(val_accuracy);
    trainingMetrics.loss.push(loss);
    trainingMetrics.val_loss.push(val_loss);
    
    // Update accuracy chart
    window.accuracyChart.data.labels = trainingMetrics.iterations;
    window.accuracyChart.data.datasets[0].data = trainingMetrics.accuracy;
    window.accuracyChart.data.datasets[1].data = trainingMetrics.val_accuracy;
    window.accuracyChart.update();
    
    // Update loss chart
    window.lossChart.data.labels = trainingMetrics.iterations;
    window.lossChart.data.datasets[0].data = trainingMetrics.loss;
    window.lossChart.data.datasets[1].data = trainingMetrics.val_loss;
    window.lossChart.update();
    
    // Update combined chart
    window.combinedChart.data.labels = trainingMetrics.iterations;
    window.combinedChart.data.datasets[0].data = trainingMetrics.accuracy;
    window.combinedChart.data.datasets[1].data = trainingMetrics.val_accuracy;
    window.combinedChart.data.datasets[2].data = trainingMetrics.loss;
    window.combinedChart.data.datasets[3].data = trainingMetrics.val_loss;
    window.combinedChart.update();
    
    // Update summary cards
    document.getElementById('currentEpoch').textContent = `${iteration}/50`;
    document.getElementById('currentAccuracy').textContent = `${(accuracy * 100).toFixed(2)}%`;
    document.getElementById('currentValAccuracy').textContent = `${(val_accuracy * 100).toFixed(2)}%`;
    document.getElementById('currentLoss').textContent = loss.toFixed(4);
}

// Reset form state when training is complete or errors occur
function resetFormState() {
    document.getElementById('audioFile').disabled = false;
    document.getElementById('uploadBtn').disabled = false;
    document.getElementById('recordButton').disabled = false;
    document.getElementById('trainModelBtn').disabled = false;
}

// Function to display notifications to the user
function notifyUser(type, title, message) {
    const alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) {
        console.error('Alert container not found');
        return;
    }
    
    const alertId = 'alert-' + Date.now();
    const alertHTML = `
        <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show" role="alert">
            <strong>${title}</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    
    alertContainer.innerHTML += alertHTML;
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alertElement = document.getElementById(alertId);
        if (alertElement) {
            const bsAlert = new bootstrap.Alert(alertElement);
            bsAlert.close();
        }
    }, 5000);
}

// Function to toggle visibility of advanced options
function toggleAdvancedOptions() {
    const advancedOptions = document.getElementById('advancedOptions');
    const toggleButton = document.getElementById('toggleAdvancedBtn');
    
    if (advancedOptions.classList.contains('d-none')) {
        advancedOptions.classList.remove('d-none');
        toggleButton.textContent = 'Hide Advanced Options';
    } else {
        advancedOptions.classList.add('d-none');
        toggleButton.textContent = 'Show Advanced Options';
    }
}

// Function to export training metrics as CSV
function exportMetricsCSV() {
    if (!trainingMetrics || trainingMetrics.iterations.length === 0) {
        notifyUser('warning', 'Export Failed', 'No training metrics available to export');
        return;
    }
    
    // Create CSV content
    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "Epoch,Training Accuracy,Validation Accuracy,Training Loss,Validation Loss\n";
    
    for (let i = 0; i < trainingMetrics.iterations.length; i++) {
        const row = [
            trainingMetrics.iterations[i],
            trainingMetrics.accuracy[i],
            trainingMetrics.val_accuracy[i],
            trainingMetrics.loss[i],
            trainingMetrics.val_loss[i]
        ].join(",");
        csvContent += row + "\n";
    }
    
    // Create download link
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "training_metrics.csv");
    document.body.appendChild(link);
    
    // Download the CSV file
    link.click();
    
    // Clean up
    document.body.removeChild(link);
    
    notifyUser('success', 'Export Complete', 'Training metrics exported to CSV file');
}

async function simulateTraining(datasetName) {
    const iterations = 50;
    const progressBar = document.getElementById('trainingProgressBar');
    const trainingLog = document.getElementById('trainingLog');
    
    let baseAccuracy = 0.65;
    let bestAccuracy = 0.978;
    let baseLoss = 0.95;
    let bestLoss = 0.082;

    // Add dataset loading information
    trainingLog.innerHTML += `<div>Loading dataset: ${datasetName}</div>`;
    trainingLog.innerHTML += `<div>Extracting audio features from training samples...</div>`;
    await delay(1500);
    
    trainingLog.innerHTML += `<div>Preprocessing data and splitting into training/validation sets...</div>`;
    await delay(1000);
    
    trainingLog.innerHTML += `<div>Initializing model architecture...</div>`;
    await delay(800);
    
    trainingLog.innerHTML += `<div>Beginning training process (${iterations} iterations):</div>`;
    
    for (let i = 1; i <= iterations; i++) {
        await delay(100); // Small delay for each iteration
        
        // Calculate progress as a percentage
        const progress = i / iterations;
        const progressPct = Math.round(progress * 100);
        
        // Update progress bar
        progressBar.style.width = `${progressPct}%`;
        progressBar.textContent = `${progressPct}%`;
        
        // Calculate metrics with some randomness
        // Accuracy increases over time
        const accuracy = baseAccuracy + (bestAccuracy - baseAccuracy) * progress + 
                        (Math.random() * 0.02 - 0.01);
        const boundedAccuracy = Math.min(0.99, Math.max(baseAccuracy, accuracy));
        
        // Loss decreases over time
        const loss = baseLoss - (baseLoss - bestLoss) * progress + 
                   (Math.random() * 0.04 - 0.02);
        const boundedLoss = Math.max(bestLoss, Math.min(baseLoss, loss));
        
        // Validation metrics slightly worse than training
        const valAccuracy = boundedAccuracy - (Math.random() * 0.04 + 0.01);
        const valLoss = boundedLoss + (Math.random() * 0.02 + 0.01);
        
        if (i % 5 === 0 || i === 1 || i === iterations) {
            trainingLog.innerHTML += `<div>Iteration ${i}/${iterations} - Loss: ${boundedLoss.toFixed(4)} - Accuracy: ${boundedAccuracy.toFixed(4)} - Val Loss: ${valLoss.toFixed(4)} - Val Accuracy: ${valAccuracy.toFixed(4)}</div>`;
            trainingLog.scrollTop = trainingLog.scrollHeight;
        }
    }
    
    await delay(500);
    trainingLog.innerHTML += `<div>Optimizing model...</div>`;
    await delay(800);
    trainingLog.innerHTML += `<div>Saving model weights...</div>`;
    await delay(500);
    
    return true;
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function handleAudioAnalysis(event) {
    event.preventDefault();
    
    if (!trainedModel) {
        alert('Please train the model with a dataset before analyzing audio.');
        return;
    }
    
    const audioFile = document.getElementById('audioFile').files[0];
    if (!audioFile) {
        alert('Please select an audio file to analyze');
        return;
    }
    
    // Disable buttons during analysis
    document.getElementById('uploadBtn').disabled = true;
    document.getElementById('recordButton').disabled = true;
    
    // Show "analyzing" indicator
    const uploadBtn = document.getElementById('uploadBtn');
    const originalBtnText = uploadBtn.innerHTML;
    uploadBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
    
    // Create a FormData object and append the file
    const formData = new FormData();
    formData.append('file', audioFile);
    
    // Send to backend for analysis
    fetch('http://localhost:8800/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred during analysis');
    })
    .finally(() => {
        // Reset button state
        uploadBtn.innerHTML = originalBtnText;
        document.getElementById('uploadBtn').disabled = false;
        document.getElementById('recordButton').disabled = false;
    });
}

function toggleRecording() {
    const recordButton = document.getElementById('recordButton');
    const recordingStatus = document.getElementById('recordingStatus');
    
    if (!trainedModel) {
        alert('Please train the model with a dataset before recording audio.');
        return;
    }
    
    if (audioRecorder === null) {
        // Start recording
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                audioStream = stream;
                recordedChunks = [];
                
                audioRecorder = new MediaRecorder(stream);
                audioRecorder.ondataavailable = e => {
                    if (e.data.size > 0) {
                        recordedChunks.push(e.data);
                    }
                };
                
                audioRecorder.onstop = () => {
                    const audioBlob = new Blob(recordedChunks, { type: 'audio/wav' });
                    submitRecordedAudio(audioBlob);
                };
                
                audioRecorder.start();
                recordButton.innerHTML = '<i class="fas fa-stop"></i> Stop Recording';
                recordButton.classList.remove('btn-outline-primary');
                recordButton.classList.add('btn-danger');
                recordingStatus.classList.remove('d-none');
            })
            .catch(err => {
                console.error('Error accessing microphone:', err);
                alert('Could not access microphone. Please check permissions.');
            });
    } else {
        // Stop recording
        audioRecorder.stop();
        audioStream.getTracks().forEach(track => track.stop());
        audioRecorder = null;
        audioStream = null;
        
        // Update UI
        recordButton.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
        recordButton.classList.remove('btn-danger');
        recordButton.classList.add('btn-outline-primary');
        recordingStatus.classList.add('d-none');
        
        // Show "analyzing" indicator while processing
        recordButton.disabled = true;
        recordButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
    }
}

function submitRecordedAudio(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    
    fetch('http://localhost:8800/analyze-live', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred during analysis');
    })
    .finally(() => {
        // Reset button state
        const recordButton = document.getElementById('recordButton');
        recordButton.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
        recordButton.disabled = false;
    });
}

function displayResults(data) {
    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.classList.remove('d-none');
    
    // Populate prediction result
    const predictionResult = document.getElementById('predictionResult');
    predictionResult.textContent = data.prediction;
    
    // Set confidence level
    const confidenceResult = document.getElementById('confidenceResult');
    const confidencePct = Math.round(data.confidence * 100);
    confidenceResult.textContent = `Confidence: ${confidencePct}%`;
    
    // Set result icon based on prediction
    const resultIcon = document.getElementById('resultIcon');
    switch(data.prediction.toLowerCase()) {
        case 'hungry':
            resultIcon.className = 'fas fa-utensils text-primary fa-3x mb-3';
            break;
        case 'tired':
        case 'sleepy':
            resultIcon.className = 'fas fa-bed text-primary fa-3x mb-3';
            break;
        case 'belly pain':
        case 'pain':
        case 'stomach':
            resultIcon.className = 'fas fa-stomach text-primary fa-3x mb-3';
            break;
        case 'burping':
        case 'gas':
            resultIcon.className = 'fas fa-wind text-primary fa-3x mb-3';
            break;
        case 'discomfort':
            resultIcon.className = 'fas fa-frown text-primary fa-3x mb-3';
            break;
        default:
            resultIcon.className = 'fas fa-question-circle text-primary fa-3x mb-3';
    }
    
    // Set visualization image
    const audioVisualization = document.getElementById('audioVisualization');
    audioVisualization.src = `data:image/png;base64,${data.visualization}`;
    
    // Create probability chart
    createProbabilityChart(data.all_probabilities);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function createProbabilityChart(probabilities) {
    const ctx = document.getElementById('probabilityChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (window.probabilityChart) {
        window.probabilityChart.destroy();
    }
    
    // Sort probabilities from highest to lowest
    const sortedEntries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    const labels = sortedEntries.map(entry => entry[0]);
    const data = sortedEntries.map(entry => entry[1] * 100); // Convert to percentage
    
    // Generate colors array
    const colors = generateChartColors(labels.length);
    
    window.probabilityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability (%)',
                data: data,
                backgroundColor: colors,
                borderColor: colors.map(color => color.replace('0.7', '1')),
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Prediction Probabilities'
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Probability (%)'
                    }
                }
            }
        }
    });
}

function generateChartColors(count) {
    const baseColors = [
        'rgba(54, 162, 235, 0.7)',
        'rgba(255, 99, 132, 0.7)',
        'rgba(75, 192, 192, 0.7)',
        'rgba(255, 159, 64, 0.7)',
        'rgba(153, 102, 255, 0.7)'
    ];
    
    const colors = [];
    for (let i = 0; i < count; i++) {
        colors.push(baseColors[i % baseColors.length]);
    }
    
    return colors;
}

function resetAnalysis() {
    // Hide results section
    document.getElementById('resultsSection').classList.add('d-none');
    
    // Clear file input
    document.getElementById('audioFile').value = '';
    
    // Reset recording button if needed
    const recordButton = document.getElementById('recordButton');
    if (recordButton.classList.contains('btn-danger')) {
        // Force stop any ongoing recording
        if (audioRecorder !== null) {
            audioRecorder.stop();
            audioStream.getTracks().forEach(track => track.stop());
            audioRecorder = null;
            audioStream = null;
        }
        
        recordButton.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
        recordButton.classList.remove('btn-danger');
        recordButton.classList.add('btn-outline-primary');
        document.getElementById('recordingStatus').classList.add('d-none');
    }
}