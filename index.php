<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Baby Cry Classifier</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="./style.css">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-light">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-baby text-primary"></i> Baby Cry Classifier
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/explain.php">How It Works</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="hero-section text-center mb-5">
                    <h1 class="display-4">Decode Your Baby's Cries</h1>
                    <p class="lead">Our AI model analyzes baby cries to help you understand what your baby needs.</p>
                </div>

                <div class="row mb-5">
                    <div class="col-md-6 mb-4">
                        <div class="card shadow-sm h-100">
                            <div class="card-body text-center">
                                <h3><i class="fas fa-upload text-primary"></i> Upload Audio</h3>
                                <p>Upload a recording of your baby's cry</p>
                                <form id="uploadForm" enctype="multipart/form-data">
                                    <div class="mb-3">
                                        <input type="file" class="form-control" id="audioFile" name="file" accept="audio/*">
                                    </div>
                                    <button type="submit" class="btn btn-primary" id="uploadBtn">
                                        <i class="fas fa-check"></i> Analyze
                                    </button>
                                </form>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6 mb-4">
                        <div class="card shadow-sm h-100">
                            <div class="card-body text-center">
                                <h3><i class="fas fa-microphone text-primary"></i> Live Recording</h3>
                                <p>Record your baby's cry directly</p>
                                <div class="d-grid gap-2">
                                    <button id="recordButton" class="btn btn-outline-primary">
                                        <i class="fas fa-microphone"></i> Start Recording
                                    </button>
                                    <div id="recordingStatus" class="mt-2 d-none">
                                        <div class="spinner-grow text-danger" role="status">
                                            <span class="visually-hidden">Recording...</span>
                                        </div>
                                        <span class="ms-2">Recording...</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Results Section (Initially Hidden) -->
                <div id="resultsSection" class="card shadow-sm mb-5 d-none">
                    <div class="card-body">
                        <h3 class="card-title text-center mb-4">Analysis Results</h3>
                        
                        <div class="row">
                            <div class="col-md-6">
                                <div class="result-box text-center p-4 mb-4">
                                    <h4>Prediction</h4>
                                    <div class="prediction-result">
                                        <i id="resultIcon" class="fas fa-3x mb-3"></i>
                                        <h2 id="predictionResult" class="mb-2"></h2>
                                        <p id="confidenceResult"></p>
                                    </div>
                                </div>
                                
                                <div class="chart-container">
                                    <canvas id="probabilityChart"></canvas>
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <h4 class="text-center mb-3">Audio Analysis</h4>
                                <div id="visualizationContainer" class="text-center">
                                    <img id="audioVisualization" class="img-fluid" src="" alt="Audio visualization">
                                </div>
                            </div>
                        </div>
                        
                        <div class="text-center mt-4">
                            <button id="newAnalysisBtn" class="btn btn-outline-primary">
                                <i class="fas fa-redo"></i> New Analysis
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="row mb-5">
                    <div class="col-12">
                        <div class="card shadow-sm">
                            <div class="card-body">
                                <h3 class="card-title">What Your Baby's Cries Mean</h3>
                                <div class="row">
                                    <div class="col-md-4 mb-3">
                                        <div class="cry-type-card">
                                            <i class="fas fa-utensils text-primary"></i>
                                            <h5>Hungry Cry</h5>
                                            <p>Rhythmic, repetitive sound that may increase in intensity.</p>
                                        </div>
                                    </div>
                                    <div class="col-md-4 mb-3">
                                        <div class="cry-type-card">
                                            <i class="fas fa-bed text-primary"></i>
                                            <h5>Tired Cry</h5>
                                            <p>Nasal, whiny sound, often accompanied by eye rubbing or yawning.</p>
                                        </div>
                                    </div>
                                    <div class="col-md-4 mb-3">
                                        <div class="cry-type-card">
                                            <i class="fas fa-stomach text-primary"></i>
                                            <h5>Belly Pain</h5>
                                            <p>Intense cry that starts suddenly, often with legs drawing up.</p>
                                        </div>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <div class="cry-type-card">
                                            <i class="fas fa-wind text-primary"></i>
                                            <h5>Burping</h5>
                                            <p>Fussiness that resolves quickly after releasing gas.</p>
                                        </div>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <div class="cry-type-card">
                                            <i class="fas fa-frown text-primary"></i>
                                            <h5>Discomfort</h5>
                                            <p>Variable intensity, may be related to temperature, diaper, or position.</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer class="footer mt-auto py-3 bg-light">
        <div class="container text-center">
            <span class="text-muted">© 2025 Baby Cry Classifier - Helping Parents Understand Their Babies</span>
        </div>
    </footer>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="app.js"></script>
</body>
</html>