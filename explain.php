<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>How It Works - Baby Cry Classifier</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-light">
        <div class="container">
            <a class="navbar-brand" href="index.php">
                <i class="fas fa-baby text-primary"></i> Baby Cry Classifier
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="index.php">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link active" href="explain.php">How It Works</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="card shadow-sm mb-5">
                    <div class="card-body">
                        <h1 class="card-title">How Our Baby Cry Classifier Works</h1>
                        
                        <div class="mt-4">
                            <h3>The Science Behind Baby Cries</h3>
                            <p>
                                Babies communicate their needs through different types of cries. Research has shown that these cries have 
                                distinct acoustic patterns that can be identified and classified. Our AI system has been trained to recognize 
                                these patterns and associate them with different needs.
                            </p>
                        </div>
                        
                        <div class="row mt-5">
                            <div class="col-md-6">
                                <h3>Our Technology</h3>
                                <div class="technology-step mb-4">
                                    <div class="step-number">1</div>
                                    <h5>Audio Preprocessing</h5>
                                    <p>We extract the sound wave from your audio file or recording and prepare it for analysis.</p>
                                </div>
                                
                                <div class="technology-step mb-4">
                                    <div class="step-number">2</div>
                                    <h5>Feature Extraction</h5>
                                    <p>Our system analyzes the audio to extract key features such as:</p>
                                    <ul>
                                        <li>MFCCs (Mel-Frequency Cepstral Coefficients)</li>
                                        <li>Spectral Centroid</li>
                                        <li>Zero Crossing Rate</li>
                                        <li>Chroma Features</li>
                                    </ul>
                                </div>
                                
                                <div class="technology-step">
                                    <div class="step-number">3</div>
                                    <h5>Classification</h5>
                                    <p>A machine learning model analyzes these features and classifies the cry into one of five categories.</p>
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <h3>Cry Categories</h3>
                                <div class="accordion" id="cryAccordion">
                                    <div class="accordion-item">
                                        <h2 class="accordion-header">
                                            <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#collapseHungry">
                                                <i class="fas fa-utensils me-2 text-primary"></i> Hungry
                                            </button>
                                        </h2>
                                        <div id="collapseHungry" class="accordion-collapse collapse show" data-bs-parent="#cryAccordion">
                                            <div class="accordion-body">
                                                Hungry cries tend to be rhythmic and repetitive. They usually start softly and build in intensity.
                                                The cry often follows a pattern of "neh, neh" sounds and may be accompanied by rooting or sucking motions.
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="accordion-item">
                                        <h2 class="accordion-header">
                                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseTired">
                                                <i class="fas fa-bed me-2 text-primary"></i> Tired
                                            </button>
                                        </h2>
                                        <div id="collapseTired" class="accordion-collapse collapse" data-bs-parent="#cryAccordion">
                                            <div class="accordion-body">
                                                Tired cries are often nasal and whiny. They may be accompanied by eye rubbing,
                                                yawning, or a distant stare. The cry might sound like an "owh, owh" sound and generally
                                                increases in intensity if the baby isn't helped to sleep.
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="accordion-item">
                                        <h2 class="accordion-header">
                                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseBellyPain">
                                                <i class="fas fa-stomach me-2 text-primary"></i> Belly Pain
                                            </button>
                                        </h2>
                                        <div id="collapseBellyPain" class="accordion-collapse collapse" data-bs-parent="#cryAccordion">
                                            <div class="accordion-body">
                                                Belly pain cries are typically more intense and higher pitched. They often start suddenly
                                                and may be accompanied by the baby drawing up their legs or clenching their fists.
                                                The sound may resemble an "eairh" sound and can be more urgent than other cries.
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="accordion-item">
                                        <h2 class="accordion-header">
                                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseBurping">
                                                <i class="fas fa-wind me-2 text-primary"></i> Burping
                                            </button>
                                        </h2>
                                        <div id="collapseBurping" class="accordion-collapse collapse" data-bs-parent="#cryAccordion">
                                            <div class="accordion-body">
                                                Burping cries are often short bursts of fussiness that resolve quickly after the baby
                                                releases gas. They may sound like "eh, eh" sounds and are often accompanied by a
                                                squirming or uncomfortable body posture.
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="accordion-item">
                                        <h2 class="accordion-header">
                                            <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseDiscomfort">
                                                <i class="fas fa-frown me-2 text-primary"></i> Discomfort
                                            </button>
                                        </h2>
                                        <div id="collapseDiscomfort" class="accordion-collapse collapse" data-bs-parent="#cryAccordion">
                                            <div class="accordion-body">
                                                Discomfort cries vary in intensity but often indicate the baby is uncomfortable due to
                                                temperature, wet diaper, restrictive clothing, or position. These cries might stop
                                                temporarily when the baby is distracted but resume when the distraction ends.
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="mt-5">
                            <h3>Training Data</h3>
                            <p>
                                Our model was trained on the Donate-a-Cry corpus, a dataset containing thousands of baby cry samples
                                labeled by experienced caregivers and pediatricians. This extensive training allows our system to
                                recognize patterns across different babies and recording conditions.
                            </p>
                        </div>
                        
                        <div class="mt-5 text-center">
                            <h3>Important Note</h3>
                            <div class="alert alert-info">
                                <i class="fas fa-info-circle me-2"></i>
                                While our tool can help identify possible reasons for your baby's cry, it should be used as
                                an assistive tool only. Always trust your parental instincts and consult healthcare professionals
                                when you're concerned about your baby's well-being.
                            </div>
                            
                            <a href="/" class="btn btn-primary mt-3">
                                <i class="fas fa-arrow-left me-2"></i> Back to Analyzer
                            </a>
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

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>