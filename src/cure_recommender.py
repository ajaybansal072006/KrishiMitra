def recommend_cure(disease_name):
    """
    Suggests treatments and prevention tips for detected disease.
    """
    recommendations = {
        "Healthy": [
            "Maintain current watering and fertilizer routine.",
            "Ensure sunlight 6–8 hours daily."
        ],
        "Leaf_Blight": [
            "Spray Mancozeb 2g/L water every 10 days.",
            "Avoid overhead irrigation.",
            "Remove infected leaves immediately."
        ],
        "Rust": [
            "Apply Propiconazole (Tilt) 1ml/L water.",
            "Ensure proper plant spacing.",
            "Spray Sulfur Dust to reduce spread."
        ],
        "Mosaic": [
            "Control aphids using neem oil spray.",
            "Use virus-free certified seeds.",
            "Destroy infected plants early."
        ]
    }
    return recommendations.get(disease_name, ["No specific recommendation available."])
