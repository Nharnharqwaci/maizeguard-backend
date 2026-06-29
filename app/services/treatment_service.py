def get_treatment(prediction):

    prediction = (
        prediction.lower()
    )

    if prediction == "healthy":

        return [

            "Your maize crop appears healthy.",

            "Continue regular crop monitoring.",

            "Maintain proper irrigation practices.",

            "Apply fertilizer according to recommendations.",

            "Keep observing for early signs of disease."

        ]

    elif prediction == "msv":

        return [

            "Maize Streak Virus detected.",

            "Remove severely infected plants.",

            "Control leafhopper vectors.",

            "Plant resistant maize varieties.",

            "Monitor neighboring crops carefully."

        ]

    elif prediction == "not maize":

        return [

            "This does not appear to be a maize leaf.",

            "The current system supports maize leaf analysis only.",

            "Please upload a clear maize leaf image.",

            "Ensure the leaf occupies most of the image."

        ]

    else:

        return [

            "The prediction is uncertain.",

            "Please upload a clearer image.",

            "Ensure adequate lighting.",

            "Capture only a single maize leaf.",

            "Try another image for better results."

        ]