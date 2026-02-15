# Rapido Analytics Platform

This project provides a comprehensive analytics and prediction platform for ride-hailing data, focusing on customer cancellations, driver delays, fare prediction, and ride outcomes. It is organized for modularity, scalability, and ease of use.

## Project Structure

```
Rapido/
│
├── README.md                # Project overview and instructions
├── app/                     # Streamlit app and UI pages
│   ├── app.py               # Main entry point for the Streamlit app
│   └── pages/               # Individual app pages
│       ├── AnalyticsExplorer.py
│       ├── CustomerCancellationPrediction.py
│       ├── DriverDelayPrediction.py
│       ├── FarePrediction.py
│       ├── HomePage.py
│       └── RideOutcomePrediction.py
│
├── data/                    # Data storage
│   ├── cleaned/             # Cleaned datasets
│   └── raw/                 # Raw datasets
│       ├── bookings.csv
│       ├── customers.csv
│       ├── drivers.csv
│       ├── location_demand.csv
│       └── time_features.csv
│
├── src/                     # Source code for models and pipelines
│   ├── __init__.py
│   ├── models/              # Model definitions
│   └── pipelines/           # ML pipelines
│       ├── CustomerCancellationRiskModel.py
│       ├── DriverDelayPredictionModel.py
│       ├── FarePredictionModel.py
│       └── RideOutcomePredictionModel.py
│
├── tests/                   # Unit tests for models
│   ├── CustomerCancellationRiskModelTest.py
│   ├── DriverDelayPredictionModelTest.py
│   ├── FarePredictionModelTest.py
│   └── RideOutcomePredictionModelTest.py
│
└── utils/                   # Utility scripts
	└── data_loader.py
```

## Getting Started

1. Clone the repository.
2. Install dependencies (see requirements.txt if available).
3. Run the app:
   ```bash
   streamlit run app/app.py
   ```

## Features
- Interactive analytics dashboards
- Predictive models for cancellations, delays, fares, and ride outcomes
- Modular and extensible codebase

## License
MIT License
