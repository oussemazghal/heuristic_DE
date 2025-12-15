# Optimization Dashboard

A modern web application for comparing evolutionary algorithms (DE, jDE, PSO, PSO-H, GA) on CEC 2017 benchmark functions.

## Features
- **Algorithms**: Differential Evolution (DE), Self-Adaptive DE (jDE), Particle Swarm Optimization (PSO), Hybrid PSO (PSO-H), Genetic Algorithm (GA).
- **Benchmarks**: Supports CEC 2017 functions (requires `cec2017` package) with fallback to standard functions.
- **Visualization**: Interactive convergence curves (log scale) using Plotly.js.
- **Analysis**: Summary tables, ranking, and detailed solution inspection.
- **Tech Stack**: FastAPI (Backend), Vanilla HTML/CSS/JS (Frontend).

## Prerequisites
- Python 3.8+
- `pip`

## Installation

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn numpy pydantic
   # Optional: Install CEC2017 for full benchmark suite
   # pip install cec2017
   ```

## Running the Application

1. Navigate to the project root directory.
2. Run the FastAPI server:
   ```bash
   uvicorn backend.main:app --reload
   ```
3. Open your browser and visit:
   ```
   http://127.0.0.1:8000
   ```

## Project Structure
```
.
├── backend/
│   ├── main.py          # FastAPI app entry point
│   ├── algorithms.py    # Optimization algorithms logic
│   └── models.py        # Pydantic data models
├── frontend/
│   ├── index.html       # Main UI
│   ├── style.css        # Styling
│   └── script.js        # Frontend logic
└── README.md
```
