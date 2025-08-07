# CordaX

> Connecting Beams, Reshaping Diffraction.

CordaX aims to be a general-purpose toolkit for handling complex scientific data, built on the principle of "connecting" disparate datasets to "reshape" them into meaningful insights.

At present, our development is centered on providing a high-performance solution for time-resolved ultrafast experiments at X-ray Free-Electron Lasers (XFELs). The toolkit is especially tailored for data generated at the **Pohang Accelerator Laboratory (PAL-XFEL)**.

# Project Heritage: Legacy System Attribution
## Legacy Codebase Origin

This project builds upon and refactors the legacy system **XFEL_data**, originally developed by:
- Name: Kim Jooheun
- GitHub: @paulpeterkim
- Role: Lead Developer & System Architect
- Active Years: 2021–2023
- Contact: paulpeterkim@gmail.com
---
# Usage
## Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- [Git](https://git-scm.com/) installed

## Setting Up the Environment

1. **Clone the repository:**
    ```bash
    git clone https://github.com/SJB7777/CordaX.git
    cd CordaX
    ```

2. **Create the virtual environment:**
    ```bash
    conda create -n cordax python=3.13
    conda activate cordax
    pip install -r requirements.txt
    ```

## Configuration Setup

1. Create a `config.yaml` file in your project root directory with the following content:

```yaml
runs:
  [136, 137, 143]

path:
  log_dir: ./logs

  load_dir: path\of\raw\data
  analysis_dir: path\of\analysis\result

  output_dir: ${path.analysis_dir}\output_data
  mat_dir: ${path.analysis_dir}\mat_files
  processed_dir: ${path.analysis_dir}\processed_data

param:
  pump_setting: 15HZ
  hutch: eh1
  detector: jungfrau2
  xray: HX
  y1: 0
  y2: 1
  x1: 2
  x2: 3
  sdd: 1.3
  dps: 7.5e-05
  beam_energy: 9.7
```

2. Modify the following paths according to your system setup:
   - `load_dir`: Path to raw HDF5 data files
   - `analysis_dir`: Root directory for analysis outputs
   - Adjust other paths as needed while maintaining the directory structure

## Integrating
Run the integrating script with:

```bash
python integrating_main.py
```

The script will:
1. Read configuration from `config.yaml`
2. Process specified runs (136, 137, 143)
3. Use detector parameters from the configuration
4. Save outputs in the specified directory structure
5. Store logs in `./logs`

## Analyzing
```bash
python analyzing_main.py
```

## Important Parameters
- `runs`: List of run numbers to process
- `detector`: Detector configuration (jungfrau2)
- `beam_energy`: X-ray beam energy in keV
- `sdd`: Sample-to-detector distance in meters
- Path configurations for input/output directories

Note: Ensure all path locations are accessible and have proper read/write permissions before running the script.

---
# Project Overview
The **PAL-XFEL Data Handling Toolkit** is a specialized suite of tools developed for the efficient processing and analysis of PAL-XFEL data, with a focus on time-resolved ultrafast experiments studying polaronic lattice distortions in Strontium Titanate (STO) nanocrystals. This toolkit is designed to enhance data handling, reduce processing times, and ensure the integrity of analytical results, making it an essential asset for our research on perovskite-oxide materials.

## File Structure of Raw Data
```bash
p0110.h5
├── detector
│   └── eh1
│      └── jungfrau2
│         └── image
├── metadata
├── oxc
├── pd
├── qbpm
│   └── eh1
│      └── qbpm1
└── oh
```

## Goals
- **Efficient Data Management**: Utilize HDF5 for structured storage and rapid data access, tailored for time-resolved datasets.
- **Accelerated Data Processing**: Implement parallel processing techniques to speed up I/O operations, critical for handling the rapid sequence of ultrafast experiments.
- **Data Quality Assurance**: Apply preprocessing methods to clean and standardize data for reliable analysis, ensuring precision in time-resolved measurements.
- **Automated Workflows**: Develop Python scripts to automate the data processing pipeline, from raw data to analytical outputs, streamlining the analysis of complex ultrafast phenomena.

## Future Enhancements
- **Scalability**: Continue to optimize the toolkit to handle increasing data volumes, ensuring it remains effective as experimental datasets grow.
- **Adaptability**: Ensure the tools can be adapted to various research needs, particularly those involving new time-resolved ultrafast techniques.
- **Collaboration**: Share insights and tools within our research community to foster collective progress in the field of ultrafast science.

## Getting Involved
This toolkit is primarily for internal use within our research group, but we welcome feedback and suggestions from our colleagues. Please reach out if you have ideas or wish to contribute to our ongoing development.
