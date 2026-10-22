# SMARTCARB Dataset & Preprocessing Guide

This document describes the synthetic satellite dataset, noise generation pipeline, and partitioning protocol used in our research manuscript:
> **"Restoration-Aware Joint Learning for Robust CO₂ Plume Segmentation from Noisy XCO₂ Observations"** (*Computers & Geosciences*).

---

## 1. Dataset Origin & Description

The experimental evaluation is based on synthetic observations from the **SMARTCARB** dataset developed within the European Space Agency (ESA)-funded SMARTCARB project:
* **Mission Concept:** Copernicus Anthropogenic Carbon Dioxide Monitoring (**CO2M**) constellation.
* **Atmospheric Modeling:** High-resolution simulations produced with the **COSMO-GHG** atmospheric transport model at approximately $1\,\text{km} \times 1\,\text{km}$ horizontal grid spacing.
* **Geographic Coverage:** European domain centered over Berlin and surrounding industrialized regions in Germany and neighboring countries.
* **Satellite Sampling:** Resampled along simulated polar sun-synchronous orbit swaths at $2\,\text{km} \times 2\,\text{km}$ ground footprint resolution, covering approximately $250\,\text{km}$ swath widths.
* **Primary Physical Variable:** Column-averaged dry-air mole fraction of carbon dioxide ($\text{XCO}_2$), expressed in parts per million ($\text{ppm}$).

---

## 2. Five Evaluated Geographic Scenarios

The framework was systematically validated across five distinct emission scenarios exhibiting heterogeneous plume morphology and background regimes:

| Region | Source Type | Characteristics | Background Regime |
|---|---|---|---|
| **Berlin** | Megacity / Urban Area | Diffuse, multi-point urban plume complex | Continental background ($\sim 402\,\text{ppm}$) |
| **Île-de-France (Paris)**| Megacity / Urban Area | Sprawling metropolitan footprint with complex radial dispersion | Variable oceanic-continental transition |
| **Boxberg** | Coal-fired Power Plant | Intensive localized point source with sharp core enhancements | Strong gradient with localized advection |
| **Lippendorf** | Lignite Power Plant | Point source plume in complex undulating terrain | Moderate regional background variability |
| **Jänschwalde** | Thermal Power Station | Large point-source plume with elongated downstream trajectories | High contrast core fading into ambient levels |

---

## 3. Observation Tile Sampling & Noise Model

### Spatial Cropping
* Full swath soundings are segmented into regular spatial patches of size **$160 \times 160$ pixels**.
* For testing spatial compatibility, input dimensions should be divisible by 4 (to match the two downsampling stages in the ConvTransformer backbone).

### Synthetic Instrument Noise
In satellite remote sensing, individual sounding retrievals from Short-Wave Infrared (SWIR) spectrometers suffer from detector noise, surface albedo variation, and optical scattering. Following CO2M instrument requirements:
$$\mathbf{X} = X^{\text{clean}} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)$$
where $\sigma_{\text{noise}} \in [0.5, 0.7]\,\text{ppm}$, matching the anticipated single-sounding precision of CO2M spectrometers.

### Physical Denormalization Parameters
Inside the physics-consistency loss module, normalized network predictions are mapped back to absolute physical mole fraction ($\text{ppm}$) using empirical statistics:
$$X_{\text{phys}} = \hat{\mathbf{X}} \cdot 3.42 + 402.0\,\text{ppm}$$
* Target mean baseline: $402.0\,\text{ppm}$
* Target scaling factor: $3.42\,\text{ppm}$
* Ambient background threshold: $c_{\text{thresh}} = 402.0\,\text{ppm}$

---

## 4. Strict Partitioning & Data Leakage Prevention

To ensure rigorous validation:
1. **Scene-Level Separation:** All observation scenes, simulation dates, and regional tiles corresponding to the evaluation test scenes (Berlin, Île-de-France, Boxberg, Lippendorf, Jänschwalde) were strictly withheld from denoiser pretraining.
2. **Pretraining Corpus:** Pretraining of `EfficientDenoiser` was conducted exclusively on designated training partitions of the SMARTCARB archive.
3. **Evaluation Protocol:** All metrics (NWBCE, IoU, Precision, Recall) are reported on unseen test scenes across all 5 evaluation regions.

---

## 5. Public SMARTCARB Archive Access

The simulated SMARTCARB observation datasets are openly available for research through:
* **ESA SMARTCARB Project Archive:** [https://doi.org/10.5281/zenodo.4046727](https://doi.org/10.5281/zenodo.4046727)
* **Kuhlmann et al. (2021):** *Quantifying CO2 and NO2 emissions from simulated satellite observations*, Earth System Science Data.
