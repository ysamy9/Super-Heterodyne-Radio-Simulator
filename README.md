Super-Heterodyne Receiver Simulator
Analog Communication Theory Project

Welcome! This repository contains a real-time Digital Signal Processing (DSP) application designed to simulate the architecture and functionality of a **Super-Heterodyne Radio Receiver**. This project was developed as a core component of our **Analog Communication Theory** course. 

The simulator provides a hands-on way to visualize how signals move through a receiver chain—from high-frequency "airwaves" down to audible baseband audio—supporting multiple modulation schemes and simulating real-world channel conditions like noise and distance-based attenuation.

Simulation Chain
This project isn't just a media player; it’s a full mathematical model of receiver hardware. The software replicates:
* **RF Stage:** Tuneable pre-selection filtering to isolate specific stations.
* **The Mixer & LO:** Frequency translation using a Local Oscillator to shift signals to a fixed Intermediate Frequency (IF).
* **IF Stage:** High-selectivity filtering at 25 kHz to eliminate adjacent channel interference.
* **Demodulation:** Specialized blocks for **AM (DSB-LC)** and **NBFM (Narrowband FM)**.
* **AGC (Automatic Gain Control):** Dynamic gain adjustment based on the "distance" of the transmitter to maintain consistent audio volume.

Features
* **Interactive Tuning:** A slider-based interface to scan frequencies between 80 kHz and 220 kHz.
* **Live Audio:** Real-time playback of demodulated signals using the `sounddevice` library.
* **Visual Analytics:** * View the **FDM Airwave** spectrum.
    * Monitor the signal at every stage (Mixer, IF, LPF).
    * Analyze the impact of **RF Bypass** and **LO Offset** on signal integrity.
* **Noise Simulation:** Toggle between raw channel noise and in-band filtered noise to see the "Pre-selector" effect in action.

  made by :
This project was a collaborative effort by:
* **Youssef Samy**
* **Shahd Ali**
* **Mahdi Ibrahim**

**Supervised by: Dr. Doaa Gamal | Dr. Samar Mokhtar
