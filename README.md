# 📈 MTF GUI (Modulation Transfer Function Analysis Tool)

The **MTF GUI** is a Python-based graphical application built to help engineers **analyse, visualise, and compare MTF data** across wafers and lenses.

It replaces manual and script-heavy workflows with an **interactive, configurable GUI**, allowing users to explore MTF performance efficiently and consistently.

---

## 🎯 Objective

To build a **user-friendly GUI** that:
- Displays MTF measurement parameters
- Visualises wafer-level MTF data
- Allows flexible configuration and reuse
- Supports analysis across **multiple wafers, frequencies, and lenses**

---

## 👷 Designed for Engineers

The GUI was designed with **engineering workflows in mind**:

- 🖱️ No command-line usage required  
- 📊 Interactive wafer maps and plots  
- 📁 Config files to reuse common setups  
- 🧠 Minimal technical knowledge needed to operate  
- 🪟 Ability to open **multiple wafer windows** simultaneously  

This allows engineers to focus on **analysis and decision-making**, not tooling.

---

## 🧠 Key Features

### 🔧 Main Window
Users can:
- View measured **MTF parameters**
- Adjust **S1–T9 and TFC values**
- Create, save, and load **configuration files**
- Select frequency numbers for CSV reference
- Open multiple wafer windows
- Toggle heat maps when multiple frequencies are selected

---

### 🧪 Wafer Window
Users can:
- Visualise wafer maps based on selected lens parameters
- Change colour mapping dynamically
- Adjust **upper and lower percentiles**
- Switch between:
  - Contoured wafer maps  
  - Pass / Fail wafer maps  
  - Heat maps (for multiple frequencies)

---

### 📤 Data Export
- Export **Pass / Fail wafer maps** to CSV
- Automatically generate timestamped output files
- Processed CSV files are saved consistently without manual handling

---

## ⚙️ How It Works (High-Level Flow)

1. User launches the GUI via batch file
2. DAT and CSV files are loaded from the working directory
3. Cleaned DAT files are generated automatically
4. User configures parameters via the Main Window
5. Selected frequencies open new Wafer Windows
6. Wafer maps, graphs, and analytics are rendered dynamically
7. Optional export to CSV for downstream analysis

---

## 🧩 Modular & Scalable Design

The application was structured to be:
- **Modular** — logic split across multiple scripts
- **Reusable** — configuration files prevent repeated setup
- **Scalable** — supports multiple wafers and windows concurrently

Key supporting modules include:
- DAT preprocessing
- CSV aggregation
- Spline analytics
- Wafer map generation
- Graph rendering

---

## 🧰 Tech Stack

- 🐍 Python  
- 🪟 Tkinter (GUI)  
- 📊 Pandas  
- 📈 Plotly  
- 🖼️ PIL  
- 🔢 NumPy  
- 🎨 colour  
- 📦 Pmw  
- 🧪 Kaleido  

---

## ⚠️ Limitations

- Wafer map loading can be slow for:
  - Large numbers of lenses
  - Many RC values
- No progress bar during long-running operations
- Performance depends on dataset size

---

## 📚 Challenges & Lessons Learned

### Challenges
- Designing complex GUIs with Tkinter
- Managing widget layout (frames vs canvas)
- Handling multiple active wafer windows
- Implementing configuration file logic
- Maintaining responsiveness with large datasets

### Key Learnings
- Modular code is essential for large GUI applications
- Clear separation between UI and logic improves maintainability
- Config-driven design saves significant user time
- GUI usability is just as important as algorithm correctness

---

## 🚀 Impact

- Standardised MTF analysis across engineers  
- Reduced manual data handling  
- Improved consistency and repeatability  
- Enabled faster exploration of wafer-level performance  

---

⭐ This tool was developed as part of an R&D internship to modernise and streamline MTF analysis workflows.
