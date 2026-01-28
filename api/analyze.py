import matplotlib
matplotlib.use("Agg")

from src.eda import eda_plots_analisys

def run_analysis(version):
    return eda_plots_analisys.analyze_all(version)