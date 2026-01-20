import matplotlib
matplotlib.use("Agg")  # 👈 IMPORTANT

from src.eda import eda_plots_analisys

def run_analysis():
    return eda_plots_analisys.analyze()