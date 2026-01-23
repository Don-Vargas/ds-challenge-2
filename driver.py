from src.alternative_eda.ydata_eda import run_eda

if __name__ == "__main__":
    output_dir = run_eda()
    print(f"EDA reports generated in: {output_dir}")