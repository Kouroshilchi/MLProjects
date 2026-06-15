import sys
sys.path.append("../")
import kagglehub

path = kagglehub.dataset_download("uciml/adult-census-income" , path="AdultCensusIncomeDataset")

print("Path to dataset files:", path)