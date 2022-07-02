import numpy as np
import pandas as pd
from os import listdir
from os import walk
import os


EPOCHS = 100
BATCHSIZE =1024
PATH = os.getcwd()+'/Data/'
nrows = 5000


if __name__ == '__main__':
    
    
    print("1. Preprocessing Rajkumar2021 Paper")
    print("2. Test Rajkumar2021 Paper")
    print("3. Preprocessing Abdullah 2021 Paper")
    print("4. Test Abdullah 2021 Paper")
    print("5. Preprocessing Devrim 2022 Paper")
    print("6. Test Devrim 2022 Paper")
    print("7. Proposed Preprocessing: Create Dataset")
    print("8. Create PCA Dataset using processed dataset")
    print("9. Visualiation")
    print("10. Proposed Model")

   
    selection = int(input("Enter your implemention checker: "))
    
    if selection == 1:
        from RajKumar2021.data_utils import clean_dataset
        clean_dataset(PATH, nrows)
    
    elif selection ==2:
        from RajKumar2021.model import test_model
        test_model(PATH, nrows)

    elif selection == 3:
        from Abdullah2021.data_utils import clean_dataset
        clean_dataset(PATH, nrows)
    
    elif selection ==4:
        from Abdullah2021.model import test_model
        test_model(PATH, EPOCHS, BATCHSIZE, nrows)
        
    elif selection ==5:
        from Devrim2022.data_utils import clean_dataset
        clean_dataset(PATH, nrows)
    
    elif selection ==6:
        from Devrim2022.model import test_model
        test_model(PATH,EPOCHS, BATCHSIZE, nrows)
        
    elif selection ==7:
        from Proposed.data_utils import make_datasets
        make_datasets(PATH, nrows)
    
    
    elif selection ==8:
        from Proposed.data_utils import create_pca_dataset
        create_pca_dataset()
        
    elif selection ==9:
        from Proposed.data_utils import load_pca_dataset
        from vis_utils import plot_files_preprocessing, class_distributions
        from Proposed.pca_analysis import principal_component_varience, plot_pca_analysis
        print("1. Plotting Files Size Preprocessing")
        print("2. Plotting Preprocesses Class Distribution")
        print("3. Plotting Principal Component Varience")
        print("4. Plotting PCA Analysis Plot")

        dataset = load_pca_dataset()
        plot_files_preprocessing()
        class_distributions(dataset)
        plot_pca_analysis(dataset)
        principal_component_varience()
        
    elif selection ==10:
        from Proposed.train import trainer
        trainer(EPOCHS, BATCHSIZE)
       
            
        
       