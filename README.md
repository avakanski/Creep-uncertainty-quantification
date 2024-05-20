# Uncertainty Quantification in Multivariable Regression for Material Property Prediction with Bayesian Neural Networks

[![arXiv](https://img.shields.io/badge/arXiv-2311.02495-b31b1b)](https://arxiv.org/abs/2311.02495) [![Scientific Reports](https://img.shields.io/badge/Scientific_Reports-DOI%3A_10.1038/s41598-024-61189-x-brightgreen.svg)](https://doi.org/10.1038/s41598-024-61189-x)

With the increased use of data-driven approaches and machine learning-based methods in material science, the importance of reliable uncertainty quantification (UQ) of the predicted variables for informed decision-making cannot be overstated. UQ in material property prediction poses unique challenges, including the multi-scale and multi-physics nature of materials, intricate interactions between numerous factors, limited availability of large curated datasets, etc. 

In this work, we introduce a physics-informed Bayesian Neural Networks (BNNs) approach for UQ, which integrates knowledge from governing laws in materials to guide the models toward physically consistent predictions. To evaluate the approach, we present case studies for predicting the creep rupture life of steel alloys. Experimental validation with three datasets of creep tests demonstrates that this method produces point predictions and uncertainty estimations that are competitive or exceed the performance of conventional UQ methods such as Gaussian Process Regression. Additionally, we evaluate the suitability of the proposed approach for UQ in an active learning scenario and report competitive performance. The most promising framework for creep life prediction is BNNs based on Markov Chain Monte Carlo approximation of the posterior distribution of network parameters, as it provided more reliable results in comparison to BNNs based on variational inference approximation or related NNs with probabilistic outputs.

## 📁 Repository Organization
This repository is organized as follows:
- **Uncertainty Quantification directory**: contains codes related to the application of Machine Learning methods for predicting creep rupture life of steel alloy materials. The objective is to calculate both single-point estimates and uncertainty estimates for the predicted creep rupture life. The codes provide implementation of 8 Machine Learning methods, which include conventional Machine Learning approaches (Quantile Regression, Natural Gradient Boosting Regression, Gaussian Process Regression), Neural Networks-based approaches with deterministic network parameters (Deep Ensemble, Monte Carlo Dropout), and Neural Network-based approaches with probabilistic network parameters (Variational Inference BNNs, Markov Chain Monte Carlo BNNs). For comparison, the implementation codes for standard Neural Networks with deterministic network parameters that output single-point predictions are also provided.
- **Physics-Informed Machine Learning directory**: contains code for predicting creep rupture life via Physics-Informed Machine Learning methods based on integrating knowledge from governing laws for creep modeling into data-driven approaches. Implementation codes are provided for the best-performing methods for uncertainty quantification, which include Gaussian Process Regression, Variational Inference BNNs, and Markov Chain Monte Carlo BNNs. For comparison, the implementation codes for standard Neural Networks with deterministic network parameters are also provided.
- **Active Learning directory**: contains code for an Active Learning task for predicting creep rupture life, where the learning objective is to iteratively select data points with the highest epistemic uncertainty and diversity for faster model training with fewer data points. Implementation codes are provided for the best-performing methods for uncertainty quantification, including Gaussian Process Regression, Variational Inference BNNs, and Markov Chain Monte Carlo BNNs.

## 📊 Data and Evaluation Metrics
The implemented methods for predicting creep rupture life are evaluated on three creep datasets: 
- Stainless Steel (SS) 316 alloys dataset (from the National Institute for Materials Science), containing 617 samples with 20 features per sample.
- Nickel-based superalloys dataset (from <a href="https://www.sciencedirect.com/science/article/pii/S0927025622000386">Han et al., 2022</a>), containing 153 samples with 15 features per sample.
- Titanium alloys dataset (from <a href="https://pubs.aip.org/aip/aml/article/1/1/016102/2878729/Machine-learning-assisted-interpretation-of-creep">Swetlana et al., 2023</a>), containing 177 samples with 24 features per sample.

The set of used performance metrics for evaluating the implemented methods include predictive accuracy metrics (Pearson Correlation Coefficient, $R^2$ Coefficient of Determination, Root-mean-square Deviation, Mean Absolute Error) and uncertainty quantification metrics (Coverage, Mean Interval Width, Composite Metric).

## ▶️ Use
The codes are provided as Jupyter Notebook files. To reproduce the results, run the .ipynb files. 

## 🔨 Requirements
keras  2.6.0  
tensorflow 2.6.0  
pyro-ppl 1.8.5  
torchbnn 1.2  
torch 2.0.1  
pandas 1.5.3  
numpy 1.23.5  
scikit-learn 1.2.2  

## 📖 Citation
If you use the codes or the methods in your work, please cite the following <a href="https://arxiv.org/abs/2311.02495">article</a>:   

    @ARTICLE{Li2024,
    TITLE={Uncertainty Quantification in Multivariable Regression for Material Property Prediction with Bayesian Neural Networks},
    AUTHOR={Longze Li and Jiang Chang and Aleksandar Vakanski and Yachun Wang and Tiankai Yao and Min Xian},
    JOURNAL = {Scientific Reports},
    YEAR = {2024},
    VOLUME = {14},
    ISSUE = {1},
    ARTICLE-NUMBER = {10543},
    URL = {https://doi.org/10.1038/s41598-024-61189-x},
    ISSN = {2045-2322},
    DOI = {10.1038/s41598-024-61189-x}
    }

## 🚩 License
<a href="License - MIT.txt">MIT License</a>

## 👏 Acknowledgments
This work was supported by the University of Idaho – Center for Advanced Energy Study (CAES) Seed Funding FY23 Grant. This work was also supported through the INL Laboratory Directed Research & Development
(LDRD) Program under DOE Idaho Operations Ofce Contract DE-AC07-05ID14517 (project tracking number 23A1070-069FP). 

## ✉️ Contact or Questions
<a href="https://www.webpages.uidaho.edu/vakanski/">A. Vakanski</a>, e-mail: vakanski at uidaho.edu

