# Uncertainty Quantification in Multivariable Regression for Material Property Prediction with Bayesian Neural Networks

[![arXiv](https://img.shields.io/badge/arXiv-2019.10435-b31b1b)](https://arxiv.org/abs/2311.02495)

With the increased use of data-driven approaches and machine learning-based methods in material science, the importance of reliable uncertainty quantification (UQ) of the predicted variables for informed decision-making cannot be overstated. UQ in material property prediction poses unique challenges, including the multi-scale and multi-physics nature of materials, intricate interactions between numerous factors, limited availability of large curated datasets, etc. In this work, we introduce a physics-informed Bayesian Neural Networks (BNNs) approach for UQ, which integrates knowledge from governing laws in materials to guide the models toward physically consistent predictions. To evaluate the approach, we present case studies for predicting the creep rupture life of steel alloys. Experimental validation with three datasets of creep tests demonstrates that this method produces point predictions and uncertainty estimations that are competitive or exceed the performance of conventional UQ methods such as Gaussian Process Regression. Additionally, we evaluate the suitability of the proposed approach for UQ in an active learning scenario and report competitive performance. The most promising framework for creep life prediction is BNNs based on Markov Chain Monte Carlo approximation of the posterior distribution of network parameters, as it provided more reliable results in comparison to BNNs based on variational inference approximation or related NNs with probabilistic outputs.

## Use
The codes are provided as Jupyter Notebook files. To reproduce the results, run the .ipynb files. 

## Requirements
keras  2.6.0  
tensorflow 2.6.0  
pyro-ppl 1.8.5  
torchbnn 1.2  
torch 2.0.1  
pandas 1.5.3  
numpy 1.23.5  
scikit-learn 1.2.2  

## Citation
If you use the codes or the methods in your work, please cite the following <a href="https://arxiv.org/abs/2311.02495">article</a>:   

    @ARTICLE{Li2023,
    title={Uncertainty Quantification in Multivariable Regression for Material Property Prediction with Bayesian Neural Networks},
    author={Longze Li and Jiang Chang and Aleksandar Vakanski and Yachun Wang and Tiankai Yao and Min Xian},
    year={2023},
    eprint={2311.02495},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
    }

## License
<a href="License - MIT.txt">MIT License</a>

## Acknowledgments
This work was supported by the University of Idaho – Center for Advanced Energy Study (CAES) Seed Funding FY23 Grant.

## Contact or Questions
<a href="https://www.webpages.uidaho.edu/vakanski/">A. Vakanski</a>, e-mail: vakanski at uidaho.edu
