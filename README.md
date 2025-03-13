# HHDE
HHDE: A Hyper-heuristic Differential Evolution with Novel Boundary Repair Technique for Complex Optimization

## Abstract
Inspired by the architecture of the hyper-heuristic (HH) algorithm, we design a mutation operator archive, a crossover operator archive, and a boundary repair operator archive to propose a novel hyper-heuristic differential evolution (HHDE). The mutation operator and crossover operator archives contain multiple representative search operators derived from variants of DE. A learning-free probabilistic selection function serves as the high-level component of the HHDE and is employed to determine the optimization sequence during optimization automatically. Additionally, we focus on the boundary repair operator, an element often overlooked in the design of the evolutionary algorithm (EA). Based on the previous research, our designed boundary repair operator archive introduces two novel boundary repair techniques: optimum inheritance and iterative opposite-based mapping. Comprehensive numerical experiments on CEC2017, CEC2020, and CEC2022 benchmark functions are conducted to evaluate the performance of our proposed HHDE. A range of other state-of-the-art optimizers and advanced variants of DE are employed as competitor algorithms. The experimental results and statistical analysis confirm the competitiveness and efficiency of HHDE. The source code of HHDE can be found in https://github.com/RuiZhong961230/HHDE.

## Citation
@article{Zhong:25,  
title = {HHDE: A Hyper-heuristic Differential Evolution with Novel Boundary Repair Technique for Complex Optimization},  
journal = {The Journal of Supercomputing},  
volume = {},  
pages = {},  
year = {2025},  
note = {Accepted},  
author = {Rui Zhong and Shilong Zhang and Jun Yu and Masaharu Munetomo},  
}  

## Datasets and Libraries
CEC benchmarks and Engineering problems are provided by opfunu==1.0.0 and enoppy==0.1.1 libraries, respectively. 

## Contact
If you have any questions, please don't hesitate to contact zhongrui[at]iic.hokudai.ac.jp
