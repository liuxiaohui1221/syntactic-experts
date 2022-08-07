# ECSpell
Code for paper "General and Domain Adaptive Chinese Spelling Check with Error Consistent Pretraining"
## Data usage
Path: `Data/domains_data`
- For zero-shot tasks, you should combine the *.train file and *.test file.
- For common tasks, the *.train file is used to do training and do evaluating while *.test is adopted to do predicting.
## Usage:
``` shell
cd glyce
python setup.py develop
pip show glyce   # to ensure the successful installation of glyce lib
```
[Model weights](https://drive.google.com/file/d/1HlfDbMpXR6YHiBuJS8s_K3ZKG6j0fvc5/view?usp=sharing)