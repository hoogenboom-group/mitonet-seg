# mitonet-seg
Automatic MitoNet[1] inference on Volume EM datasets in WebKnossos.

### Installation

- Clone repository (assumes `git` is installed)  
`git clone https://github.com/hoogenboom-group/mitonet-seg`

- Make a Python virtual environment:  
`python3.9 -m venv /path/to/new/virtual/environment`

- Activate:  
`source /path/to/new/virtual/environment/bin/activate`

- Install dependencies:    
`pip install -r requirements.txt`

### Usage
Edit `mitonet-inference.py` with desired parameters. Then run script by  
`python3 mitonet-inference.py`

### References
- [1]: Conrad, R., & Narayan, K. (2023). Instance segmentation of mitochondria in electron microscopy images with a generalist deep learning model trained on a diverse dataset. Cell Systems, 14(1), 58-71.
