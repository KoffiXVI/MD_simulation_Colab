# MD_simulation_Colab
Pipeline for duplex MD simulations leveraging colab interactions with google drive

# For a local machine

Clone the repository as follows:
```sh
git clone <this-repository-url>
cd <repository-name>
```

On a local machine, install all dependencies using the provided .yaml file

````sh
conda env create -f dependencies.yaml
conda activate md_sim
````

# For Google Colab

In your Colab session, clone the repository as follows:
```sh
git clone <this-repository-url>
cd <repository-name>
```

For a Colab environment, install the packages using the following command:

````sh
! <manager> install -y -c conda-forge openmm "cuda-version=13.0" openmmforcefields ambertools mdtraj pandas scipy pdbfixer  
````

`cuda version 13.0` was selected for compatibility with the Colab env that was used at the time of production. Adapt it to your env if need be

`<manager>`here refers to the package manager you prefer to use for installation (i.e mamba, conda). To use mamba in google colab, you should install condacolab first at the top of your notebook:

```sh
try:
    import condacolab
    condacolab.check()
except ImportError:
    !pip install -q condacolab
    import condacolab
    condacolab.install()
```

Inside a Colab env, after setting up your drive as the working directory, you'll need to preface the function import calls with their parent folder name:

Instead of:
```sh
import functions.colab_funcs as colab_funcs
import functions.constants
```

You get:
```sh
import MD_simulation_Colab.functions.colab_funcs as colab_funcs
import MD_simulation_Colab.functions.constants
``` 

