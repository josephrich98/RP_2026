# RP_2026

This is the GitHub repository containing all code for the preprint  
**pupl**  
by *Joseph Matthew Rich and Lior Pachter*.

**Preprint:** [pupl](DOI link to be added)

**pupl:** [GitHub - pachterlab/pupl](https://github.com/pachterlab/pupl.git)

## Getting Started

To run the code in this repository, follow these steps:

```sh
git clone https://github.com/pachterlab/RP_2026.git
cd RP_2026
```

We recommend using an environment manager such as conda. Some additional non-python packages must be installed for full functionality. If using conda (recommended), simply run the following:

```sh
conda env create -f environment.yml
conda activate RP_2026
```

Otherwise, install these packages manually as-needed (see environment.yml for the list of packages and recommended versions).

Once the environment is set up, install the repository as a package.

```sh
pip install .
```

---

## Repository Contents

`notebook/`: Jupyter notebooks to reproduce each main and supplemental figure, named according to the figure number.
`RP_2026/`: Core functions used within notebooks
`scripts/`: Long scripts for generating variant indices or running variant calling with pupl and other tools

## License  
This project is licensed under the **BSD 2-Clause License**. See the [LICENSE](LICENSE) file for details.

---

For any issues or contributions, feel free to open a pull request or issue in this repository.
