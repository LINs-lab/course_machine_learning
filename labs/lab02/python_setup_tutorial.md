# Setup Guide for Coding Machine Learning and Optimization Methods

In order to implement the algorithms seen in class and work on the projects, we'll be using Python notebooks. This first lab will serve as an introduction to the Python language, the environment we are going to be using, and how to do basic vector and matrix manipulations.

## The environment

We recommend an easy-to-use online environment (Google Colab) or a local installation (Anaconda).

### Online: Google Colab

Google colab provides a similar environment to Noto, with additional access to GPUs (not needed in the first few labs). Note that you need to take care of storing the files permanently yourself (storing on Google Drive, downloading to a local machine, ...).

You can open any exercise by adapting `XY` with the lab number and `PATH_TO_FILE` with the path of the notebook you wish to open:  
`https://github.com/LINs-lab/course_machine_learning/blob/main/labs/labXY/PATH_TO_FILE`

E.g. for the numpy introduction `npprimer.ipynb`:  
[`http://colab.research.google.com/github/LINs-lab/course_machine_learning/blob/main/labs/lab02/npprimer.ipynb`](http://colab.research.google.com/github/LINs-lab/course_machine_learning/blob/main/labs/lab02/npprimer.ipynb)

You can also create an empty notebook by following this [link](https://colab.research.google.com/) and clicking `"NEW NOTEBOOK"`, or you can open a pre-existing notebook (.ipynb extension) by selecting the `Upload` tab.

If for some reason you've opened a python2 notebook, you can switch to python3 by going in `Runtime > Change runtime type`. There you can also add a GPU to your notebook if necessary.

### Offline: Python distribution Anaconda

If you prefer to have an environment locally on your computer, you can use the [Anaconda](https://www.anaconda.com/) distribution to run Python 3, as it is easy to install and comes with most packages we will need. To install Anaconda, go to [the download page](https://www.anaconda.com/download/) and get the Python installer for your OS - make sure to use the newer version 3.x, not 2.x. Follow the instructions of the installer and you're done.
> **Warning!** The installer will ask you if you want to add Anaconda to your path. Your default answer should be yes, unless you have specific reasons not to want this.


### Development Environment

During the course, we will use [**Jupyter Notebooks**](http://jupyter.org/), which is a great tool for exploratory and interactive programming and in particular for data analysis. Notebooks are browser-based, and you start a notebook on your localhost by typing `jupyter notebook` in the console. Notebooks are already available by default by Anaconda. The interface is pretty intuitive, but there are a few tweaks and shortcuts that will make your life easier, which we'll detail in the next section. You can of course ask any of the TAs for help on using the Notebooks.

### The Notebook System

For additional resources on how the notebook system works, we recommend

* [The Jupyter notebook beginner's guide](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/index.html)
* [The official documentation](http://jupyter-notebook.readthedocs.io/en/latest/index.html)

#### Examples

We provide you with an example of a notebook for [this second lab](https://github.com/LINs-lab/course_machine_learning/tree/main/labs/lab02), but if you want to see some more examples already, feel free to take a look at

* The introductory notebooks available at [Try Jupyter](https://try.jupyter.org/). It spawns an instance of the Jupyter Notebook, which won't save any of your changes.
  *Note: it might not be available if their server is under too much load.*
* [A gallery of interesting IPython Notebooks](https://github.com/jupyter/jupyter/wiki#a-gallery-of-interesting-jupyter-notebooks) by the Ipython Notebook team

#### Tips & Tricks

There are a few handy commands that you should start every notebook with


	# Plot figures in the notebook (instead of a new window)
	%matplotlib notebook

	# Automatically reload modules
	%load_ext autoreload
	%autoreload 2

	# The usual imports
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd

#### Keyboard shortcuts

* Adding cells
	* `a` adds an empty cell above the selected one,
	* `b` adds it below.
* Running code
	* `Enter` enters the edition mode of the currently selected cell.
	* `Shift-Enter` runs the current cell and goes to the next one.
	* `Ctrl-Enter` runs the current cell and leave it selected.
* Autocompletion (Jupyter notebook)
  * `Tab` pops up the Autocompletion when you are in the middle of writing a function call/class name and shows the arguments of the function being called when used after an opening parenthesis.
  * `Shift-Tab` pops up the help/documentation of the function its used on
* Autocompletion (Google Colab)
  * `Ctrl-Space` pops up the Autocompletion when you are in the middle of writing a function call/class name and shows the arguments of the function being called when used after an opening parenthesis.
  * Clicking on a function name and hovering over it will pop up the help/documentation for that function.

* For a complete list of shortcuts, go to `help > keyboard shortcuts`

## Python

We will be working in Python. If you already have been introduced to Python, feel free to skip this section. If you come from another background, you might want to take some tutorials in addition to this lab in the next week to feel comfortable with it. You do not need to become an expert in Python, but you should be comfortable with the general syntax, some of the idiosyncrasies of Python and know how to do basic vector and matrix algebra. For the last part, we will be using NumPy, a library we will introduce later.

For a nice introduction to Python, you should take a look at [the Python tutorial](https://docs.python.org/3/tutorial/index.html). Here are some reading recommendations:

* Skim through Sections 1-3 to get an idea of the Python syntax if you never used it
* Pay a little more attention to Section 4, especially

	* Section 4.2 on for loops, as they behave like `foreach` by default, which may be disturbing if you are more accustomed to coding in lower level languages.
	* Section 4.7 on functions, default argument values and named arguments, as they are a real pleasure to use (compared to traditional, order-based arguments) once you are used to it.
* Section 5 on Data Structures, especially how to use Lists, Dictionnaries and Tuples if you have not used a language with those concepts before
* You can keep Sections 6-9 on Modules, IO, Exceptions and Objects for later - when you know you will be needing it.
* Section 10 on the standard library and [the standard library index](https://docs.python.org/3/library/index.html) are worth a quick scroll to see what's available.
* Do not bother with Sections 11-16 for now.

Here are some additional resources on Python:

* [Python's standard library reference](https://docs.python.org/3/library/index.html)
* [Debugging and profiling](https://docs.python.org/3/library/debug.html)
* If you want to, some exercises are available at [learnpython.org](http://www.learnpython.org/)


## NumPy and Vector Calculations

Our `npprimer.ipynb` notebook as part of the first lab has some useful commands and exercises to help you get started with NumPy.

We recommend [this list of small exercises](https://www.machinelearningplus.com/101-numpy-exercises-python/) to get started with NumPy arrays etc.

If you are familiar with Matlab, a good starting point is [this guide](https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). Be careful that we will use way more the `array` data structure compared to the `matrix` data structure.

A good and probably more complete reference is [this one](https://sites.engineering.ucsb.edu/~shell/che210d/numpy.pdf).


### Installation FAQ

> **Other shell.** If you are using another shell (e.g. zsh on Mac OSX), after installing Anaconda you still need to add the installed software to your path, that is to add it to the correct profile of your shell. To do so, run the following commands in your terminal ` touch ~/.bash_profile; open ~/.bash_profile`. It will open your bash profile where you'll see the line that was added by the Python installer. Copy it. Then ` touch ~/.zshrc; open ~/.zshrc`, that will open the profile for zsh, you can paste the line at the bottom of the file.

> **Alternative Python IDEs.** While we recommend plain Jupyter Notebooks, if you are more comfortable using a more traditional IDE, you can give [**PyCharm**](https://www.jetbrains.com/pycharm/) a try. Your WestLake email gives you access to the free educational version. You should keep this option in mind if you need a full fledged debugger to find a nasty bug.

And of course, as a third alternative, you can always use a [decent text editor](https://www.sublimetext.com/) and run your code from the console or any plugin. Keep in mind that the TAs might not be able to help you with your setup if you go down this path.

## Download the exercises content & basic Git tutorial

### Simplest: no git

You can click on the green `code` button on the main [page](https://github.com/LINs-lab/course_machine_learning) and select `Download ZIP`. We advise against this method as you might have to re-download the repository every time some new content is posted there.

### Still simple: using GitHub Desktop

GitHub Desktop simplifies the interaction with a GitHub repository by providing a simple GUI, check it out [here](https://desktop.github.com/). GitHub Desktop supports most 'real' `git` usecases such as the ones described below.  

### More advanced: Git via command line

`Git` is the most widely used version control system. It's a tool to share and help you collaboratively develop and maintain code. GitHub is a Git repository hosting service, it allows you to create GitHub repositories you can interact with using `git`.

`Git` is typically used via the terminal. To install Git, follow this [link](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

**Download repository.** Once Git is installed you can pull a GitHub repository using: `git clone <url.git>`, e.g. `git clone https://github.com/LINs-lab/course_machine_learning.git`.

**Collaborative coding.** A standard workflow when working as a group is to implement features through pull-requests (PR):
* You do not want to break the master/main branch by mistake, so you start by creating and moving to a new branch: `git checkout -b <name-of-my-new-branch>`
* Now you're safe on your new branch, the modifications you're making won't affect the master branch. You can modify/create new files as if you were on the master branch e.g.

```bash
# let's say we modify file.py here
git status # check the status of the files git is tracking
git add file.py
git commit -m "some message clearly explaining the modification"
```
* Once you are done doing all the modifications you want you can push to your new branch: `git push origin <name-of-my-new-branch>`. If you prefer a GUI instead of the command line, all above steps can also be performed in [GitHub desktop](https://desktop.github.com/), or your IDE of choice.
* Finally you can open a PR from the GitHub user interface. Typically you would ask your colleagues to review your PR and accept it or ask for modifications.
* Once your PR is accepted and merged, do not forget to switch back to master: `git checkout master` and pull your approved changes `git pull origin master`.

## Additional References

[A good Python and NumPy Tutorial from Stanford.](https://github.com/kuleshov/cs228-material/blob/master/tutorials/python/cs228-python-tutorial.ipynb)
