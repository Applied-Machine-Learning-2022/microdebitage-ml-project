[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=8127865&assignment_repo_type=AssignmentRepo)
<!--
Name of your teams' final project
-->
# final-project
## [National Action Council for Minorities in Engineering(NACME)](https://www.nacme.org) Google Applied Machine Learning Intensive (AMLI) at the `PARTICIPATING_UNIVERSITY`

<!--
List all of the members who developed the project and
link to each members respective GitHub profile
-->
Developed by: 
- [Kimi Medina-Castellano](https://github.com/kimimedina) - `University of Kentucky`

## Description
<!--
Our task for this project was to create a model to accurately classify samples of microdebitage simulants into 5 tool production stages using numerical data provided by Dr. Phyllis Johnson. The level of desired accuracy was not defined, but we managed to reach 80% accuracy using Rodrigo's K-Nearest Neighbors (KNN) model. I tried two models: decision trees and random forest. 

### Complications
There were three extra columns that our obsidian data had whereas our chert data did not. These were Transparency, Curvature, and Angularity. Initially, I kept these columns for obsidian, added empty columns with the same names to the chert data sets, and filled the empty rows with zeros. Running my two models with this cleaned data gave me 80.1% and 58.7% accuracy in my decision trees and random forest model, respectively. After I tuned the hyperparameters for the decision tree model, my accuracy increased to 80.8%.

We double-checked with Dr. Johnson on whether or not chert microdebitage can be transparent and why there was no data for those three columns. She said they did not have those parameters turned on and should have had that data. This prompted the team to remove the three columns alltogether. I ran the updated dataset into both of my models and got about 37% for a basic and tuned model. With my low accuracy, we went with Rodrigo's KNN model.
-->

## Usage instructions
<!--
Give details on how to install fork and install your project. You can get all of the python dependencies for your project by typing `pip3 freeze requirements.txt` on the system that runs your project. Add the generated `requirements.txt` to this repo.
-->
1. Fork this repo
2. Change directories into your project
3. On the command line, type `pip3 install requirements.txt`
4. ....
