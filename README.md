[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=8127865&assignment_repo_type=AssignmentRepo)
<!--
Name of your teams' final project
-->
# final-project
## [National Action Council for Minorities in Engineering(NACME)](https://www.nacme.org) Google Applied Machine Learning Intensive (AMLI) at the University Of Kentucky


<!--
List all of the members who developed the project and
link to each members respective GitHub profile
-->
Developed by: 
- [Kimi Medina-Castellano](https://github.com/kimimedina) - `University of Kentucky`
- [Rodrigo Aguilar Barrios](https://github.com/Rodrigox30) - `University of California, Berkeley` 
- [Luke Taylor](https://github.com/LukeTaylor1) - `University of Kentucky` 
- [Jose Cruz](https://github.com/Resoj) - `University of Kentucky`

## Objective
To determine the production stages of ancient stone tools using automated measurements taken of experimental microdebitage.

## Goals
To create a model that will accurately determine the production stage that an ancient stone tool was undergoing based off the features of the microdebitage left behind.

## The Dataset
We received data for two different stone tools in the form of Excel files via Box (file sharing platform). The Excel files contain features about the physical properties of the microdebitage and each file contains data for each stage of the tool. In preparing the data, we removed features that were the exact same value across all the datasets – values that would make no difference – and removed features that immediately seemed irrelevant to our objective.

## Description
The quoted paragraphs below are taken directly from the proposal presented to us by our mentors. 
> "Understanding how stone tools were made and used is essential for interpreting the social, economic, and political processes of ancient societies. Stone is also the most durable of all materials used by Pre-Columbian societies in North America and is therefore one of the most common heritage artifact types recovered from archaeological sites. Because the tools themselves are often transported from the places they were made, archaeologists must analyze the debitage, or leftover stone pieces, that are knocked off when making a stone tool. The smallest of these artifacts, microdebitage, measures < 6mm and tends to be less vulnerable to post-depositional movement from human activities, such as cleaning and sweeping, and natural processes, such as bioturbation and erosion. Because of this, archaeologists can study the spatial organization of microdebitage to better understand where stone tools were being made at archaeological sites. However, the study of microdebitage is hindered by tedious and time-consuming methods. For example, an experienced analyst will invest up to 10 hours to separate microdebitage from an archaeological soil matrix sample, following water-screening or flotation and then manual inspection through microscopy. Recently, a novel method using automated dynamic image analysis was implemented to quantify shape variables of microdebitage and differentiate it from natural particles in archaeological soil samples in a fraction of the time (Johnson et al., 2021), paving the way for efficient analysis of microdebitage to address a range of heritage science research questions.

> This Google-NACME AMLI summer project envisions the application of machine learning techniques to automated measurements taken of experimental microdebitage, in order to test whether different stone tool production stages can be classified."

## Team Workflow
Each group member tried a different approach to develop an effective model. Below are links to each member's READ.ME file that details and explains their linear path for this project. [Link](https://github.com/Applied-Machine-Learning-2022/final-project-the-rock-group-uk/blob/main/July%2028%20-%20Classifying%20Microdebitage%20-%20Final%20Project.pdf) to slides from group presentation
* [Kimi](https://github.com/Applied-Machine-Learning-2022/final-project-the-rock-group-uk/blob/bbd838bf069fd07990fa04373e4fc64074db671d/README-DecisionTrees_RandomForest.md): [Merging Data](https://github.com/Applied-Machine-Learning-2022/final-project-the-rock-group-uk/blob/ad4689c6836f959d342926b97656cd3bbcaa5966/Kimi_Final_Project_AllData.ipynb), [Decision Tree / Random Forest (All Data)](https://github.com/Applied-Machine-Learning-2022/final-project-the-rock-group-uk/blob/09c670e26a88696ed198b198747a1fe082c86c1d/Decision_Tree_Random_Forest_FinalProject.ipynb), [Chert Decision Tree](https://github.com/Applied-Machine-Learning-2022/final-project-the-rock-group-uk/blob/491a5fb447ce8486959d6766b72b051aacb202d7/Chert_Model__Decision_Trees.ipynb)
* Rodrigo [TensorFlow](https://github.com/Applied-Machine-Learning-2022/final-project-the-rock-group-uk/blob/main/README-KNeighborsClassifier.md) [KNeighborsClassifier](https://github.com/Applied-Machine-Learning-2022/final-project-the-rock-group-uk/blob/main/README-KNeighborsClassifier.md)
* [Luke](https://github.com/Applied-Machine-Learning-2022/final-project-the-rock-group-uk/blob/main/LukeReadMe.md)
* [Jose](https://github.com/Applied-Machine-Learning-2022/final-project-the-rock-group-uk/blob/main/Obsidian%20Decision%20Tree%20Model.md)
