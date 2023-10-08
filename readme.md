# Exploring the Land Mines Dataset

## Introduction
This is just an exercise in understanding ML-models trained on the "land mines" dataset
that can be found [here](https://archive.ics.uci.edu/dataset/763/land+mines-1). If you are
interested, I definitely recommend reading [[1]](#1), which is the publication belonging to this dataset.
What makes this dataset stand out is that it is quite low dimensional (each sample contains only 
three different features, one of which is categorical). This should make it easier for us to
understand what is going on when trying to classify the type of land mine a measurement belongs to.

## Installation / Following along
If you want to follow along (or play with the code), simply
1. Use git to clone this repo.
2. Install the requirements.txt running `pip install -r requirements.txt` in the root folder of the repo.
3. Run the different scripts in the `mine_classification` folder as they are mentioned in the text.

## References
<a id="1">[1]</a> 
C. Yilmaz, H. T. Kahraman and S. Söyler, "Passive Mine Detection and Classification Method
Based on Hybrid Model" IEEE Access, 2018, [URL](https://ieeexplore.ieee.org/ielx7/6287639/8274985/08443331.pdf?tp=&arnumber=8443331&isnumber=8274985&ref=),
DOI: 10.1109/ACCESS.2018.2866538


