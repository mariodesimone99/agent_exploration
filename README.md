# Frontier Exploration Algorithm
This Project is an implementation of frontier exploration algorithm, the latter is a detailed guide about the structure of codes and files.  

## Libraries
First in the file **`requirements.txt`** there is a list of the library used

-**`numpy`** to handle the grid-map  
-**`pygame`** to get a visual representation in a game-like fashion  
-**`matplotlib`** to make plot

## Configuration Parameters
The project is made of a single python file, at the top there are the import libraries, definitions of some global variables used through the code, after that there are some configuration settings parameters:

-**`WINDOW_HEIGHT`** and **`WINDOW_WIDTH`** are the dimensions of the pygame instance and of the grid map  
-**`BLOCK_SIZE`** is the dimension of each square tile in the pyagem instance  
-**`EXIT_THRESHOLD`** is a number between 0 and 1, representing the amount of total cell to visit before exiting the game  
-**`KNOWN_THRESHOLD`** and **`OCCUPIED_THRESHOLD`** are probabilities representing of much of the total cell is prior-knowledge of the agent and the number of the obstacle of the map  
-**`RANDOM_MOVE`** is the probability of making a random (legal) move instead of the optimal one  
-**`ALPHA`** is a scale factor for the cost of making moves  
-**`FOV`** is a parameter representing the length of the field-of-view in the 4 dimensions (up, left, down and right)

## Code Specs and Notes
The rest of the code is made of 2 classes definitions:

-**`Map`**: representing the concept of map and its drawing  
-**`MapHandler`**: an entity with the goal of handling both the agent and the real map across all of the possible stages from the initialization to the update and visualization

and a **`main`** function which instantiate and loop-update the map until **`EXIT_THRESHOLD`** is reached.  
Some notes about algorithm and code:

-the algorithm can be slowed down by decommenting and changing the parameter into the **`pygame.delay`** function found in the **`draw`** function in class **`Map`**  
-to avoid the fall into loop there have been used 2 strategies: I avoid to make the opposite move of a move if it already been visited (for example I avoid a pattern like left-right-left) and the use of random moves (both in direction and length) once in a while (see **`RANDOM_MOVE`** parameter)  
-the algorithm at least in principle is able to explore all kind of maps but the time may become unfeasible due to the large amount of random moves required  
-during execution lots of images file are temporarily saved into a tmp folder, in order to get a **`.gif`** file, this folder is removed right after the creation of the file  
-due to the both the visualization and save of images files, the instances can become laggy and slow especially if **`pygame.delay`** function is decommented  
-as **`gain`** function I used the number of new cells found along an axis due to the field-of-view, while as **`cost`** I used the number of tile to traverse to reach a position, if no obstacle are encountered along the fov axis, then the agent reach the new frontier otherwise policy to stop before or take into account other moves have been implemented (see strategies above)  
-a plot of the number of cell explored for each epoch is made and saved at the end of the execution
