Info on compiling and running each of the Subtask programs.



--------- SUBTASK 1 ---------

1.  Compile the program from the terminal using g++ or similar c++ compiler. e.g. type "g++ Subtask1.cpp `pkg-config --cflags --libs opencv` -o Subtask1" into the terminal to compile the program to an executable named "Subtask1".

2.  Ensure the file 'frontalface.xml' is located within the same directory as the compiled executable.

3.  Run the executable from the terminal with a single dart image as the input parameter, e.g. type "./Subtask1 dart1.jpg" into the terminal.

4.  The program will ask how many faces should have been detected in the image. Enter the correct number.

5.  The program will then ask for the ground truths to be input. This is done by clicking the top left and then bottom right hand corner of each face. Once this is done, the program will save the output 'detected.jpg' image and display some relevant information in the terminal.



--------- SUBTASK 2 ---------

This is compiled and run in the same way as Subtask1. Please ensure the file 'dartcascade.xml' is saved in the same directory as the executable.



--------- SUBTASK 3 ---------

1.  Compile the program from the terminal using g++ or similar c++ compiler. e.g. type "g++ Subtask3.cpp `pkg-config --cflags --libs opencv` -o Subtask3" into the terminal to compile the program to an executable named "Subtask3".

2.  Ensure the file 'cascade.xml' is located within the same directory as the compiled executable.

3.  Run the executable from the terminal with a single dart image as the input parameter, e.g. type "./Subtask3 dart1.jpg" into the terminal.

4.  The program will display the thresholded gradient magnitude image and a visual representation of the hough transformation. Hit any key to proceed. 
	NB - do NOT close images by clicking the 'x' in the upper left corner.

5.  The program will then display the output of the hough line detection algorithm, and the output image with detected dart boards bounded by boxes. Press any key to proceed.
	NB - do NOT close images by clicking the 'x' in the upper left corner.

6.  The terminal will then prompt for the correct number of dart boards to be input. Type the correct number and hit Enter.

7.  The program will then ask for the ground truths to be input. This is done by clicking the top left and then bottom right hand corner of each board. Once this is done, the program will save the output 'detected.jpg' image and display some relevant information in the terminal.



--------- SUBTASK 4 ---------

This is compiled and run in the same way as Subtask3. Please ensure the file 'Merge2.xml' is saved in the same directory as the executable.

