///////////////////////////////                COMS30121               ///////////////////////////////
///////////////////////////////   Image Processing & Computer Vision   ///////////////////////////////
///////////////////////////////    "The Object Detection Challenge"    ///////////////////////////////
///////////////////////////////                SubTask 2               ///////////////////////////////
///////////////////////////////            J BLYTH, J BOSTOCK          ///////////////////////////////

//HEADER INCLUSION
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

//FUNCTION HEADERS
void detectAndDisplay( Mat frame );
void CallBackFunc(int event, int x, int y, int flags, void* userdata);

//GLOBAL VARIABLES
String cascade_name = "cascade.xml";
CascadeClassifier cascade;

Mat frame;
int tlx, tly, brx, bry, numboards;
bool boxStarted = false;

int a = 0;
const float union_area = 0.5;

vector<Rect> boards;
vector<Rect> grounds;



/** @function main */
int main(int argc, const char** argv)
{

int truepos = 0;
int falsepos = 0;
int falseneg = 0;
float precision, recall, F1;

  // 1. Read image from file 
     frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

     //if fail to read the image
     if ( ! frame.data ) 
     { 
          cout << "Error loading the image" << endl;
          return -1; 
     }

  // 2. Load the Strong Classifier in a structure called `Cascade'
	// Classifier e.g. a set of parameters that define a certain feature to be searched for.
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  // 3. Detect boards and Display Result
	detectAndDisplay( frame );

  // 4. Define Ground Rules 
	//ask for numboards
	cout << "How many boards should have been detected? ";
	cin >> numboards;
	grounds.resize(numboards);
	cout << "Please click top left and bottom right hand corner of each dartboard:" << '\n';

	     //Create a window
	     namedWindow("My Window", 1);

	     //set the callback function for any mouse event
	     setMouseCallback("My Window", CallBackFunc, NULL);

	     //show the image
	     imshow("My Window", frame);

	    //wait until user presses any key
	     while( a < numboards )
		{ waitKey(30);
		}
	
	//calculate truepos
	for (int i=0; i < numboards; i++)
	{
		for (int j=0; j < boards.size(); j++)
		{
			bool intersects = ((grounds[i] & boards[j]).area() > union_area*boards[j].area());
			truepos = truepos + intersects;
			// Method to prevent ground truth being counted for multiple detected rectangles
			if (intersects == 1)
				break;
		}
	}
	
	//calculate falseneg and falsepos & output them to console
	falseneg = numboards - truepos;
	falsepos = boards.size() - truepos;
	precision = (float)truepos/((float)truepos + (float)falsepos);
	recall = (float)truepos/((float)truepos + (float)falseneg);
	F1 = 2*((precision * recall)/(precision + recall));

	cout << "True Positives: " << truepos << endl;
	cout << "False Positives: " << falsepos << endl;
	cout << "False Negatives: " << falseneg << endl;
	cout << "Precision: " << precision << endl;
	cout << "Recall: " << recall << endl;
	cout << "F1-Score: " << F1 << endl;

  // 5. Save Result Image
	imwrite( "detected.jpg", frame );

     return 0;

}



/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{

	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, boards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of boards found
	cout << "Found " << boards.size() << " boards." << endl;

       // 4. Draw box around boards found
	for( int i = 0; i < boards.size(); i++ )
	{
		rectangle(frame, Point(boards[i].x, boards[i].y), Point(boards[i].x + boards[i].width, boards[i].y + boards[i].height), Scalar( 0, 255, 0 ), 2);
	}

}



/** @function CallBackFunc - handles mouse clicks */
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	     // determines whether 1st or 2nd click of box
	     if  ( event == EVENT_LBUTTONDOWN )
	     {
		if (boxStarted == false)
		{
			grounds[a].x = x;
			grounds[a].y = y;
			boxStarted = true;
		}
		else
		{
			grounds[a].width = x - grounds[a].x;
			grounds[a].height = y - grounds[a].y;
			boxStarted = false;
			rectangle(frame, Point(grounds[a].x, grounds[a].y), Point(grounds[a].x + grounds[a].width, grounds[a].y + grounds[a].height), Scalar( 255, 0, 0 ), 2);
			imshow("My Window", frame);
			cout << "board " << a+1 << " ground defined." << endl;
			a = a + 1;
		}
     	     }

}
