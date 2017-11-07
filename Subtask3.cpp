///////////////////////////////                COMS30121               ///////////////////////////////
///////////////////////////////   Image Processing & Computer Vision   ///////////////////////////////
///////////////////////////////    "The Object Detection Challenge"    ///////////////////////////////
///////////////////////////////                SubTask 3               ///////////////////////////////
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

// FUNCTION HEADERS

void sobel(Mat &input, Mat &sobelOutput);

void showHough(Mat &input, Mat &output);

void violajones(Mat &input, vector <Rect> &output);

void houghCircle(Mat &input, vector <Vec3f> &output);

void houghLineIntersects(Mat &input, vector <Point2f> &intersectCoords);

bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r);

void CallBackFunc(int event, int x, int y, int flags, void *userdata);


// GLOBRAL VARIABLES

bool boxStarted = false;
int a = 0;
vector <Rect> grounds;
Mat dst;


// ** FUNCTION MAIN **
// function carries out various methods of object and line detection, and then combines them to return most likely position of dartboard(s)
int main(int argc, const char **argv) {
  //0. SETUP

  // declare parameters
  Mat src;
  Mat houghspace;
  vector <Rect> vjones;
  vector <Vec3f> circles;
  vector <Point2f> intersectCoords;
  int numboards = 0;
  const float union_area = 0.5;
  int truepos = 0;
  int falsepos = 0;
  int falseneg = 0;
  float precision, recall, F1;

  //read in image
  src = imread(argv[1], 1);
  dst = src;


  //1. COMPUTE AND DISPLAY HOUGH SPACE

  //show hough space of image (lines)
  showHough(src, houghspace);
  namedWindow("Hough Space", CV_WINDOW_AUTOSIZE);
  imshow("Hough Space", houghspace);
  waitKey(0);
  destroyAllWindows();


  //2. OBJECT DETECTION

  //carry out Viola-Jones object detection
  violajones(src, vjones);
  cout << "Viola Jones method detected " << vjones.size()
       << " potential dartboards." << endl;


/*		//display Viola-Jones output
		for( int i = 0; i < vjones.size(); i++ )
		{
			rectangle(dst, Point(vjones[i].x, vjones[i].y), Point(vjones[i].x + vjones[i].width, vjones[i].y + vjones[i].height), Scalar( 0, 255, 0 ), 2);
		}
*/

  //carry out hough circle detection
  houghCircle(src, circles);
  cout << "Hough Transform (circles) detected " << circles.size()
       << " potential dartboards." << endl;

/*		//display circles
		for( size_t i = 0; i < circles.size(); i++ )
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// circle center
			circle( dst, center, 3, Scalar(0,255,0), -1, 8, 0 );
			// circle outline
			circle( dst, center, radius, Scalar(0,0,255), 3, 8, 0 );
		}
		namedWindow( "Hough Circles", CV_WINDOW_AUTOSIZE );
		imshow( "Hough Circle", dst );
		waitKey(0);
*/

  //call function to generate matrix of line intersections
  houghLineIntersects(src, intersectCoords);


  //3. 'VOTE' FOR INDIVIDUAL VIOLA JONES BOXES IF CIRCLES OR LINE INTERSECTS ARE DISCOVERED IN THE SAME PLACE

  //create vjscore variable based on number of viola jones boards detected.
  //increment vjscore for each box if hough lines/circles also fall within that area
  vector<int> vjscore(vjones.size(), 0);

  //check viola jones boxes to see if circle centers lie within specified threshold
  int circle_overlaps = 0;
  float xthresh = 0.25;
  float ythresh = 0.25;
  for (int c = 0;
       c < circles.size(); c++) // c denotes the current circle being looked at
  {
    for (int r = c; r <
                    vjones.size(); r++) // r denotes the current rectangle being looked at
    {
      // check the x coord of circle centre point lies within vjones box
      if (circles[c][0] > (vjones[r].x + xthresh * vjones[r].width) &&
          circles[c][0] < (vjones[r].x + vjones[r].width * (1 - xthresh))) {
        // check the y coord of circle centre point lies within vjones box
        if (circles[c][1] > (vjones[r].y + ythresh * vjones[r].height) &&
            circles[c][1] < (vjones[r].y + vjones[r].height * (1 - ythresh))) {
          circle_overlaps = circle_overlaps + 1;
          vjscore[r] = vjscore[r] + 1;


        }
      }
    }
  }
  cout << "Viola Jones & Hough circles agree at " << circle_overlaps
       << " locations." << endl;


  // repeat above code to add vjones points for high line intersection areas
  int intersect_overlaps = 0;
  for (int i = 0; i <
                  intersectCoords.size(); i++) // i denotes the current line intersection point being looked at
  {
    for (int r = i; r <
                    vjones.size(); r++) // r denotes the current rectangle being looked at
    {
      // check the x coord of line intersect point lies within vjones box
      if (intersectCoords[i].x > (vjones[r].x + xthresh * vjones[r].width) &&
          intersectCoords[i].x <
          (vjones[r].x + vjones[r].width * (1 - xthresh))) {
        // check the y coord of line intersect point lies within vjones box
        if (intersectCoords[i].y > (vjones[r].y + ythresh * vjones[r].height) &&
            intersectCoords[i].y <
            (vjones[r].y + vjones[r].height * (1 - ythresh))) {
          intersect_overlaps += 1;
          vjscore[r] = vjscore[r] + 1;


        }
      }
    }
  }

  cout << "Hough Transform (Lines) detected " << intersectCoords.size()
       << " areas with high numbers of line intersections." << endl;
  cout << "Viola Jones & Hough lines agree at " << intersect_overlaps
       << " locations." << endl;

  /* uncomments to display vjscore to console
  cout << "vjscore: ";
  for(int r=0; r<vjscore.size(); r++)
  {
    cout << vjscore[r] << " ";
  }
  cout << endl;
  */

  //display the v-jones boxes with a score of 1 or higher
  vector <Rect> almostWinners;
  vector <Rect> winners;
  for (int r = 0; r < vjscore.size(); r++) {
    if (vjscore[r] > 0) {
      //rectangle(dst, Point(vjones[r].x, vjones[r].y), Point(vjones[r].x + vjones[r].width, vjones[r].y + vjones[r].height), Scalar( 0, 255, 0 ), 2);
      almostWinners.push_back(Rect(vjones[r]));
    }
  }

  // for/if loops to remove duplicate bounding boxes
  for (int r = 0; r < almostWinners.size(); r++) {
    // if there is only one almostWinner, just make it the Winner
    if (almostWinners.size() == 1) {
      winners.push_back(Rect(almostWinners[r]));
    } else {
      for (int r2 = r + 1; r2 < almostWinners.size(); r2++) {
        //if 2 boxes overlap, only return the smallest one
        if ((almostWinners[r] & almostWinners[r2]).area() > 0) {
          if (almostWinners[r2].width < almostWinners[r].width) {
            winners.push_back(Rect(almostWinners[r2]));
          } else {
            winners.push_back(Rect(almostWinners[r]));
          }
        } else //if no overlap just pass almostWinner to Winner
        {
          winners.push_back(Rect(almostWinners[r2]));
        }
      }
    }
  }

  //remove duplicates
  for (int w = 0; w < winners.size(); w++) {
    for (int w2 = w + 1; w2 < winners.size(); w2++) {
      if (winners[w] == winners[w2]) {
        winners.erase(winners.begin() + w2);
      }
    }
  }

  cout << "Estimated number of dartboards: " << winners.size() << endl;
  //display winning rectangles
  for (int r = 0; r < winners.size(); r++) {
    rectangle(dst, Point(winners[r].x, winners[r].y),
              Point(winners[r].x + winners[r].width,
                    winners[r].y + winners[r].height), Scalar(0, 255, 0), 2);
  }
  imshow("Detected Dart Boards", dst);
  waitKey(0);

  // 4. DEFINE GROUNDS & COMPUTE F1 SCORE 
  //ask for numboards
  cout << "How many boards should have been detected? ";
  cin >> numboards;
  grounds.resize(numboards);
  cout << "Please click top left and bottom right hand corner of each board:"
       << '\n';

  //Create a window
  //  namedWindow("My Window", 1);

  //set the callback function for any mouse event
  setMouseCallback("Detected Dart Boards", CallBackFunc, NULL);

  //show the image
  imshow("Detected Dart Boards", dst);

  //wait until user presses any key
  while (a < numboards) {
    waitKey(30);
  }

  //calculate truepos
  for (int i = 0; i < numboards; i++) {
    for (int j = 0; j < winners.size(); j++) {
      bool ground_intersects = ((grounds[i] & winners[j]).area() >
                                union_area * winners[j].area());
      truepos = truepos + ground_intersects;
      // Method to prevent ground truth being counted for multiple detected rectangles
      if (ground_intersects == 1)
        break;
    }
  }

  //calculate falseneg and falsepos & output them to console
  falseneg = numboards - truepos;
  falsepos = winners.size() - truepos;
  precision = (float) truepos / ((float) truepos + (float) falsepos);
  recall = (float) truepos / ((float) truepos + (float) falseneg);
  F1 = 2 * ((precision * recall) / (precision + recall));

  cout << "True Positives: " << truepos << endl;
  cout << "False Positives: " << falsepos << endl;
  cout << "False Negatives: " << falseneg << endl;
  cout << "Precision: " << precision << endl;
  cout << "Recall: " << recall << endl;
  cout << "F1-Score: " << F1 << endl;


  return 0;
}




////////////////////////// FUNCTION DEFINITIONS /////////////////////////////////////////


// ** FUNCTION SOBEL EDGE DETECTION **
void sobel(Mat &input, Mat &sobelOutput) {
  //Declare sobel threshold value
  int sobel_threshold = 60;

  //Declare matrices within scope of loop
  Mat sobelOutput_x;
  Mat sobelOutput_y;
  Mat sobelGradient;
  // intialise the output using the input
  sobelOutput.create(input.size(), CV_64F);
  sobelOutput_x.create(input.size(), input.type());
  sobelOutput_y.create(input.size(), input.type());
  sobelGradient.create(input.size(), CV_64F);

  // DECLARE SOBEL KERNELS (PARTIAL DERIVATIVES)
  Mat sobel_xmat = (Mat_<double>(3, 3) << -1, 0, 1,
      -2, 0, 2,
      -1, 0, 1);

  Mat sobel_ymat = (Mat_<double>(3, 3) << -1, -2, -1,
      0, 0, 0,
      1, 2, 1);

  // we need to create a padded version of the input
  // or there will be border effects
  int kernelRadiusX = (sobel_xmat.size[0] - 1) / 2;
  int kernelRadiusY = (sobel_xmat.size[1] - 1) / 2;

  Mat paddedInput;
  copyMakeBorder(input, paddedInput,
                 kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
                 BORDER_REPLICATE);

  // now we can do the sobelling

  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      double sum_x = 0.0;
      double sum_y = 0.0;
      for (int m = -kernelRadiusX; m <= kernelRadiusX; m++) {
        for (int n = -kernelRadiusY; n <= kernelRadiusY; n++) {
          // find the correct indices we are using
          int imagex = i + m + kernelRadiusX;
          int imagey = j + n + kernelRadiusY;
          int kernelx = m + kernelRadiusX;
          int kernely = n + kernelRadiusY;

          // get the values from the padded image and the kernel
          int imageval = (int) paddedInput.at<uchar>(imagex, imagey);
          double kernalval_x = sobel_xmat.at<double>(kernelx, kernely);
          double kernalval_y = sobel_ymat.at<double>(kernelx, kernely);

          // do the multiplication
          sum_x += imageval * kernalval_x;
          sum_y += imageval * kernalval_y;
        }
      }
      // set the output value as the sum of the convolution
      //normalise these values so they lie in the range of 0 and 255
      sobelOutput_x.at<uchar>(i, j) = (uchar)((sum_x / 8) + 128);
      sobelOutput_y.at<uchar>(i, j) = (uchar)((sum_y / 8) + 128);
      sobelOutput.at<double>(i, j) = std::sqrt(
          (sum_x * sum_x) + (sum_y * sum_y));
      if (sobelOutput.at<double>(i, j) > sobel_threshold) {
        sobelOutput.at<double>(i, j) = 200;
      } else sobelOutput.at<double>(i, j) = 0;
      sobelGradient.at<double>(i, j) = (
          atan2(sobelOutput_y.at<uchar>(i, j), sobelOutput_x.at<uchar>(i, j)) *
          180 / 3.14159265);

    }
  }

  //imwrite( "x.jpg", sobelOutput_x );
  //imwrite( "y.jpg", sobelOutput_y );
  //imwrite( "edge.jpg", sobelOutput );
  //imwrite( "gradient.jpg", sobelGradient );

}






// ***** FUNCTION SHOW HOUGH SPACE ****

void showHough(Mat &img_ori, Mat &normal_accum) {

  // 0. DECLARE VARIABLES
  //image matrices
  Mat img_blur;
  Mat img_grey;
  Mat img_edge;

  // deg->rad conversion factor
  const float DEG2RAD = 2 * 3.142 / 360;

  // blur image
  blur(img_ori, img_blur, Size(5, 5));

  // convert to greyscale
  cvtColor(img_blur, img_grey, CV_BGR2GRAY);

  // edge detect
  sobel(img_grey, img_edge);

  // show output of sobel function (thresholded magnitude gradient)
  //imshow("orig", img_ori);
  //imshow("2blur", img_blur);
  //imshow("3grey", img_grey);
  imshow("Sobel Edge Detection", img_edge);

  // count image rows and cols
  int w = img_edge.cols;
  int h = img_edge.rows;

  // set image centre coords
  double centre_x = w / 2;
  double centre_y = h / 2;

  // 2. SETUP ACCUMULATOR MATRIX
  //establish max r value (max distance from centre,
  //i.e. corner of image. use pythagorus to calculate)
  double rmax = ((sqrt(2.0) * (double) (h > w ? h : w) / 2.0));

  //declare accumulator matrix, rmax*2 rows, 180 columns
  //(one column per degree - i.e. check every angle for a line)
  int accum_h = round(rmax * 2);
  Mat accum = (Mat_<double>(accum_h, 180));

  //set accumulator to zero
  for (int r = 0; r < accum_h; r++) {
    for (int t = 0; t < 180; t++) {
      accum.at<double>(r, t) = 0;
    }
  }

  // 3. COMPUTE r FOR EVERY ANGLE AT EVERY DETECTED EDGE & POPULATE ACCUMULATOR MATRIX
  //loop through y
  for (int y = 0; y < h; y++) {
    //loop through x
    for (int x = 0; x < w; x++) {
      //check image for detected edge (ie pixel above threshold)
      if (img_edge.at<uchar>(y, x) > 0) {
        //for angles 0 -> 180
        for (int t = 0; t < 180; t++) {
          //calculates r for given angle
          double r = (((double) x - centre_x) * cos((double) t * DEG2RAD)) +
                     (((double) y - centre_y) * sin((double) t * DEG2RAD));



          //increment appropriate accumulator bin
          accum.at<double>(round(r + rmax), t)++;
        }
      }
    }
  }

  // 4. OUTPUT HOUGH SPACE IMAGE (VISUAL REPRESENTATION OF ACCUMULATOR)

  // first need to normalise for 0-255
  //create new matrix called normal_accum - use this for generating the hough space image
  resize(accum, normal_accum, Size(), 1, 1, INTER_CUBIC);

  // find max value in accumulator and call it accmax
  int accmax = 0;
  for (int r = 0; r < accum_h; r++) {
    for (int t = 0; t < 180; t++) {
      if (accum.at<double>(r, t) > accmax)
        accmax = accum.at<double>(r, t);
    }
  }

  // divide all values in normal_accum by accmax. This normalises all accumulator values from 0 to 1.
  for (int r = 0; r < accum_h; r++) {
    for (int t = 0; t < 180; t++) {
      normal_accum.at<double>(r, t) = (accum.at<double>(r, t)) / accmax;
    }
  }

  // resize so always fits on screen
  resize(normal_accum, normal_accum, Size(300, 600), 1, 1, INTER_CUBIC);

}




// *** FUNCTION VIOLA JONES OBJECT DETECTION *** 

void violajones(Mat &src, vector <Rect> &boards) {

  String cascade_name = "cascade.xml";
  CascadeClassifier cascade;

  // 1. Load the Strong Classifier in a structure called `Cascade'
  // Classifier e.g. a set of parameters that define a certain feature to be searched for.
  cascade.load(cascade_name);

  Mat src_gray;

  // 2. Prepare Image by turning it into Grayscale and normalising lighting
  cvtColor(src, src_gray, CV_BGR2GRAY);
  equalizeHist(src_gray, src_gray);

  // 3. Perform Viola-Jones Object Detection
  cascade.detectMultiScale(src_gray, boards, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE,
                           Size(50, 50), Size(500, 500));

}



// *** FUNCTION HOUGH CIRCLE DETECTION *** 

void houghCircle(Mat &src, vector <Vec3f> &circles) {
  /// Convert it to gray
  Mat src_gray;
  cvtColor(src, src_gray, CV_BGR2GRAY);

  /// Reduce the noise so we avoid false circle detection
  GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);

  /// Apply the Hough Transform to find the circles
  //HoughCircles(InputArray image, OutputArray circles, int method, double dp, double minDist, double param1=100, double param2=100, int minRadius=0, int maxRadius=0 )
  double lThreshold = 30;
  double uThreshold = 255;
  HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows / 4,
               uThreshold, lThreshold, 50, 600);
}




// **** FUNCTION HOUGHLINE INTERSECTIONS ****

void houghLineIntersects(Mat &src, vector <Point2f> &intersectCoords) {
  // 0. IMAGE PREP
  //declare params
  Mat dst, color_dst;

  // edge detection
  Canny(src, dst, 50, 400, 3);

  // convert to greyscale
  cvtColor(dst, color_dst, CV_GRAY2BGR);

  // 1. CARRY OUT HOUGH LINES
  // carries out hough line detection and stores saved lines to variable 'lines'
  vector <Vec4i> lines;
  HoughLinesP(dst, lines, 1, CV_PI / 360, 80, 30, 10);
  for (size_t i = 0; i < lines.size(); i++) {
    line(color_dst, Point(lines[i][0], lines[i][1]),
         Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 1, 8);
  }

  //uncomment to show image with detected lines overlayed
  //namedWindow( "Detected Lines", 1 );
  //imshow( "Detected Lines", color_dst );

  // 2. COMPUTE INTERSECT POINTS FOR DETECTED LINES
  //first declare variables
  Point2f intersect;
  vector <Point2f> intersects;
  float w = src.cols;
  float h = src.rows;

  // scroll through vector of lines, compare it with itself and record where each line intersects
  for (int i = 0; i < lines.size(); i++) {
    for (int j = i; j < lines.size(); j++) {
      intersection(Point(lines[i][0], lines[i][1]),
                   Point(lines[i][2], lines[i][3]),
                   Point(lines[j][0], lines[j][1]),
                   Point(lines[j][2], lines[j][3]), intersect);

      // exclude intersects that happen outside the borders of the image
      if (intersect.x > 0 && intersect.x < w && intersect.y > 0 &&
          intersect.y < h)
        // write intersect coords to intersects vector
        intersects.push_back(Point2f(intersect));
    }
  }

  // 3. Determine size of intersect_count matrix for an increment of (pixelStep) pixels
  // Set incremental step in terms of pixels for cell divisions
  int pixelStep = 10;

  // divides source image into sections by pixelstep
  Mat intersectCount = (Mat_<double>(round((src.rows) / pixelStep),
                                     round((src.cols) / pixelStep)));

  for (int i = 0; i < (intersectCount.rows); i++) {
    //loop through x of image matrix
    for (int j = 0; j < (intersectCount.cols); j++) {
      intersectCount.at<double>(i, j) = 0;
    }
  }

  // 4. LOOP THROUGH EACH CELL OF NEWLY DIVIDED IMAGE MATRIX AND COUNT EACH POINT OF LINE INTERSECTS THAT FALL WITHIN RANGE OF CELL
  //loop through y of image matrix
  for (int i = 0; i < (intersectCount.rows); i++) {
    //loop through x of image matrix
    for (int j = 0; j < (intersectCount.cols); j++) {
      //loop through vector
      for (int v = 0; v < intersects.size(); v++) {
        //check image for detected edge (ie pixel above threshold)
        if ((intersects[v].x > j * pixelStep) &&
            (intersects[v].x < j * pixelStep + pixelStep) &&
            (intersects[v].y > i * pixelStep) &&
            (intersects[v].y < i * pixelStep + pixelStep)) {
          intersectCount.at<double>(i, j)++;
        }
      }
    }
  }

  /* //uncomment to print intersect count matrix to console
  cout << "source width = " << src.cols << endl;
  cout << "source height = " << src.rows << endl;
  cout << "intersectCount width = " << intersectCount.cols << endl;
  cout << "intersectCount height = " << intersectCount.rows << endl;

  cout << "Line Intersection Matrix:"<< endl;
  for(int r=0; r<intersectCount.rows; r++)
    {
      for (int c=0; c<intersectCount.cols; c++)
      {
        cout.width(4);cout<< intersectCount.at<double>(r, c);
      }
    cout<<endl;
    }
  cout<<endl<<endl; */

  //check intersect counter for max value
  int max_intersects = 0;

  // check where max number of line intersections occurs
  Point2f max_intersects_coords;
  for (int r = 0; r < intersectCount.rows; r++) {
    for (int c = 0; c < intersectCount.cols; c++) {
      if (intersectCount.at<double>(r, c) > max_intersects) {
        max_intersects = intersectCount.at<double>(r, c);
        max_intersects_coords.y = r * pixelStep;
        max_intersects_coords.x = c * pixelStep;
      }
    }
  }


  //cout << "Hough Transform (Lines) detected a maximum number of " << max_intersects << " intersections at " << max_intersects_coords << endl;

  // threshold matrix to store coordinates of high line intersection regions in a vector of points
  float intersect_thresh = 0.3;
  for (int r = 0; r < intersectCount.rows; r++) {
    for (int c = 0; c < intersectCount.cols; c++) {
      if (intersectCount.at<double>(r, c) >
          (intersect_thresh * max_intersects)) {
        intersectCoords.push_back(Point2f((c * pixelStep), (r * pixelStep)));
      }
    }
  }

}


// **** FUNCTION TO FIND INTERSECTION POINT OF 2 LINES ***

bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r) {
  Point2f x = o2 - o1;
  Point2f d1 = p1 - o1;
  Point2f d2 = p2 - o2;

  float cross = d1.x * d2.y - d1.y * d2.x;
  if (abs(cross) < /*EPS*/1e-8)
    return false;

  double t1 = (x.x * d2.y - x.y * d2.x) / cross;
  r = o1 + d1 * t1;
  return true;
}


// **** CallBackFunc - handles mouse clicks ****
void CallBackFunc(int event, int x, int y, int flags, void *userdata) {
  // determines whether 1st or 2nd click of box
  if (event == EVENT_LBUTTONDOWN) {
    if (boxStarted == false) {
      grounds[a].x = x;
      grounds[a].y = y;
      boxStarted = true;
    } else {
      grounds[a].width = x - grounds[a].x;
      grounds[a].height = y - grounds[a].y;
      boxStarted = false;
      rectangle(dst, Point(grounds[a].x, grounds[a].y),
                Point(grounds[a].x + grounds[a].width,
                      grounds[a].y + grounds[a].height), Scalar(255, 0, 0), 2);
      imshow("Detected Dart Boards", dst);
      imwrite("detected.jpg", dst);
      cout << "Board " << a + 1 << " ground defined." << endl;
      a = a + 1;
    }
  }

}
