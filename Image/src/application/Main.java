package application;
	
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import javafx.application.Application;
import javafx.stage.Stage;
import javafx.scene.Scene;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;


public class Main extends Application {
	@Override
	public void start(Stage primaryStage) {
		try {
			// Load the OpenCV library
			System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
			
			
			// Read the input image
	        String inputImagePath = "D:\\lena.jpg";
	        Mat inputImage = Imgcodecs.imread(inputImagePath); // Read the image using Imgcodecs
	        
	        
	        // Check if the image was loaded successfully
	        if (inputImage.empty()) {
	            System.out.println("Failed to load the input image.");
	            return;
	        }
	        
	        Mat grayImage = new Mat();
	        Mat binaryImage = new Mat();
	        
	        int originalWidth = inputImage.cols(); // Get the width of the input image
	        int originalHeight = inputImage.rows(); // Get the height of the input image
	        
	     // Create a window to display the Original image
	        HighGui.namedWindow("Original Image", HighGui.WINDOW_NORMAL); // Create a window with the specified name and normal window mode
	        Mat subsampledImage = inputImage.clone(); // Create a copy of the input image to perform subsampling
	        HighGui.imshow("Original Image", subsampledImage);
	     //   HighGui.waitKey(); ----->> When running this sentence, I need to close the original window in order for the next window to open
	        
	        
	        convert2Gray(subsampledImage, grayImage); //call convert to gray level method
	        convert2Binary(grayImage, binaryImage); //call convert to binary level method        
	        resize(originalWidth, originalHeight, grayImage); // call resize method to resize gray image 
	        flipping(grayImage);
	        blurring(grayImage);
	        negative(grayImage);
	        
	        histogramBasedSearch(grayImage);
	        imageCalculation(grayImage);
	        Entropy(grayImage);
	        contrast(grayImage);
	        normalizedHistogram(grayImage, primaryStage);
	        cumulativeHistogram(grayImage);
	        HighGui.waitKey(0);	     	        
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		launch(args);
	}
	
	
	/** method to convert original image from RGB level to gray level */
	private static void convert2Gray(Mat originalImage, Mat grayImage) {
        Imgproc.cvtColor(originalImage, grayImage, Imgproc.COLOR_BGR2GRAY); // convert the image color level
        HighGui.imshow("Grayscale Image", grayImage); // display the image on the screen
      //HighGui.waitKey(); // Wait to display the image
	}
	
	
	/** method to convert gray image from gray level to binary level */
	private static void convert2Binary(Mat grayImage, Mat binaryImage) {
        Imgproc.threshold(grayImage, binaryImage, 128, 255, Imgproc.THRESH_BINARY); // convert the image color level
        HighGui.imshow("Binary Image", binaryImage); // display the image on the screen
    //    HighGui.waitKey(); // Wait to display the image
	}
	
	
	/** method to resize gray image from original size to desired size*/
	private static void resize(int originalWidth, int originalHeight, Mat grayImage) {
		Mat resizeGray = grayImage.clone();
		
		// Set the desired size for the gray image
        int desiredWidth = 256; 
        int desiredHeight = 256; 

        // Subsample the image gradually from its original size to the desired size
        double scaleFactorX = (double) desiredWidth / originalWidth; 
        double scaleFactorY = (double) desiredHeight / originalHeight; 

        // Calculate the current width and height based on the scale factor
        int currentWidth = (int) (originalHeight * scaleFactorX); 
        int currentHeight = (int) (originalHeight * scaleFactorY);
        
        // Resize the image to the current size
        Imgproc.resize(resizeGray, resizeGray, new Size(currentWidth, currentHeight)); // Resize the gray image

        // Display the resized image in the window
        HighGui.imshow("Resize Gray Image", resizeGray); 
     //   HighGui.waitKey(0); 
	}
	
	/** method to calculate mean and standard deviation of the gray image*/
	private static void imageCalculation(Mat grayImage) {
		int rows = grayImage.rows();
        int cols = grayImage.cols();

        /** calculating the mean */
        // loops to sum the intensity of each pixel in the image
        double sumIntensity = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double intensity = grayImage.get(i, j)[0];
                sumIntensity += intensity;
            }
        }
        // Calculate the mean intensity value
        int totalPixels = rows * cols;
        double meanIntensity = sumIntensity / totalPixels;
        System.out.println("The mean of the grayscale image: " + meanIntensity);
        
        
        /** calculating standard deviation*/
        double sum = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double intensity = grayImage.get(i, j)[0];
                double opr = Math.pow((intensity - meanIntensity), 2);
                sum += opr;
            }
        }
        double standardDev = Math.sqrt((sum / (totalPixels)));
        System.out.println("The standard deviation of the grayscale image: " + standardDev);
	}
	
	/**method to calculate cumulative histogram and of the gray image show it in scene*/
	private static void cumulativeHistogram(Mat gray) {
		int rows = gray.rows();
        int cols = gray.cols();
        
		// Calculate the histogram of pixel intensities
        int[] histogram = new int[256];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int intensity = (int) gray.get(i, j)[0];
                histogram[intensity]++;
            }
        }
        // Calculate the total number of pixels
        int totalPixels = rows * cols;
        
        // Calculate the normalized histogram
        double[] normalizedHistogram = new double[256];
        for (int i = 0; i < histogram.length; i++) {
            normalizedHistogram[i] = (double) histogram[i] / totalPixels;
        }
        
        // Calculate the cumulative histogram
        double[] cumulativeHistogram = new double[256];
        cumulativeHistogram[0] = normalizedHistogram[0];
        for (int i = 1; i < 256; i++) {
            cumulativeHistogram[i] = normalizedHistogram[i] + cumulativeHistogram[i - 1];
        }
        
        CategoryAxis xAxis = new CategoryAxis();
        NumberAxis yAxis = new NumberAxis();

        // Create the histogram chart
        BarChart<String, Number> histogramChart = new BarChart<>(xAxis, yAxis);
        histogramChart.setTitle("Cumulative Histogram");
        xAxis.setLabel("Intensity");
        yAxis.setLabel("Frequency");

        // Create a series for the histogram data
        XYChart.Series<String, Number> series = new XYChart.Series<>();
        series.setName("histogram");

        // Add the histogram data to the series
        for (int i = 0; i < 256; i++) {
            series.getData().add(new XYChart.Data<>(String.valueOf(i), cumulativeHistogram[i]));
        }

        // Add the series to the histogram chart
        histogramChart.getData().add(series);
        
     // Create the scene and add the chart to it
        Scene scene = new Scene(histogramChart, 500, 400);
        Stage stage = new Stage();

        // Set the stage title and scene, then show the stage
        stage.setTitle("Histogram");
        stage.setScene(scene);
        stage.show();
	}
	
	/**method to calculate entropy of the gray image*/
	private static void Entropy(Mat gray) {
		int rows = gray.rows();
        int cols = gray.cols();
        
		// Calculate the histogram of pixel intensities
        int[] histogram = new int[256];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int intensity = (int) gray.get(i, j)[0];
                histogram[intensity]++;
            }
        }
        // Calculate the total number of pixels
        int totalPixels = rows * cols;
 
        // Calculate the probability distribution of pixel intensities
        double[] probability = new double[256];
        for (int i = 0; i < histogram.length; i++) {
            probability[i] = (double) histogram[i] / totalPixels;
        }

        // Calculate the entropy
        double entropy = 0;
        for (int i = 0; i < probability.length; i++) {
            if (probability[i] > 0) {
            	double opr = probability[i] * (Math.log(probability[i]) / Math.log(2));
                entropy -= opr;
            }
        }
        System.out.println("Entropy of pixel intensities: " + entropy);
	}
	
	/**method to calculate normalized histogram and of the gray image show it in scene*/
	private static double [] normalizedHistogram (Mat gray, Stage primaryStage) {
		int rows = gray.rows();
        int cols = gray.cols();
        
		// Calculate the histogram of pixel intensities
        int[] histogram = new int[256];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int intensity = (int) gray.get(i, j)[0];
                histogram[intensity]++;
            }
        }
        // Calculate the total number of pixels
        int totalPixels = rows * cols;
        
        // Calculate the normalized histogram
        double[] normalizedHistogram = new double[256];
        for (int i = 0; i < histogram.length; i++) {
            normalizedHistogram[i] = (double) histogram[i] / totalPixels;
        }
        
        CategoryAxis xAxis = new CategoryAxis();
        NumberAxis yAxis = new NumberAxis();

        // Create the histogram chart
        BarChart<String, Number> histogramChart = new BarChart<>(xAxis, yAxis);
        histogramChart.setTitle("Normalized Histogram");
        xAxis.setLabel("Intensity");
        yAxis.setLabel("Frequency");

        // Create a series for the histogram data
        XYChart.Series<String, Number> series = new XYChart.Series<>();
        series.setName("histogram");

        // Add the histogram data to the series
        for (int i = 0; i < 256; i++) {
            series.getData().add(new XYChart.Data<>(String.valueOf(i), normalizedHistogram[i]));
        }

        // Add the series to the histogram chart
        histogramChart.getData().add(series);
        
        // Create the scene and add the chart to it
        Scene scene = new Scene(histogramChart, 500, 400);

        // Set the stage title and scene, then show the stage
        primaryStage.setTitle("Histogram");
        primaryStage.setScene(scene);
        primaryStage.show();
        return normalizedHistogram;
	}

	/** method to do Contrast Enhancement in the gray scale image*/ 
	private static void contrast(Mat grayImage) {
		Mat contraseGray = grayImage.clone();
		
		// Get the image dimensions
        int rows = contraseGray.rows();
        int cols = contraseGray.cols();

        // Compute the minimum and maximum pixel intensities
        double minIntensity = Double.MAX_VALUE;
        double maxIntensity = Double.MIN_VALUE;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double intensity = contraseGray.get(i, j)[0];
                if (intensity < minIntensity) {
                    minIntensity = intensity;
                }
                if (intensity > maxIntensity) {
                    maxIntensity = intensity;
                }
            }
        }

        // Apply contrast enhancement
        double intensityRange = maxIntensity - minIntensity;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double intensity = contraseGray.get(i, j)[0];
                double newIntensity = ((intensity - minIntensity) / intensityRange) * 0.5 * 255;
                contraseGray.put(i, j, newIntensity);
            }
        }
        // Save the resulting enhanced image
        String contrastImagePath = "C:\\Users\\hp\\OneDrive\\Desktop\\Malak\\ImageProcessingAssignment1\\C.jpg";
        Imgcodecs.imwrite(contrastImagePath, contraseGray);
        HighGui.imshow("Contrast Image", contraseGray); 
	}

	/** method to do flipping in gray image*/
	private static void flipping(Mat grayImage) {
		Mat flippedImage = grayImage.clone();
		
		int rows = grayImage.rows();
        int cols = grayImage.cols();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double intensity = grayImage.get(i, cols - j - 1)[0];
                flippedImage.put(i, j, intensity);
            }
        }
        
     // Save the resulting enhanced image
        String flippedImagePath = "C:\\Users\\hp\\OneDrive\\Desktop\\Malak\\ImageProcessingAssignment1\\flipped.jpg";
        Imgcodecs.imwrite(flippedImagePath, flippedImage);
        HighGui.imshow("Fipped Image", flippedImage); 
	}
	
	/** method to do blurring in gray image using average filter*/
	private static void blurring(Mat grayImage) {
		Mat bluredImage = grayImage.clone();
		
		int rows = grayImage.rows();
        int cols = grayImage.cols();
        int filterSize = 3; // Size of the filter 

        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                double sumIntensity = 0;
                // Compute the sum of intensities in the neighborhood
                for (int m = -1; m <= 1; m++) {
                    for (int n = -1; n <= 1; n++) {
                        sumIntensity += grayImage.get(i + m, j + n)[0];
                    }
                }
                // Calculate the average intensity
                double averageIntensity = sumIntensity / (filterSize * filterSize);
                // Update the pixel intensity in the filtered image
                bluredImage.put(i, j, averageIntensity);
            }
        }
        
     // Save the resulting enhanced image
        String filterdImagePath = "C:\\Users\\hp\\OneDrive\\Desktop\\Malak\\ImageProcessingAssignment1\\blured.jpg";
        Imgcodecs.imwrite(filterdImagePath, bluredImage);
        HighGui.imshow("blured Image", bluredImage);
	}

	/** method to do negative image in gray image*/
	private static void negative(Mat grayImage) {
		Mat negativeImage = grayImage.clone();
		
		int rows = grayImage.rows();
        int cols = grayImage.cols();

        // Iterate over each pixel of the image
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double intensity = grayImage.get(i, j)[0];
                double negativeIntensity = 255 - intensity;
                negativeImage.put(i, j, negativeIntensity);
            }
        }
        // Save the resulting enhanced image
        String negativeImagePath = "C:\\Users\\hp\\OneDrive\\Desktop\\Malak\\ImageProcessingAssignment1\\negative.jpg";
        Imgcodecs.imwrite(negativeImagePath, negativeImage);
        HighGui.imshow("Negative Image", negativeImage);
	}
	
	/** method to do crop in gray image*/
	private static Mat crop(Mat grayImage, int x, int y, int h, int w) {
		Mat croppedImage = new Mat(h, w, grayImage.type());
		
		for (int i = y; i < y + h; i++) {
	        for (int j = x; j < x + w; j++) {
	            double intensity = grayImage.get(i, j)[0];
	            croppedImage.put(i - y, j - x, intensity);
	        }
	    }
		// Save the resulting enhanced image
        String croppedImagePath = "C:\\Users\\hp\\OneDrive\\Desktop\\Malak\\ImageProcessingAssignment1\\negative.jpg";
        Imgcodecs.imwrite(croppedImagePath, croppedImage);
        HighGui.imshow("Cropped Image", croppedImage);
        return croppedImage;
	}

	
	private static void histogramBasedSearch (Mat grayImage) {
		Stage stage = new Stage();
		Mat extractedStrip = crop(grayImage, 230, 230, 60, 130);
		
		// Compute the histogram of the current strip
        double[] originalHistogram = normalizedHistogram(grayImage, stage);
        double[] extractedHistogram = normalizedHistogram(extractedStrip, stage);
        
        compareHistograms(originalHistogram, extractedHistogram);
        normalizedHistogram(grayImage, stage);
        normalizedHistogram(extractedStrip, stage);
	}
	
	/** method to compare between original histogram and cropped image histogram*/
	private static void compareHistograms(double[] hist1, double[] hist2) {
        double similarityScore = 0;
        for (int i = 0; i < hist1.length; i++) {
            similarityScore += Math.sqrt(hist1[i] * hist2[i]);
        }
        System.out.println("Similarity score is: "+(1 - similarityScore)); 
    }
}

