package io.opencv.first.matrixanalysis;

import io.opencv.OpenCvBased;
import io.opencv.util.Matrices;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_GRAYSCALE;
import static org.opencv.imgproc.Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C;
import static org.opencv.imgproc.Imgproc.ADAPTIVE_THRESH_MEAN_C;
import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_NONE;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2HSV;
import static org.opencv.imgproc.Imgproc.MORPH_OPEN;
import static org.opencv.imgproc.Imgproc.RETR_EXTERNAL;
import static org.opencv.imgproc.Imgproc.RETR_TREE;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;

public class RenameShit extends OpenCvBased {

    public static void main(String[] args) throws IOException {
//        File file = new ClassPathResource("images/14_photoshoped.jpg").getFile();
//        File file = new ClassPathResource("images/paint.jpg").getFile();
        File file = new ClassPathResource("images/IMG_1374.JPG").getFile();

        //todo try sharpen image
        try (Matrices matrices = new Matrices()) {
            Mat orig = matrices.fromSupplier("orig", () -> Imgcodecs.imread(file.getAbsolutePath()));

            Mat gauss = matrices.newMatrix("gauss");
            Imgproc.GaussianBlur(orig, gauss, new Size(5, 5), 0.1);
            writeImage(gauss, file, "_gauss.jpg");

            Mat canny = matrices.newMatrix("canny");
            Imgproc.Canny(gauss, canny, 1, 10);
            writeImage(canny, file, "_canny.jpg");

            Mat erode = matrices.newMatrix("erode");
            Imgproc.dilate(canny, erode, matrices.newMatrix("kernel"));
            writeImage(erode, file, "_erode.jpg");

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = matrices.newMatrix("hierarchy");
            Imgproc.findContours(erode, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
            System.out.println("Number of contours is: " + contours.size());

            Map<Rect, MatOfPoint> rectMatOfPointMap = new HashMap<>();
            for (int i = 0; i < contours.size(); i++) {
                MatOfPoint contour = contours.get(i);
                Rect rect = Imgproc.boundingRect(contour);
                if (rect.width > 20 && rect.height > 20) {
                    if (hierarchy.get(0, i)[2] == -1) { // means contour has no children
                        rectMatOfPointMap.put(rect, contour);

                        Imgproc.rectangle(orig, rect.br(), rect.tl(), new Scalar(500));
                    }
                }
            }

            System.out.println(rectMatOfPointMap.size());

            Mat contoursMat = matrices.fromSupplier("contours", () -> new Mat(new Size(orig.width(), orig.height()), CV_8UC3));
            Imgproc.drawContours(contoursMat, new ArrayList<>(rectMatOfPointMap.values()), -1, new Scalar(500));
            writeImage(contoursMat, file, "_contours.jpg");

            writeImage(orig, file, "_orig.jpg");



            /*// for each rectangle shit
            for (Rect rect : rectsOfInterest) {

                // analyze each rectangle shit of interest for same matrices
                for (int row = rect.y; row <= rect.y + rect.height; row++) {
                    for (int column = rect.x; column <= rect.x + rect.width; column++) {

                        for (int i = 0; i < curves.size(); i++) {
                            MatOfPoint2f curve = curves.get(0);
                            Imgproc.pointPolygonTest(curve, new Point(column, row), false);
                        }
                    }
                }
            }*/
        }
    }

    /*//thresholding
            Mat threshold = matrices.newMatrix("threshold");
            Imgproc.adaptiveThreshold(orig, threshold, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
            writeImage(threshold, file, "_threshold.jpg");*/

    /*
            //https://stackoverflow.com/questions/18627970/adaptive-threshold-with-blurry-image
            Mat hsv = matrices.newMatrix("hsv");
            Imgproc.cvtColor(orig, hsv, COLOR_BGR2HSV);
            writeImage(hsv, file, "_hsv.jpg");*/

    private static void writeImage(Mat orig, File file, String name) {
        Imgcodecs.imwrite(file.getName() + name, orig);
    }
}
