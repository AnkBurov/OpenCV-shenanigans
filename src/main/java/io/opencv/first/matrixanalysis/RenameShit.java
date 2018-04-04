package io.opencv.first.matrixanalysis;

import io.opencv.OpenCvBased;
import io.opencv.util.Matrices;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
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
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static io.opencv.first.matrixanalysis.SamePixelColorMatricsFinder.isMatrixOfSameColor;
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

    private static final int MATRIX_SIZE = 20;
    private static final int MAX_DEVIATION = 2;

    public static void main(String[] args) throws IOException {
        //        File file = new ClassPathResource("images/14_photoshoped.jpg").getFile();
        File file = new ClassPathResource("images/IMG_1365.JPG").getFile();
//        File file = new ClassPathResource("images/paint.jpg").getFile();
//                File file = new ClassPathResource("images/IMG_1374.JPG").getFile();

        //todo try sharpen image
        try (Matrices matrices = new Matrices()) {
            Mat orig = matrices.fromSupplier("orig", () -> Imgcodecs.imread(file.getAbsolutePath()));
            Mat orig_pic = matrices.fromSupplier("orig_pic", orig::clone);

            Mat gauss = matrices.newMatrix("gauss");
            Imgproc.GaussianBlur(orig, gauss, new Size(5, 5), 0.1);
            writeImage(gauss, file, "_gauss.jpg");

            Mat canny = matrices.newMatrix("canny");
            Imgproc.Canny(gauss, canny, 1, 10);
            writeImage(canny, file, "_canny.jpg");

            Mat erode = matrices.newMatrix("erode");
            Mat dilateKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                                                       new Size(4, 4));
            Imgproc.dilate(canny, erode, dilateKernel);
            writeImage(erode, file, "_erode.jpg");

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = matrices.newMatrix("hierarchy");
            Imgproc.findContours(erode, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
            System.out.println("Number of contours is: " + contours.size());

            Map<Rect, MatOfPoint> rectMatOfPointMap = new HashMap<>();
            for (int i = 0; i < contours.size(); i++) {
                MatOfPoint contour = contours.get(i);
                Rect rect = Imgproc.boundingRect(contour);
                if (rect.width > MATRIX_SIZE && rect.height > MATRIX_SIZE) {
                    if (hierarchy.get(0, i)[2] == -1) { // means contour has no children
                        rectMatOfPointMap.put(rect, contour);

                        Imgproc.rectangle(orig, rect.br(), rect.tl(), new Scalar(500));
                    }
                }
            }
            writeImage(orig, file, "_orig.jpg");

            System.out.println(rectMatOfPointMap.size());

            Mat contoursMat = matrices.fromSupplier("contours", () -> new Mat(new Size(orig.width(), orig.height()), CV_8UC3));
            Imgproc.drawContours(contoursMat, new ArrayList<>(rectMatOfPointMap.values()), -1, new Scalar(500));
            writeImage(contoursMat, file, "_contours.jpg");

            Map<Rect, MatOfPoint> contoursWithSameMatrices = detectSameMatrices(orig, rectMatOfPointMap);
            Imgcodecs.imwrite(file.getName() + "_analyzed.jpg", orig);

            List<Point> pointsOfSameColorContours = new ArrayList<>();

            for (Map.Entry<Rect, MatOfPoint> entry : contoursWithSameMatrices.entrySet()) {
                Rect rect = entry.getKey();
                MatOfPoint contour = entry.getValue();
                MatOfPoint2f curve = new MatOfPoint2f(contour.toArray());

                List<Point> pointsOfContourArea = new ArrayList<>();
                for (int row = rect.y; row <= rect.y + rect.height; row++) {
                    for (int column = rect.x; column <= rect.x + rect.width; column++) {
                        Point point = new Point(column, row);
                        boolean belongsToContour = Imgproc.pointPolygonTest(curve, point, false) >= 0;
                        if (belongsToContour) {
                            pointsOfContourArea.add(point);
                        }
                    }
                }
                System.out.println("Points inside of contour: " + pointsOfContourArea.size());

                List<Double> aggregatedPixelValues = new ArrayList<>();
                for (Point point : pointsOfContourArea) {
                    double[] pixel = orig_pic.get((int) point.y, (int) point.x);
                    Arrays.stream(pixel).forEach(aggregatedPixelValues::add);
                }

                //calculate standard variance
                Statistics statistics = new Statistics(aggregatedPixelValues.toArray(new Double[aggregatedPixelValues.size()]));
                double stdDev = statistics.getStdDev();
                if (stdDev < MAX_DEVIATION) {
                    pointsOfSameColorContours.addAll(pointsOfContourArea);
                }
            }

            for (Point sameColorPoint : pointsOfSameColorContours) {
                orig_pic.put((int) sameColorPoint.y, (int) sameColorPoint.x, 0, 255, 0);
            }
            Imgcodecs.imwrite(file.getName() + "_analyzed_similar.jpg", orig_pic);
        }
    }

    private static Map<Rect, MatOfPoint> detectSameMatrices(Mat orig, Map<Rect, MatOfPoint> rectMatOfPointMap) {
        Map<Rect, MatOfPoint> contoursWithSameMatrices = new HashMap<>();

        List<Map.Entry<Integer, Integer>> rowColumns = new ArrayList<>();

        // local cache of already analyzed matrices
        Map<Point, Boolean> analyzedMatricesByStartPoint = new HashMap<>();

        // for each rectangle shit
        for (Map.Entry<Rect, MatOfPoint> entry : rectMatOfPointMap.entrySet()) {
            Rect rect = entry.getKey();
            MatOfPoint contour = entry.getValue();

            // analyze each rectangle shit of interest for same matrices
            for (int row = rect.y; row <= rect.y + rect.height; row++) {
                for (int column = rect.x; column <= rect.x + rect.width; column++) {
                    analyzedMatricesByStartPoint.computeIfAbsent(new Point(column, row), point -> {
                        int matrixRow = (int) point.y;
                        int matrixColumn = (int) point.x;
                        Boolean matrixOfSameColor = isMatrixOfSameColor(orig, matrixRow, matrixColumn, MATRIX_SIZE);
                        if (matrixOfSameColor) {
//                            System.out.println("Matrics starts on row | column " + matrixRow + "|" + matrixColumn + " contains only same color");
                            rowColumns.add(new AbstractMap.SimpleEntry<>(matrixRow, matrixColumn));

                            contoursWithSameMatrices.put(rect, contour);
                        }
                        return matrixOfSameColor;
                    });
                }
            }
        }

        // rectangle the similar matrices and update the image
        for (Map.Entry<Integer, Integer> rowColumn : rowColumns) {
            Point startingPoing = new Point(rowColumn.getValue() - 1, rowColumn.getKey() - 1);
            Point endingPoint = new Point(rowColumn.getValue() - 1 + MATRIX_SIZE, rowColumn.getKey() - 1 + MATRIX_SIZE);
            Imgproc.rectangle(orig, startingPoing, endingPoint, new Scalar(0, 255, 0));
        }

        return contoursWithSameMatrices;
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
